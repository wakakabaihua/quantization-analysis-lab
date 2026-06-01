"""
AWQ (Activation-Aware Weight Quantization) backend.

Implements the AWQ algorithm from scratch using only PyTorch — no external
packages required beyond those already in the project's dependencies.

Reference:
    Lin et al. "AWQ: Activation-aware Weight Quantization for LLM Compression
    and Acceleration." MLSys 2024. https://arxiv.org/abs/2306.00978

Key idea:
    Not all weight channels contribute equally to the output.  Channels
    paired with large input activations have a disproportionate effect:
    their quantization error is amplified by the activation magnitude.

    AWQ searches for a per-input-channel scale vector s ∈ ℝ^{d_in} such that:
        • Weight channels are scaled UP:   W' = W × diag(s)
        • Input activations are scaled DOWN: x' = x / s
        (so W' @ x' = W × s @ (x / s) = W @ x — output is preserved)

    Quantizing W' instead of W reduces the effective quantization error on
    important (large-activation) channels by giving them more representation
    range in the quantized grid.

Scale search (per Linear layer):
    For alpha ∈ {0, 1/n_grid, …, 1}:
        s(alpha) = |x̄|^alpha          where x̄ = per-channel mean activation magnitude
        W_scaled  = W × diag(s(alpha))
        W_q       = Q(W_scaled)         (fake-quantize)
        error     = ‖X @ Wᵀ − X @ (W_q / diag(s))ᵀ‖²_F
    Choose alpha* = argmin error

Inference module (AWQLinear):
    Stores W_q = Q(W × s*) and s*.
    Forward: y = W_q @ (x / s*) + bias

Key differences from FakeQuantBackend:
    • AWQ scales important weight channels to use more of the quantization
      range, reducing effective error on influential neurons.
    • The scale is a function of observed activations, not just weight statistics.

Key differences from GPTQBackend:
    • GPTQ corrects quantization error via weight updates; AWQ prevents error
      by improving the quantization grid alignment.
    • AWQ produces a modified inference module (AWQLinear); GPTQ does not.

No additional dependencies required.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import QuantBackend
from ..kernels import (
    pack_int4,
    quantize_to_uint4,
    compute_groupwise_qparams,
    int4_dequant_gemm,
)


# ---------------------------------------------------------------------------
# AWQLinear — inference module (real INT4 packed storage)
# ---------------------------------------------------------------------------

class AWQLinear(nn.Module):
    """
    Linear layer with AWQ activation-aware scaling.

    Storage modes
    ~~~~~~~~~~~~~
    INT4 packed (``num_bits == 4``):
        qweight   [out_features, in_features // 8]   int32  8 INT4 per int32
        scales    [n_groups, out_features]            float32
        qzeros    [n_groups, out_features]            int32
        awq_scale [in_features]                       float32  per-channel AWQ scale

        On CUDA with Triton: Triton kernel dequantises qweight → fp16, then
        cuBLAS handles the GEMM.  On CPU: pure-PyTorch fallback.

    Float32 fallback (``num_bits != 4``):
        weight_fp32  [out_features, in_features]   float32  fake-quantised weight
        awq_scale    [in_features]                  float32  per-channel AWQ scale

        Uses ``F.linear`` directly.  Provides identical semantics with higher
        precision for INT8 / other bit widths that do not yet have a packed
        kernel.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        awq_scale:    torch.Tensor,
        *,
        qweight:      Optional[torch.Tensor] = None,   # INT4 packed mode
        scales:       Optional[torch.Tensor] = None,
        qzeros:       Optional[torch.Tensor] = None,
        weight_fp32:  Optional[torch.Tensor] = None,   # float32 fallback mode
        group_size:   int                    = 128,
        bias:         Optional[torch.Tensor] = None,
        num_bits:     int                    = 4,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.group_size   = group_size
        self.num_bits     = num_bits
        self._use_packed  = (qweight is not None)

        self.register_buffer("awq_scale", awq_scale)  # [K] float32

        if self._use_packed:
            self.register_buffer("qweight", qweight)   # [N, K//8] int32
            self.register_buffer("scales",  scales)    # [G, N]    float32
            self.register_buffer("qzeros",  qzeros)    # [G, N]    int32
        else:
            self.register_buffer("weight_fp32", weight_fp32)  # [N, K] float32

        if bias is not None:
            self.bias = nn.Parameter(bias.clone(), requires_grad=False)
        else:
            self.bias = None

    # ------------------------------------------------------------------
    # Factory: construct from the AWQ-scaled float32 weight
    # ------------------------------------------------------------------

    @classmethod
    def from_float(
        cls,
        in_features:  int,
        out_features: int,
        W_scaled:     torch.Tensor,
        awq_scale:    torch.Tensor,
        symmetric:    bool,
        num_bits:     int                    = 4,
        group_size:   int                    = 128,
        bias:         Optional[torch.Tensor] = None,
    ) -> "AWQLinear":
        """
        Build an AWQLinear from the AWQ-scaled float32 weight W × diag(s*).

        For ``num_bits == 4``: packs weights to INT4 and stores in the packed
        format.  For other bit widths: computes the fake-quantised float32
        weight and falls back to ``F.linear``.

        Args:
            W_scaled:   [N, K] float32, W × diag(s*).
            awq_scale:  [K]   float32, per-input-channel AWQ scale s*.
            symmetric:  Whether symmetric quantisation was used.
            num_bits:   Bit-width.  Only INT4 uses the packed kernel.
            group_size: Quantisation group size.  Pass ``in_features`` for
                        per-channel (single-group) quantisation.
            bias:       Optional [N] float32 bias.
        """
        K = W_scaled.shape[1]
        gs = min(group_size, K)

        scales, qzeros = compute_groupwise_qparams(
            W_scaled, group_size=gs, num_bits=num_bits, symmetric=symmetric
        )
        W_uint = quantize_to_uint4(W_scaled, scales, qzeros, group_size=gs, num_bits=num_bits)

        if num_bits == 4:
            qweight = pack_int4(W_uint)
            return cls(
                in_features, out_features, awq_scale,
                qweight=qweight, scales=scales, qzeros=qzeros,
                group_size=gs, bias=bias, num_bits=num_bits,
            )
        else:
            # Float32 fallback: dequantise the integer grid back to float32.
            K_dim = W_scaled.shape[1]
            group_idx    = torch.arange(K_dim, device=W_uint.device) // gs
            scales_exp   = scales[group_idx, :].T.float()   # [N, K]
            qzeros_exp   = qzeros[group_idx, :].T.float()   # [N, K]
            weight_fp32  = (W_uint.float() - qzeros_exp) * scales_exp
            return cls(
                in_features, out_features, awq_scale,
                weight_fp32=weight_fp32.to(W_scaled.dtype),
                group_size=gs, bias=bias, num_bits=num_bits,
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled = x / self.awq_scale.to(x.device)
        if self._use_packed:
            return int4_dequant_gemm(
                x_scaled, self.qweight, self.scales, self.qzeros,
                group_size=self.group_size, bias=self.bias,
            )
        return F.linear(x_scaled, self.weight_fp32, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"num_bits={self.num_bits}, "
            f"group_size={self.group_size}"
        )


# ---------------------------------------------------------------------------
# Internal quantization helpers
# ---------------------------------------------------------------------------

def _compute_qparams(
    W: torch.Tensor,
    num_bits: int,
    symmetric: bool,
    per_channel: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-row (or global) scale and zero_point for W [d_out, d_in]."""
    d_out = W.shape[0]
    if symmetric:
        quant_max = 2 ** (num_bits - 1) - 1
        if per_channel:
            abs_max = W.abs().amax(dim=1).clamp(min=1e-8)    # [d_out]
        else:
            abs_max = W.abs().amax().clamp(min=1e-8).expand(d_out)
        scale = abs_max / quant_max
        zp = torch.zeros(d_out, dtype=torch.int32, device=W.device)
    else:
        quant_max = 2 ** num_bits - 1
        if per_channel:
            w_min = W.amin(dim=1)
            w_max = W.amax(dim=1)
        else:
            w_min = W.amin().expand(d_out)
            w_max = W.amax().expand(d_out)
        w_min = torch.minimum(w_min, torch.zeros_like(w_min))
        w_max = torch.maximum(w_max, torch.zeros_like(w_max))
        scale = ((w_max - w_min) / quant_max).clamp(min=1e-8)
        zp = (-w_min / scale).round().to(torch.int32)
    return scale, zp   # [d_out]


def _batch_fake_quant(
    W: torch.Tensor,
    scale: torch.Tensor,
    zp: torch.Tensor,
    num_bits: int,
    symmetric: bool,
) -> torch.Tensor:
    """Fake-quantize weight matrix W using per-row scale/zp [d_out]."""
    s = scale.to(W.dtype).unsqueeze(1)          # [d_out, 1]
    z = zp.to(W.dtype).unsqueeze(1)
    if symmetric:
        qmax = 2 ** (num_bits - 1) - 1
        qmin = -qmax - 1
        q = (W / s).round().clamp(qmin, qmax)
        return q * s
    else:
        qmax = 2 ** num_bits - 1
        q = (W / s + z).round().clamp(0, qmax)
        return (q - z) * s


def _find_awq_scale(
    W: torch.Tensor,
    X: torch.Tensor,
    num_bits: int,
    symmetric: bool,
    per_channel: bool,
    n_grid: int = 20,
) -> torch.Tensor:
    """
    Grid search for the optimal per-input-channel AWQ scale s.

    For each alpha in a uniform grid [0, 1]:
        s(alpha) = |x̄|^alpha
        W'       = W × diag(s)
        W_q      = Q(W')
        W_eff    = W_q / diag(s)        ← effective weight at inference
        error    = ‖X @ Wᵀ − X @ W_effᵀ‖²_F

    Returns the s that achieves the minimum error.

    Args:
        W:           [d_out, d_in] weight matrix (float32).
        X:           [n, d_in] calibration activations (float32).
        num_bits:    Quantization bit-width.
        symmetric:   Symmetric or asymmetric quantization.
        per_channel: Per-output-channel or per-tensor weight quantization.
        n_grid:      Number of alpha values to try (including 0 and 1).

    Returns:
        s: [d_in] optimal per-input-channel scale.
    """
    # Per-channel activation magnitude (mean |x| per input dimension)
    x_scale = X.abs().mean(dim=0).clamp(min=1e-8)    # [d_in]

    # Reference output (unquantized)
    ref = X @ W.T                                     # [n, d_out]

    best_error = float("inf")
    best_s = torch.ones_like(x_scale)

    for step in range(n_grid + 1):
        alpha = step / n_grid

        # AWQ scale: s_i = |x̄_i|^alpha
        s = x_scale.pow(alpha).clamp(min=1e-8)        # [d_in]

        # Scale weight columns by s: W' = W × diag(s)
        W_scaled = W * s.unsqueeze(0)                 # [d_out, d_in]

        # Fake-quantize scaled weights
        scale_q, zp_q = _compute_qparams(W_scaled, num_bits, symmetric, per_channel)
        W_q = _batch_fake_quant(W_scaled, scale_q, zp_q, num_bits, symmetric)

        # Effective weight at inference: W_eff = W_q / diag(s)
        W_eff = W_q / s.unsqueeze(0)                  # [d_out, d_in]

        # Quantized output error
        out_q = X @ W_eff.T                           # [n, d_out]
        error = (ref - out_q).pow(2).mean().item()

        if error < best_error:
            best_error = error
            best_s = s.clone()

    return best_s   # [d_in]


def _set_module(model: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


# ---------------------------------------------------------------------------
# AWQBackend
# ---------------------------------------------------------------------------

class AWQBackend(QuantBackend):
    """
    AWQ post-training quantization backend (pure PyTorch, no extra packages).

    Searches for per-input-channel scales that minimise quantization error for
    important weight channels (those with large input activations).

    Calibration:
        calibrate() attaches forward hooks to all Linear layers, runs the
        calibration function, and stores the flattened input activations.

    Conversion:
        For each Linear layer, convert() runs the AWQ scale search, computes
        the optimal scale s*, and replaces the layer with an AWQLinear that
        stores W_q = Q(W × s*) and applies x / s* at inference time.

    Fallback:
        Layers not reached during calibration fall back to standard MinMax
        fake quantization with scale=1 (no AWQ scaling).
    """

    def __init__(
        self,
        num_bits: int = 4,
        symmetric: bool = True,
        per_channel: bool = True,
        weight_only: bool = True,
        calibration_method: str = "minmax",
        calibration_percentile: float = 99.99,
        n_grid: int = 20,
    ) -> None:
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.weight_only = weight_only
        self.calibration_method = calibration_method
        self.calibration_percentile = calibration_percentile
        self.n_grid = n_grid

        self._activations: Dict[str, torch.Tensor] = {}
        self._hooks: List[Any] = []

    @property
    def name(self) -> str:
        return "awq"

    # ------------------------------------------------------------------
    # QuantBackend interface
    # ------------------------------------------------------------------

    def calibrate(
        self,
        model: nn.Module,
        calibration_fn: Callable[[], None],
    ) -> None:
        """
        Collect input activations for every Linear layer.

        Activations are flattened to [n, in_features] for the scale search.
        """
        buffers: Dict[str, List[torch.Tensor]] = {}

        def make_hook(layer_name: str) -> Callable:
            def hook(module: nn.Module, inp: Tuple, out: torch.Tensor) -> None:
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                buffers.setdefault(layer_name, []).append(x)
            return hook

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_forward_hook(make_hook(name))
                self._hooks.append(h)

        model.eval()
        with torch.no_grad():
            calibration_fn()

        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        for name, chunks in buffers.items():
            self._activations[name] = torch.cat(chunks, dim=0)

    def convert(self, model: nn.Module) -> nn.Module:
        """
        Return a deep copy of the model with AWQ-quantized Linear layers.

        Each nn.Linear is replaced with an AWQLinear that stores:
            weight    = Q(W × diag(s*))
            awq_scale = s*

        The original model is not mutated.
        """
        model_q = copy.deepcopy(model)

        for name, module in list(model_q.named_modules()):
            if not isinstance(module, nn.Linear):
                continue

            W = module.weight.data.float()
            X = self._activations.get(name)
            bias = module.bias.data.clone() if module.bias is not None else None

            if X is not None:
                X = X.to(W.device)
                # Find optimal AWQ scale
                s = _find_awq_scale(
                    W, X, self.num_bits, self.symmetric, self.per_channel, self.n_grid
                )   # [d_in]
                W_scaled = W * s.unsqueeze(0)           # [d_out, d_in]
            else:
                # Fallback: no calibration data, AWQ scale = 1
                s = torch.ones(module.in_features, device=W.device, dtype=W.dtype)
                W_scaled = W

            awq_layer = AWQLinear.from_float(
                in_features=module.in_features,
                out_features=module.out_features,
                W_scaled=W_scaled,
                awq_scale=s.to(module.weight.dtype),
                symmetric=self.symmetric,
                num_bits=self.num_bits,
                group_size=module.in_features,  # per-channel (single group)
                bias=bias,
            )
            _set_module(model_q, name, awq_layer)

        return model_q
