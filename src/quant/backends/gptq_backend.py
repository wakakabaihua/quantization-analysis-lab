"""
GPTQ quantization backend.

Implements the GPTQ algorithm from scratch using only PyTorch — no external
packages required beyond those already in the project's dependencies.

Reference:
    Frantar et al. "GPTQ: Accurate Post-Training Quantization for Generative
    Pre-trained Transformers." ICLR 2023. https://arxiv.org/abs/2210.17323

Key idea:
    Standard PTQ (FakeQuantBackend) quantizes each weight element independently
    of the others. GPTQ instead uses second-order information — the Hessian of
    the per-layer squared output error w.r.t. the weights — to propagate each
    column's quantization error to all subsequent columns, partially compensating
    for it. This yields significantly lower output error at 4-bit, especially
    for weight-only quantization.

Algorithm (per Linear layer):
    1. Collect input activations X ∈ ℝ^{n × d_in} from calibration data.
    2. Compute Hessian:  H = (2/n) · Xᵀ X + λ·I    (λ = damp_percent · mean(diag(H)))
    3. Compute H⁻¹ via Cholesky decomposition.
    4. Process columns of W in blocks of size `blocksize`:
        a. Quantize column i:   q_i = Q(w_i)
        b. Compute error:       e_i = (w_i − q_i) / H⁻¹_{ii}
        c. Update remaining columns in block:
               w_{j>i} -= e_i · H⁻¹_{i, j>i}
    5. Propagate accumulated block errors to all later columns:
               W[:, i2:] -= E @ H⁻¹[i1:i2, i2:]

Key difference from FakeQuantBackend:
    FakeQuantBackend minimises per-element quantisation error independently.
    GPTQ minimises the *layer output* error jointly across all columns.

Supports:
    - Arbitrary bit-widths (most useful at INT4).
    - Symmetric and asymmetric quantization.
    - Per-channel and per-tensor weight quantization.
    - Weight-only mode (activations computed in float).

No additional dependencies required.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import QuantBackend
from ..kernels import pack_int4, int4_dequant_gemm


# ---------------------------------------------------------------------------
# Internal quantization helpers
# ---------------------------------------------------------------------------

def _compute_qparams(
    W: torch.Tensor,
    num_bits: int,
    symmetric: bool,
    per_channel: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute quantization scale and zero_point for weight matrix W.

    Returns:
        scale:      [d_out] (one value per output channel, or broadcast scalar).
        zero_point: [d_out] int tensor.
    """
    d_out = W.shape[0]

    if symmetric:
        quant_max = 2 ** (num_bits - 1) - 1
        if per_channel:
            abs_max = W.abs().amax(dim=1).clamp(min=1e-8)   # [d_out]
        else:
            abs_max = W.abs().amax().clamp(min=1e-8).expand(d_out)
        scale = abs_max / quant_max
        zp = torch.zeros(d_out, dtype=torch.int32, device=W.device)
    else:
        quant_max = 2 ** num_bits - 1
        if per_channel:
            w_min = W.amin(dim=1)   # [d_out]
            w_max = W.amax(dim=1)
        else:
            w_min = W.amin().expand(d_out)
            w_max = W.amax().expand(d_out)
        w_min = torch.minimum(w_min, torch.zeros_like(w_min))
        w_max = torch.maximum(w_max, torch.zeros_like(w_max))
        scale = ((w_max - w_min) / quant_max).clamp(min=1e-8)
        zp = (-w_min / scale).round().to(torch.int32)

    return scale, zp   # both [d_out]


def _fake_quant_col(
    col: torch.Tensor,
    scale: torch.Tensor,
    zp: torch.Tensor,
    num_bits: int,
    symmetric: bool,
) -> torch.Tensor:
    """
    Fake-quantize a single column vector of shape [d_out].

    Args:
        col:   [d_out] weight column.
        scale: [d_out] per-row scale.
        zp:    [d_out] per-row zero-point.

    Returns:
        Fake-quantized column with the same shape and dtype as col.
    """
    s = scale.to(col.dtype)
    z = zp.to(col.dtype)
    if symmetric:
        quant_min = -(2 ** (num_bits - 1))
        quant_max = 2 ** (num_bits - 1) - 1
        q = (col / s).round().clamp(quant_min, quant_max)
        return q * s
    else:
        quant_max = 2 ** num_bits - 1
        q = (col / s + z).round().clamp(0, quant_max)
        return (q - z) * s


def _gptq_quantize_weight(
    W: torch.Tensor,
    X: torch.Tensor,
    num_bits: int,
    symmetric: bool,
    per_channel: bool,
    damp_percent: float,
    blocksize: int,
) -> torch.Tensor:
    """
    Apply the GPTQ algorithm to quantize weight matrix W.

    Args:
        W:            [d_out, d_in] weight matrix (float32).
        X:            [n, d_in] collected input activations (float32).
        num_bits:     Quantization bit-width.
        symmetric:    Symmetric or asymmetric quantization.
        per_channel:  Per output-channel or per-tensor quantization scales.
        damp_percent: Fraction of mean Hessian diagonal used for damping (1e-2).
        blocksize:    Number of input columns processed per block (128).

    Returns:
        W_q: fake-quantized weight matrix (same shape and dtype as W).
    """
    device = W.device
    orig_dtype = W.dtype
    d_out, d_in = W.shape

    # --- Hessian H = (2/n) · Xᵀ X ---
    W_f = W.float()
    X_f = X.float()
    n = X_f.shape[0]
    H = (2.0 / n) * (X_f.T @ X_f)           # [d_in, d_in]

    # Diagonal damping for numerical stability
    damp = damp_percent * H.diag().mean()
    H.diagonal().add_(damp)

    # H inverse via Cholesky (more stable than torch.linalg.inv for PSD matrices)
    try:
        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    except torch.linalg.LinAlgError:
        # Fallback: pseudo-inverse (handles rank-deficient calibration data)
        H_inv = torch.linalg.pinv(H)

    # --- Pre-compute quantization parameters from the initial W ---
    # Using consistent scales throughout the column updates keeps the
    # quantization grid fixed, which simplifies analysis.
    scale, zp = _compute_qparams(W_f, num_bits, symmetric, per_channel)

    # --- Blockwise column quantization ---
    W_q = W_f.clone()

    for i1 in range(0, d_in, blocksize):
        i2 = min(i1 + blocksize, d_in)
        block_len = i2 - i1

        # E: accumulated errors within this block  [d_out, block_len]
        E = torch.zeros(d_out, block_len, device=device, dtype=torch.float32)

        for i in range(block_len):
            col_idx = i1 + i
            col = W_q[:, col_idx]               # [d_out]
            q_col = _fake_quant_col(col, scale, zp, num_bits, symmetric)

            # Error propagation factor:
            #   e_i = (w_i − q_i) / H⁻¹_{ii}
            # (The per-column update to compensate: w_{j>i} -= e_i · H⁻¹_{i,j>i})
            h_inv_ii = H_inv[col_idx, col_idx].clamp(min=1e-12)
            err = (col - q_col) / h_inv_ii      # [d_out]
            E[:, i] = err

            W_q[:, col_idx] = q_col

            # Update remaining columns within this block
            if col_idx + 1 < i2:
                W_q[:, col_idx + 1 : i2] -= (
                    err.unsqueeze(1)                          # [d_out, 1]
                    * H_inv[col_idx, col_idx + 1 : i2].unsqueeze(0)  # [1, rem]
                )

        # Propagate accumulated block errors to all columns after the block
        if i2 < d_in:
            W_q[:, i2:] -= E @ H_inv[i1:i2, i2:]    # [d_out, d_in-i2]

    return W_q.to(orig_dtype)


def _set_module(model: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


# ---------------------------------------------------------------------------
# GPTQLinear — inference module (real INT4 packed storage)
# ---------------------------------------------------------------------------

class GPTQLinear(nn.Module):
    """
    Linear layer backed by GPTQ-optimised weights.

    Storage modes
    ~~~~~~~~~~~~~
    INT4 packed (``num_bits == 4``):
        qweight   [out_features, in_features // 8]   int32  8 INT4 per int32
        scales    [n_groups, out_features]            float32
        qzeros    [n_groups, out_features]            int32

        On CUDA with Triton: Triton kernel dequantises qweight → fp16,
        cuBLAS handles the GEMM.  On CPU: pure-PyTorch fallback.

    Float32 fallback (``num_bits != 4``):
        weight_fp32  [out_features, in_features]   float32  GPTQ fake-quant weight

        Uses ``F.linear`` directly.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        *,
        qweight:      Optional[torch.Tensor] = None,   # INT4 packed mode
        scales:       Optional[torch.Tensor] = None,
        qzeros:       Optional[torch.Tensor] = None,
        weight_fp32:  Optional[torch.Tensor] = None,   # float32 fallback
        group_size:   int = 128,
        bias:         Optional[torch.Tensor] = None,
        num_bits:     int = 4,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.group_size   = group_size
        self.num_bits     = num_bits
        self._use_packed  = (qweight is not None)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_packed:
            return int4_dequant_gemm(
                x, self.qweight, self.scales, self.qzeros,
                group_size=self.group_size, bias=self.bias,
            )
        return F.linear(x, self.weight_fp32, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"num_bits={self.num_bits}, "
            f"group_size={self.group_size}"
        )


# ---------------------------------------------------------------------------
# GPTQBackend
# ---------------------------------------------------------------------------

class GPTQBackend(QuantBackend):
    """
    GPTQ post-training quantization backend (pure PyTorch, no extra packages).

    Performs Hessian-based second-order weight update to minimise layer output
    error after quantization.  Most beneficial at INT4 weight-only quantization.

    Calibration:
        calibrate() attaches forward hooks to all Linear layers, runs the
        calibration function, and stores the flattened input activations.

    Conversion:
        convert() deep-copies the model, applies the GPTQ algorithm per Linear
        layer, and stores the GPTQ-optimised fake-quantized weights directly in
        the nn.Linear weight parameter.  No special inference wrapper is needed:
        GPTQ only changes the weight *values*, not the computation graph.

    Fallback:
        If a layer is never reached during calibration (no activations
        collected), it falls back to per-tensor MinMax fake quantization.
    """

    def __init__(
        self,
        num_bits: int = 4,
        symmetric: bool = True,
        per_channel: bool = True,
        weight_only: bool = True,
        calibration_method: str = "minmax",
        calibration_percentile: float = 99.99,
        damp_percent: float = 0.01,
        blocksize: int = 128,
    ) -> None:
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.weight_only = weight_only
        self.calibration_method = calibration_method
        self.calibration_percentile = calibration_percentile
        self.damp_percent = damp_percent
        self.blocksize = blocksize

        self._activations: Dict[str, torch.Tensor] = {}
        self._hooks: List[Any] = []

    @property
    def name(self) -> str:
        return "gptq"

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

        Activations are flattened to [n, in_features] so the Hessian
        computation is straightforward regardless of input rank.
        """
        buffers: Dict[str, List[torch.Tensor]] = {}

        def make_hook(layer_name: str) -> Callable:
            def hook(module: nn.Module, inp: Tuple, out: torch.Tensor) -> None:
                x = inp[0].detach().float()
                # Flatten all batch/sequence dimensions → [n, in_features]
                x = x.reshape(-1, x.shape[-1])
                if layer_name not in buffers:
                    buffers[layer_name] = []
                buffers[layer_name].append(x)
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

        # Concatenate all batches → [n_total, in_features]
        for name, chunks in buffers.items():
            self._activations[name] = torch.cat(chunks, dim=0)

    def convert(self, model: nn.Module) -> nn.Module:
        """
        Return a deep copy of the model with GPTQ-optimised INT4 layers.

        Each nn.Linear is replaced by a GPTQLinear that stores packed int32
        qweight, float32 scales, and int32 qzeros.  The original model is
        not mutated.
        """
        model_q = copy.deepcopy(model)

        for name, module in list(model_q.named_modules()):
            if not isinstance(module, nn.Linear):
                continue

            W = module.weight.data.float()
            bias = module.bias.data.clone() if module.bias is not None else None
            X = self._activations.get(name)

            # Compute quantization scales from original weight
            per_ch = self.per_channel if X is not None else False
            scale, zp = _compute_qparams(W, self.num_bits, self.symmetric, per_ch)

            if X is not None:
                W_q = _gptq_quantize_weight(
                    W, X.to(W.device),
                    self.num_bits, self.symmetric, self.per_channel,
                    self.damp_percent, self.blocksize,
                )
            else:
                # Fallback: plain per-tensor MinMax fake quantization
                scale_b = scale.unsqueeze(1)
                zp_b    = zp.to(torch.float32).unsqueeze(1)
                if self.symmetric:
                    qmax = 2 ** (self.num_bits - 1) - 1
                    qmin = -qmax - 1
                    W_q = (W / scale_b).round().clamp(qmin, qmax) * scale_b
                else:
                    qmax = 2 ** self.num_bits - 1
                    W_q = ((W / scale_b + zp_b).round().clamp(0, qmax) - zp_b) * scale_b

            # ---- Pack float32 fake-quant → real INT4 or float32 fallback ----
            if self.num_bits == 4:
                # Recover unsigned integer grid values from the fake-quant output:
                #   symmetric:   W_uint = round(W_q / scale) + 2^(bits-1)
                #   asymmetric:  W_uint = round(W_q / scale + zp)
                if self.symmetric:
                    qzero_val = 2 ** (self.num_bits - 1)   # 8 for INT4
                    qzero     = torch.full_like(zp, qzero_val)
                    W_uint    = (
                        (W_q.float() / scale.unsqueeze(1)) + qzero_val
                    ).round().clamp(0, 2 ** self.num_bits - 1).to(torch.int32)
                else:
                    qzero  = zp
                    W_uint = (
                        W_q.float() / scale.unsqueeze(1) + zp.float().unsqueeze(1)
                    ).round().clamp(0, 2 ** self.num_bits - 1).to(torch.int32)

                scales_g = scale.unsqueeze(0)   # [1, N]
                qzeros_g = qzero.unsqueeze(0)   # [1, N]
                qweight  = pack_int4(W_uint)

                gptq_layer = GPTQLinear(
                    in_features  = module.in_features,
                    out_features = module.out_features,
                    qweight      = qweight,
                    scales       = scales_g,
                    qzeros       = qzeros_g,
                    group_size   = module.in_features,
                    bias         = bias,
                    num_bits     = self.num_bits,
                )
            else:
                # Float32 fallback for non-INT4 bit widths (e.g. INT8).
                gptq_layer = GPTQLinear(
                    in_features  = module.in_features,
                    out_features = module.out_features,
                    weight_fp32  = W_q.float(),
                    group_size   = module.in_features,
                    bias         = bias,
                    num_bits     = self.num_bits,
                )
            _set_module(model_q, name, gptq_layer)

        return model_q
