"""
Mixed-Precision Quantization backend.

Assigns a (possibly different) bit-width to each Linear layer, allowing
sensitive layers to keep higher precision while robust layers are aggressively
compressed to lower bit-widths.

Two ways to specify per-layer precision:

1. Explicit configuration::

    backend = MixedPrecisionBackend(
        layer_config={
            "fc1": {"num_bits": 8},
            "fc2": {"num_bits": 4},
        },
        default_num_bits=8,
    )

2. Automatic from sensitivity scores::

    scores = {"fc1": 0.9997, "fc2": 0.9999}
    backend = MixedPrecisionBackend.from_sensitivity(
        sensitivity_scores=scores,
        threshold=0.9999,   # layers below → high_bits; above → low_bits
        high_bits=8,
        low_bits=4,
    )

3. From a parsed YAML config (used by PTQPipeline.from_config)::

    mp_cfg = {
        "default_num_bits": 8,
        "layers": {"fc1": {"num_bits": 8}, "fc2": {"num_bits": 4}},
    }
    backend = MixedPrecisionBackend.from_config_dict(mp_cfg, symmetric=True, ...)

All settings other than num_bits (symmetric, per_channel, weight_only,
calibration_method) are shared across all layers.

The quantization logic is identical to FakeQuantBackend: float32 fake
quantization using the same Observer → Calibrator → FakeQuantize pipeline.
Each layer independently uses its own bit-width for both weight_fq (and
act_fq if weight_only=False).

Memory estimates:

    theoretical_compression(model) returns FP32 vs mixed-precision weight
    sizes, letting you reason about the memory–accuracy tradeoff.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..calibrators import BaseCalibrator, get_calibrator
from ..fake_quant import FakeQuantize
from ..observers import BaseObserver, HistogramObserver, MinMaxObserver
from .base import QuantBackend
from .fake_quant_backend import QuantizedLinear, _set_module


# ---------------------------------------------------------------------------
# MixedPrecisionBackend
# ---------------------------------------------------------------------------

class MixedPrecisionBackend(QuantBackend):
    """
    Per-layer precision meta-backend for post-training quantization.

    Attributes:
        layer_config:       Dict mapping layer name → per-layer overrides.
                            Currently only ``num_bits`` is consumed;
                            future overrides could include ``symmetric``.
        default_num_bits:   Bit-width used for layers not in ``layer_config``.
        symmetric:          Shared across all layers.
        per_channel:        Per output-channel (True) or per-tensor (False).
        weight_only:        Quantize weights only if True; also quantize
                            activations otherwise.
        calibration_method: "minmax" | "percentile" | "kl".
        calibration_percentile: Used when method == "percentile".
    """

    def __init__(
        self,
        layer_config: Optional[Dict[str, Dict]] = None,
        default_num_bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = True,
        weight_only: bool = True,
        calibration_method: str = "minmax",
        calibration_percentile: float = 99.99,
    ) -> None:
        self.layer_config: Dict[str, Dict] = {
            k: dict(v) for k, v in (layer_config or {}).items()
        }
        self.default_num_bits = default_num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.weight_only = weight_only
        self.calibration_method = calibration_method
        self.calibration_percentile = calibration_percentile

        self._activation_stats: Dict[str, dict] = {}
        self._weight_stats: Dict[str, dict] = {}
        self._hooks: List[Any] = []

    @property
    def name(self) -> str:
        return "mixed_precision"

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_sensitivity(
        cls,
        sensitivity_scores: Dict[str, float],
        threshold: float = 0.9999,
        high_bits: int = 8,
        low_bits: int = 4,
        **kwargs,
    ) -> "MixedPrecisionBackend":
        """
        Auto-assign bit-widths using per-layer sensitivity scores.

        Layers with ``cosine_similarity < threshold`` are considered sensitive
        and receive ``high_bits``; layers at or above the threshold are
        considered robust and receive ``low_bits``.

        Args:
            sensitivity_scores: Dict mapping each layer name to its
                                 cosine-similarity score when quantized alone.
            threshold:          Cosine similarity below which a layer is
                                 considered sensitive (default 0.9999).
            high_bits:          Bits assigned to sensitive layers (default 8).
            low_bits:           Bits assigned to robust layers (default 4).
            **kwargs:           Extra kwargs forwarded to ``__init__``
                                (symmetric, per_channel, weight_only, …).

        Returns:
            A configured MixedPrecisionBackend instance.

        Example::

            scores = {"fc1": 0.9997, "fc2": 0.99995}
            backend = MixedPrecisionBackend.from_sensitivity(
                scores, threshold=0.9999, high_bits=8, low_bits=4
            )
            # fc1  → 8-bit (below threshold)
            # fc2  → 4-bit (above threshold)
        """
        layer_config: Dict[str, Dict] = {}
        for layer_name, score in sensitivity_scores.items():
            bits = high_bits if score < threshold else low_bits
            layer_config[layer_name] = {"num_bits": bits}
        return cls(
            layer_config=layer_config,
            default_num_bits=high_bits,
            **kwargs,
        )

    @classmethod
    def from_config_dict(
        cls,
        mp_cfg: Dict,
        **shared_kwargs,
    ) -> "MixedPrecisionBackend":
        """
        Build from the ``mixed_precision:`` section of a YAML config dict.

        Expected structure::

            default_num_bits: 8
            layers:
              fc1: {num_bits: 8}
              fc2: {num_bits: 4}

        Args:
            mp_cfg:         The ``mixed_precision`` sub-dict from the config.
            **shared_kwargs: Forwarded to ``__init__``
                             (symmetric, per_channel, weight_only, …).
        """
        default_num_bits = mp_cfg.get("default_num_bits", 8)
        layer_config = {
            k: dict(v) for k, v in mp_cfg.get("layers", {}).items()
        }
        return cls(
            layer_config=layer_config,
            default_num_bits=default_num_bits,
            **shared_kwargs,
        )

    # ------------------------------------------------------------------
    # Per-layer config helpers
    # ------------------------------------------------------------------

    def _num_bits_for(self, layer_name: str) -> int:
        """Return the configured bit-width for a named layer."""
        return int(
            self.layer_config.get(layer_name, {}).get("num_bits", self.default_num_bits)
        )

    # ------------------------------------------------------------------
    # Observer / calibrator factories
    # ------------------------------------------------------------------

    def _make_weight_observer(self) -> BaseObserver:
        if self.calibration_method in ("percentile", "kl", "histogram"):
            return HistogramObserver(per_channel=self.per_channel)
        return MinMaxObserver(per_channel=self.per_channel)

    def _make_act_observer(self) -> BaseObserver:
        if self.calibration_method in ("percentile", "kl", "histogram"):
            return HistogramObserver(per_channel=False)
        return MinMaxObserver(per_channel=False)

    def _make_calibrator(self, num_bits: int) -> BaseCalibrator:
        kwargs: dict = dict(num_bits=num_bits, symmetric=self.symmetric)
        if self.calibration_method == "percentile":
            kwargs["percentile"] = self.calibration_percentile
        return get_calibrator(self.calibration_method, **kwargs)

    # ------------------------------------------------------------------
    # QuantBackend interface
    # ------------------------------------------------------------------

    def calibrate(
        self,
        model: nn.Module,
        calibration_fn: Callable[[], None],
    ) -> None:
        """
        Collect per-layer weight and activation statistics.

        Attaches forward hooks to every Linear layer in the model, runs
        ``calibration_fn``, then stores the statistics for use in
        ``convert()``.  The model itself is not modified.
        """
        activation_observers: Dict[str, BaseObserver] = {}

        def make_hook(obs: BaseObserver) -> Callable:
            def hook(module: nn.Module, inp: Tuple, out: torch.Tensor) -> None:
                obs.update(inp[0].detach())
            return hook

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                obs = self._make_act_observer()
                activation_observers[name] = obs
                self._hooks.append(module.register_forward_hook(make_hook(obs)))

        model.eval()
        with torch.no_grad():
            calibration_fn()

        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        # Activation stats (for weight+act mode)
        for name, obs in activation_observers.items():
            try:
                self._activation_stats[name] = obs.stats
            except RuntimeError:
                pass

        # Weight stats (always collected; calibration_fn not needed for weights)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                w_obs = self._make_weight_observer()
                w_obs.update(module.weight.detach())
                self._weight_stats[name] = w_obs.stats

    def convert(self, model: nn.Module) -> nn.Module:
        """
        Return a deep copy of the model with per-layer mixed-precision
        fake quantization applied.

        Each Linear layer uses the bit-width specified in ``layer_config``
        (or ``default_num_bits`` if the layer is not listed).  A separate
        FakeQuantize instance is created per layer so that quantization
        grids are independently calibrated.

        The original model is not mutated.
        """
        model_q = copy.deepcopy(model)

        for name, module in list(model_q.named_modules()):
            if not isinstance(module, nn.Linear):
                continue

            num_bits = self._num_bits_for(name)
            calibrator = self._make_calibrator(num_bits)

            # --- Weight quantization ---
            w_stats = self._weight_stats.get(name)
            if w_stats is None:
                # Fallback: observe weights now (no calibration data available)
                w_obs = self._make_weight_observer()
                w_obs.update(module.weight.detach())
                w_stats = w_obs.stats

            w_scale, w_zp = calibrator.compute(w_stats)
            weight_fq = FakeQuantize(
                num_bits=num_bits,
                symmetric=self.symmetric,
                per_channel=self.per_channel,
                channel_axis=0,
            )
            weight_fq.set_qparams(w_scale, w_zp)

            # --- Activation quantization (weight-only mode skips this) ---
            act_fq: Optional[FakeQuantize] = None
            if not self.weight_only:
                a_stats = self._activation_stats.get(name)
                if a_stats is not None:
                    a_scale, a_zp = calibrator.compute(a_stats)
                    act_fq = FakeQuantize(
                        num_bits=num_bits,
                        symmetric=self.symmetric,
                        per_channel=False,
                        channel_axis=0,
                    )
                    act_fq.set_qparams(a_scale, a_zp)

            q_linear = QuantizedLinear(module, weight_fq, act_fq)
            _set_module(model_q, name, q_linear)

        return model_q

    # ------------------------------------------------------------------
    # Utility / introspection
    # ------------------------------------------------------------------

    def bit_assignment(self, model: nn.Module) -> Dict[str, int]:
        """
        Return the per-layer bit-width assignment for all Linear layers.

        Useful for logging and debugging the mixed-precision configuration.

        Args:
            model: The (float) model whose Linear layers are to be assigned.

        Returns:
            Dict mapping each layer's dotted name to its bit-width.

        Example::

            {"fc1": 8, "fc2": 4, "attention.q_proj": 8, ...}
        """
        return {
            name: self._num_bits_for(name)
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        }

    def theoretical_compression(self, model: nn.Module) -> Dict:
        """
        Estimate theoretical weight memory relative to FP32.

        Compares total FP32 weight bytes against the mixed-precision
        equivalent, assuming each value occupies exactly ``num_bits`` bits
        (no alignment padding, no headers).

        Args:
            model: The (float) model to analyse.

        Returns:
            Dict with keys:
                ``fp32_bytes``        — total FP32 weight bytes.
                ``mixed_bytes``       — total mixed-precision weight bytes.
                ``compression_ratio`` — fp32_bytes / mixed_bytes.
                ``per_layer``         — per-layer breakdown.
        """
        fp32_bytes = 0
        mixed_bytes = 0
        per_layer: Dict[str, Dict] = {}

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            n_params = module.weight.numel()
            layer_fp32 = n_params * 4          # float32 = 4 bytes
            num_bits = self._num_bits_for(name)
            layer_mp = n_params * num_bits / 8  # bits → bytes

            fp32_bytes += layer_fp32
            mixed_bytes += layer_mp
            per_layer[name] = {
                "num_bits": num_bits,
                "params": n_params,
                "fp32_bytes": layer_fp32,
                "mixed_bytes": layer_mp,
                "compression_ratio": layer_fp32 / layer_mp,
            }

        return {
            "fp32_bytes": fp32_bytes,
            "mixed_bytes": mixed_bytes,
            "compression_ratio": fp32_bytes / mixed_bytes if mixed_bytes > 0 else float("inf"),
            "per_layer": per_layer,
        }
