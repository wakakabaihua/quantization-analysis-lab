"""
Post-training quantization (PTQ) pipeline.

Orchestrates the full PTQ workflow by delegating to a pluggable
QuantBackend. The pipeline itself is backend-agnostic: it reads
configuration, selects the right backend, and exposes the two-phase
calibrate() / quantize() API used by scripts and tests.

Supported backends (Phase 1):
    "fake"  — float32 fake quantization (default, fully portable)

Future backends (Phase 2+):
    "torch"        — torch.ao.quantization
    "bitsandbytes" — bitsandbytes LLM.int8() / NF4
    "gptq"         — AutoGPTQ
    "awq"          — AutoAWQ

The original model is never mutated. quantize() always returns a fresh copy.
"""

from __future__ import annotations

import copy
from typing import Callable, Optional

import torch.nn as nn

from .backends import QuantBackend, get_backend
from .backends.fake_quant_backend import QuantizedLinear, _set_module  # re-export for compat
from .backends.mixed_precision_backend import MixedPrecisionBackend


class PTQPipeline:
    """
    Configures and runs post-training quantization.

    Typical usage::

        pipeline = PTQPipeline.from_config(config_dict)
        pipeline.calibrate(model, lambda: run_forward_passes(model, data))
        model_q = pipeline.quantize(model)

    When ``enabled=False`` (e.g. for the FP16 baseline config),
    calibrate() is a no-op and quantize() returns a plain deep copy.
    """

    def __init__(
        self,
        num_bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
        weight_only: bool = False,
        calibration_method: str = "minmax",
        calibration_percentile: float = 99.99,
        enabled: bool = True,
        backend: "str | QuantBackend" = "fake",
    ) -> None:
        self.enabled = enabled
        self._backend: Optional[QuantBackend] = None

        if enabled:
            if isinstance(backend, QuantBackend):
                # Accept a pre-constructed backend instance directly.
                self._backend = backend
            else:
                self._backend = get_backend(
                    backend,
                    num_bits=num_bits,
                    symmetric=symmetric,
                    per_channel=per_channel,
                    weight_only=weight_only,
                    calibration_method=calibration_method,
                    calibration_percentile=calibration_percentile,
                )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict) -> "PTQPipeline":
        """Build a PTQPipeline from a parsed YAML config dict."""
        q = config.get("quantization", {})
        cal = config.get("calibration", {})
        granularity = config.get("granularity", "per_tensor")
        enabled = q.get("enabled", True)
        backend_name = config.get("backend", "fake")

        shared_kwargs = dict(
            symmetric=q.get("symmetric", True),
            per_channel=(granularity == "per_channel"),
            weight_only=q.get("weight_only", False),
            calibration_method=cal.get("method", "minmax"),
            calibration_percentile=float(cal.get("percentile") or 99.99),
        )

        # Mixed-precision needs its own constructor path because it accepts
        # a per-layer config that the standard get_backend() call cannot carry.
        if backend_name in ("mixed_precision", "mp") and enabled:
            mp_cfg = config.get("mixed_precision", {})
            backend_instance = MixedPrecisionBackend.from_config_dict(
                mp_cfg, **shared_kwargs
            )
            return cls(enabled=enabled, backend=backend_instance)

        return cls(
            num_bits=_dtype_to_bits(q.get("dtype", "int8")),
            enabled=enabled,
            backend=backend_name,
            **shared_kwargs,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(
        self,
        model: nn.Module,
        calibration_fn: Callable[[], None],
    ) -> None:
        """
        Collect quantization statistics from representative data.

        Args:
            model:          The (float) model to calibrate.
            calibration_fn: Zero-argument callable that runs forward passes
                            over representative calibration data.
        """
        if not self.enabled or self._backend is None:
            return
        self._backend.calibrate(model, calibration_fn)

    def quantize(self, model: nn.Module) -> nn.Module:
        """
        Return a quantized copy of the model.

        When disabled, returns a plain deep copy (useful for FP16 baseline).

        Args:
            model: The (float) model that was previously calibrated.

        Returns:
            A new model with quantized layers.
        """
        if not self.enabled or self._backend is None:
            return copy.deepcopy(model)
        return self._backend.convert(model)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _dtype_to_bits(dtype: str) -> int:
    mapping = {"fp32": 32, "fp16": 16, "int8": 8, "int4": 4, "int2": 2}
    key = dtype.lower()
    if key not in mapping:
        raise ValueError(f"Unknown dtype {dtype!r}. Choose from: {list(mapping)}")
    return mapping[key]
