"""
bitsandbytes quantization backend.

Implements the QuantBackend interface using bitsandbytes (bnb):
  https://github.com/TimDettmers/bitsandbytes

Two quantization modes are supported:

  INT8 — Linear8bitLt (LLM.int8() algorithm)
    - Weights are stored as int8; activations are computed in float32/float16.
    - Outlier features (activation values > threshold) are automatically
      computed in higher precision (mixed-precision decomposition).
    - Works on both CPU and CUDA. Actual int8 CUDA kernels are used on GPU.
    - No calibration data required; quantization occurs on the first forward.

  INT4 — Linear4bit (NF4 / FP4, used in QLoRA)
    - Weights are stored in 4-bit NormalFloat (NF4) or 4-bit Float (FP4).
    - Optional double quantization (compress_statistics) stores quantization
      constants themselves in 8-bit, reducing memory further.
    - Requires a CUDA device; CPU forward is not supported by bitsandbytes.
    - No calibration data required; quantization occurs on the first forward.

Key differences from FakeQuantBackend:
  - Real compressed weight storage (memory saving is genuine, not simulated).
  - Activations are not statically quantized; the model uses bitsandbytes'
    own mixed-precision strategy at runtime.
  - No calibration step.

Key differences from TorchAOBackend:
  - Uses bitsandbytes kernels instead of torch.ao.
  - INT4 (NF4) support.
  - Designed for large language model weight compression.

Requires: pip install bitsandbytes>=0.41
"""

from __future__ import annotations

import copy
from typing import Callable

import torch
import torch.nn as nn

from .base import QuantBackend


class BitsAndBytesBackend(QuantBackend):
    """
    Quantization backend powered by bitsandbytes.

    Args:
        num_bits:                  8 → Linear8bitLt (INT8),  4 → Linear4bit (NF4/FP4).
        symmetric:                 Accepted for API compatibility; bnb manages
                                   quantization scheme internally.
        per_channel:               Accepted for API compatibility; bnb manages
                                   granularity internally.
        weight_only:               Accepted for API compatibility; bnb always
                                   quantizes weights only by design.
        calibration_method:        Accepted for API compatibility; no-op.
        calibration_percentile:    Accepted for API compatibility; no-op.
        threshold:                 (INT8 only) LLM.int8() outlier threshold.
                                   Features with activation magnitudes above
                                   this value are computed in float precision.
                                   Default 6.0 follows the original paper.
        bnb_4bit_quant_type:       (INT4 only) "nf4" or "fp4".
        bnb_4bit_compute_dtype:    (INT4 only) dtype used for the dequantized
                                   computation. Default: torch.float32.
        bnb_4bit_use_double_quant: (INT4 only) Double quantization: also
                                   quantize the quantization constants to 8-bit,
                                   saving ~0.4 bits/parameter.
    """

    def __init__(
        self,
        num_bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
        weight_only: bool = True,
        calibration_method: str = "minmax",
        calibration_percentile: float = 99.99,
        # INT8-specific
        threshold: float = 6.0,
        # INT4-specific
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_compute_dtype: torch.dtype = torch.float32,
        bnb_4bit_use_double_quant: bool = False,
    ) -> None:
        if num_bits not in (4, 8):
            raise ValueError(
                f"BitsAndBytesBackend supports 4-bit and 8-bit only, got {num_bits}-bit."
            )
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.weight_only = weight_only
        self.calibration_method = calibration_method
        self.calibration_percentile = calibration_percentile
        self.threshold = threshold
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant

    @property
    def name(self) -> str:
        return "bitsandbytes"

    # ------------------------------------------------------------------
    # QuantBackend interface
    # ------------------------------------------------------------------

    def calibrate(
        self,
        model: nn.Module,
        calibration_fn: Callable[[], None],
    ) -> None:
        """
        No-op for bitsandbytes.

        bitsandbytes quantizes weights lazily on the first forward call
        using the min/max of the weight tensor itself. No external
        calibration dataset is required.
        """
        pass  # intentional no-op

    def convert(self, model: nn.Module) -> nn.Module:
        """
        Return a deep copy of the model with nn.Linear layers replaced by
        bitsandbytes quantized linear layers.

        INT8 → ``bitsandbytes.nn.Linear8bitLt`` (LLM.int8()).
        INT4 → ``bitsandbytes.nn.Linear4bit`` (NF4 / FP4).

        The original model is not mutated.

        Note:
            INT4 layers require a CUDA device for quantized forward passes.
            The model can be constructed on CPU and moved to CUDA afterwards.
        """
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError(
                "bitsandbytes is required for BitsAndBytesBackend. "
                "Install it with: pip install bitsandbytes>=0.41"
            ) from exc

        model_q = copy.deepcopy(model)

        if self.num_bits == 8:
            self._replace_with_int8(model_q, bnb)
        else:
            self._replace_with_int4(model_q, bnb)

        return model_q

    # ------------------------------------------------------------------
    # Layer replacement helpers
    # ------------------------------------------------------------------

    def _replace_with_int8(self, model: nn.Module, bnb) -> None:
        """
        Replace every nn.Linear in *model* (in-place) with Linear8bitLt.

        ``has_fp16_weights=False`` ensures weights are stored as int8
        rather than kept as float16. The threshold parameter controls the
        LLM.int8() mixed-precision decomposition for outlier features.
        """
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue

            new_layer = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                has_fp16_weights=False,
                threshold=self.threshold,
            )
            new_layer.weight = bnb.nn.Int8Params(
                module.weight.data.clone(),
                requires_grad=False,
                has_fp16_weights=False,
            )
            if module.bias is not None:
                new_layer.bias = nn.Parameter(
                    module.bias.data.clone(), requires_grad=False
                )

            _set_module(model, name, new_layer)

    def _replace_with_int4(self, model: nn.Module, bnb) -> None:
        """
        Replace every nn.Linear in *model* (in-place) with Linear4bit.

        ``quant_type="nf4"`` uses NormalFloat4 (better than FP4 for
        normally distributed weights). ``compress_statistics=True``
        enables double quantization for extra memory savings.

        Note: Linear4bit forward requires a CUDA device.
        """
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue

            new_layer = bnb.nn.Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                compute_dtype=self.bnb_4bit_compute_dtype,
                compress_statistics=self.bnb_4bit_use_double_quant,
                quant_type=self.bnb_4bit_quant_type,
            )
            new_layer.weight = bnb.nn.Params4bit(
                module.weight.data.clone(),
                requires_grad=False,
                quant_type=self.bnb_4bit_quant_type,
            )
            if module.bias is not None:
                new_layer.bias = nn.Parameter(
                    module.bias.data.clone(), requires_grad=False
                )

            _set_module(model, name, new_layer)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _set_module(model: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    """Replace a submodule identified by a dotted attribute path."""
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)
