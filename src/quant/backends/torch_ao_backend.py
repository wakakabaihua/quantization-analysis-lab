"""
torch.ao.quantization backend.

Implements the QuantBackend interface using PyTorch's built-in
torch.ao.quantization.quantize_dynamic().

Why dynamic quantization?
    PyTorch eager-mode *static* quantization requires the model to have
    explicit QuantStub / DeQuantStub boundaries and cannot automatically
    insert requantize operations between non-quantizable layers (e.g.,
    GELU, softmax).  For arbitrary model architectures like MLPBlock and
    AttentionBlock this would require invasive model changes.

    *Dynamic* quantization (torch.ao.quantization.quantize_dynamic):
      - Quantizes weights to INT8 offline (at convert time).
      - Quantizes activations to INT8 dynamically per inference call
        using each batch's actual min/max range.
      - Requires no calibration dataset.
      - Works with any model architecture.
      - Produces real INT8 linear kernels (unlike FakeQuantBackend
        which stays entirely in float32).

    For models that have been specifically designed with QuantStub /
    DeQuantStub or can be captured by torch.fx, use prepare_fx /
    convert_fx from torch.ao.quantization.quantize_fx instead.

Constraints:
    - Only INT8 is supported (torch.ao INT8 dtype support).
    - Calibration data is not used; calibrate() is a no-op.
    - INT4 and other bit-widths require FakeQuantBackend.
"""

from __future__ import annotations

import copy
from typing import Callable

import torch
import torch.nn as nn

from .base import QuantBackend


class TorchAOBackend(QuantBackend):
    """
    Quantization backend powered by torch.ao.quantization.quantize_dynamic().

    INT8 weights are quantized at convert() time; activations are quantized
    dynamically on every forward call based on the current batch's range.
    No calibration data is required.

    Supported options:
        num_bits=8        — only 8-bit is supported.
        symmetric         — True: per_channel_symmetric / per_tensor_symmetric.
                            False: per_channel_affine / per_tensor_affine.
        per_channel       — True: PerChannelMinMaxObserver for weights.
                            False: MinMaxObserver for weights (default).
        weight_only       — ignored (dynamic quantization always quantizes
                            activations at runtime; there is no weight-only
                            mode here).
        calibration_*     — ignored; dynamic quantization needs no calibration.
    """

    def __init__(
        self,
        num_bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
        weight_only: bool = False,
        calibration_method: str = "minmax",
        calibration_percentile: float = 99.99,
    ) -> None:
        if num_bits != 8:
            raise ValueError(
                f"TorchAOBackend only supports 8-bit quantization, got {num_bits}-bit. "
                "Use FakeQuantBackend for INT4 or other bit-widths."
            )
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        # weight_only and calibration_* are accepted for API compatibility
        # but have no effect on dynamic quantization.
        self.weight_only = weight_only
        self.calibration_method = calibration_method
        self.calibration_percentile = calibration_percentile

    @property
    def name(self) -> str:
        return "torch_ao"

    # ------------------------------------------------------------------
    # QuantBackend interface
    # ------------------------------------------------------------------

    def calibrate(
        self,
        model: nn.Module,
        calibration_fn: Callable[[], None],
    ) -> None:
        """
        No-op for dynamic quantization.

        torch.ao.quantization.quantize_dynamic() computes weight scales
        offline at convert() time and activation scales dynamically at
        inference time, so no calibration dataset is required.
        """
        pass  # intentional no-op

    def convert(self, model: nn.Module) -> nn.Module:
        """
        Return a deep copy of the model with nn.Linear layers replaced by
        torch.ao DynamicQuantizedLinear (real INT8 kernels).

        The original model is not mutated.
        """
        import torch.ao.quantization as taq

        model_copy = copy.deepcopy(model)
        qconfig = self._make_qconfig(taq)
        return taq.quantize_dynamic(
            model_copy,
            {nn.Linear: qconfig},
            dtype=torch.qint8,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_qconfig(self, taq):
        """
        Build a QConfig for dynamic quantization.

        The activation observer is a placeholder (float passthrough) because
        activations are quantized dynamically at runtime, not from calibration.
        The weight observer controls whether weights are quantized per-tensor
        or per-channel.
        """
        # Activation: PlaceholderObserver tells torch.ao to quantize
        # activations dynamically at inference (not from calibration data).
        act_obs = taq.PlaceholderObserver.with_args(dtype=torch.float32)

        if self.per_channel:
            qscheme = (
                torch.per_channel_symmetric
                if self.symmetric
                else torch.per_channel_affine
            )
            weight_obs = taq.PerChannelMinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=qscheme,
            )
        else:
            qscheme = (
                torch.per_tensor_symmetric
                if self.symmetric
                else torch.per_tensor_affine
            )
            weight_obs = taq.MinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=qscheme,
            )

        return taq.QConfig(activation=act_obs, weight=weight_obs)
