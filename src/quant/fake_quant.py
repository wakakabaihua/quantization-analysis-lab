"""
Fake quantization: quantize then immediately dequantize in floating point.

Fake quantization simulates the effect of integer quantization while keeping
all arithmetic in float32. This lets us measure quantization error without
requiring actual integer compute kernels.

Quantization formula (symmetric, per-tensor):
    x_q  = clamp(round(x / scale), quant_min, quant_max)
    x_dq = (x_q - zero_point) * scale

The output x_dq has quantization error but the same dtype as x.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def fake_quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    num_bits: int,
    symmetric: bool = True,
    per_channel: bool = False,
    channel_axis: int = 0,
) -> torch.Tensor:
    """
    Apply fake quantization to a tensor.

    Args:
        x:            Input tensor (any shape).
        scale:        Scale factor(s). Shape [1] for per-tensor, [C] for per-channel.
        zero_point:   Zero-point(s). Same shape as scale.
        num_bits:     Quantization bit-width.
        symmetric:    If True use signed range [-2^(b-1), 2^(b-1)-1];
                      if False use unsigned range [0, 2^b - 1].
        per_channel:  If True, scale/zp are per-channel along channel_axis.
        channel_axis: Axis for per-channel quantization.

    Returns:
        Dequantized tensor with the same shape and dtype as x.
    """
    if symmetric:
        quant_min = -(2 ** (num_bits - 1))
        quant_max = 2 ** (num_bits - 1) - 1
    else:
        quant_min = 0
        quant_max = 2**num_bits - 1

    if per_channel:
        # Reshape scale/zp for broadcasting along channel_axis
        view_shape = [1] * x.ndim
        view_shape[channel_axis] = -1
        scale = scale.view(view_shape)
        zero_point = zero_point.view(view_shape)

    x_float = x.float()
    # Quantize: round to nearest integer, clamp to representable range
    x_q = torch.clamp(
        torch.round(x_float / scale) + zero_point.float(),
        quant_min,
        quant_max,
    )
    # Dequantize back to floating point
    x_dq = (x_q - zero_point.float()) * scale
    return x_dq.to(x.dtype)


class FakeQuantize(nn.Module):
    """
    Module wrapper for fake_quantize.

    Holds scale and zero_point as buffers so they survive device moves and
    state_dict serialization. Quantization parameters are set externally
    by the PTQ pipeline after calibration via set_qparams().

    Set `enabled = False` to disable quantization (identity passthrough).
    """

    def __init__(
        self,
        num_bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
        channel_axis: int = 0,
    ) -> None:
        super().__init__()
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.channel_axis = channel_axis
        self.enabled = True

        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0, dtype=torch.int32))

    def set_qparams(self, scale: torch.Tensor, zero_point: torch.Tensor) -> None:
        """Update scale and zero_point buffers."""
        self.scale = scale.to(dtype=torch.float32, device=self.scale.device)
        self.zero_point = zero_point.to(dtype=torch.int32, device=self.zero_point.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        return fake_quantize(
            x,
            self.scale,
            self.zero_point,
            self.num_bits,
            symmetric=self.symmetric,
            per_channel=self.per_channel,
            channel_axis=self.channel_axis,
        )

    def extra_repr(self) -> str:
        return (
            f"bits={self.num_bits}, symmetric={self.symmetric}, "
            f"per_channel={self.per_channel}, enabled={self.enabled}"
        )
