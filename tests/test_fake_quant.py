"""Tests for fake quantization correctness."""

from __future__ import annotations

import pytest
import torch

from src.quant.fake_quant import FakeQuantize, fake_quantize


class TestFakeQuantizeFunction:
    def test_output_shape_preserved(self):
        x = torch.randn(3, 4, 8)
        scale = torch.tensor(0.01)
        zp = torch.tensor(0, dtype=torch.int32)
        out = fake_quantize(x, scale, zp, num_bits=8, symmetric=True)
        assert out.shape == x.shape

    def test_dtype_preserved(self):
        x = torch.randn(10, dtype=torch.float32)
        scale = torch.tensor(0.01)
        zp = torch.tensor(0, dtype=torch.int32)
        out = fake_quantize(x, scale, zp, num_bits=8, symmetric=True)
        assert out.dtype == x.dtype

    def test_symmetric_clipping(self):
        """Values far outside range should be clipped to [quant_min, quant_max] * scale.
        Symmetric signed INT8: quant_min = -128, quant_max = 127."""
        x = torch.tensor([1000.0, -1000.0])
        scale = torch.tensor(0.01)  # max representable: 127 * 0.01 = 1.27; min: -128 * 0.01 = -1.28
        zp = torch.tensor(0, dtype=torch.int32)
        out = fake_quantize(x, scale, zp, num_bits=8, symmetric=True)
        assert float(out[0]) <= 127 * 0.01 + 1e-5
        assert float(out[1]) >= -128 * 0.01 - 1e-5

    def test_zero_input(self):
        """Zero input should always dequantize to zero (symmetric and asymmetric)."""
        x = torch.zeros(5)
        scale = torch.tensor(0.1)
        zp_sym = torch.tensor(0, dtype=torch.int32)
        zp_asym = torch.tensor(128, dtype=torch.int32)
        out_sym = fake_quantize(x, scale, zp_sym, num_bits=8, symmetric=True)
        out_asym = fake_quantize(x, scale, zp_asym, num_bits=8, symmetric=False)
        assert torch.allclose(out_sym, x)
        assert torch.allclose(out_asym, x, atol=scale.item())

    def test_per_channel_output_shape(self):
        """Per-channel quantization must preserve input shape."""
        x = torch.randn(4, 8)
        scale = torch.rand(4) * 0.05 + 0.005
        zp = torch.zeros(4, dtype=torch.int32)
        out = fake_quantize(
            x, scale, zp, num_bits=8, symmetric=True, per_channel=True, channel_axis=0
        )
        assert out.shape == x.shape

    def test_high_bits_low_error(self):
        """At 16 bits, quantization error should be negligible."""
        x = torch.randn(100)
        scale = torch.tensor(1e-4)
        zp = torch.tensor(0, dtype=torch.int32)
        out = fake_quantize(x, scale, zp, num_bits=16, symmetric=True)
        assert float((out - x).abs().max()) < float(scale) + 1e-6

    def test_asymmetric_unsigned_range(self):
        """Asymmetric mode should use [0, 255] integer range."""
        x = torch.tensor([0.0, 0.5, 1.0])
        scale = torch.tensor(1.0 / 255.0)
        zp = torch.tensor(0, dtype=torch.int32)
        out = fake_quantize(x, scale, zp, num_bits=8, symmetric=False)
        assert out.shape == x.shape
        # No value should exceed 1.0 + epsilon
        assert float(out.max()) <= 1.0 + scale.item() + 1e-6


class TestFakeQuantizeModule:
    def test_passthrough_when_disabled(self):
        fq = FakeQuantize(num_bits=8)
        fq.enabled = False
        x = torch.randn(10)
        assert torch.equal(fq(x), x)

    def test_set_qparams_updates_buffers(self):
        fq = FakeQuantize(num_bits=8)
        scale = torch.tensor(0.05)
        zp = torch.tensor(10, dtype=torch.int32)
        fq.set_qparams(scale, zp)
        assert float(fq.scale) == pytest.approx(0.05)
        assert int(fq.zero_point) == 10

    def test_output_close_to_input(self):
        """Dequantized output should be close to float input."""
        fq = FakeQuantize(num_bits=8, symmetric=True)
        fq.set_qparams(torch.tensor(0.01), torch.tensor(0, dtype=torch.int32))
        x = torch.tensor([0.1, -0.2, 0.05, 0.0])
        out = fq(x)
        # Error should be at most half an LSB = 0.005
        assert float((out - x).abs().max()) < 0.01

    def test_per_channel_module(self):
        """Per-channel FakeQuantize should work on 2-D weight tensors."""
        fq = FakeQuantize(num_bits=8, symmetric=True, per_channel=True, channel_axis=0)
        scale = torch.rand(8) * 0.01 + 0.001
        zp = torch.zeros(8, dtype=torch.int32)
        fq.set_qparams(scale, zp)
        w = torch.randn(8, 16)
        out = fq(w)
        assert out.shape == w.shape
