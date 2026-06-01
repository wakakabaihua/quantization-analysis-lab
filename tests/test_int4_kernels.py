"""
Tests for the INT4 kernel utilities: packing, dequantization, and GEMM.

Covers:
  - pack_int4 / unpack_int4: lossless round-trip for all 4-bit values.
  - quantize_to_uint4: float32 → unsigned INT4 grid, correct clamping.
  - compute_groupwise_qparams: correct scale / qzero shapes and semantics.
  - dequant_weight_cpu: matches manual dequantization formula.
  - int4_dequant_gemm (CPU path): output matches F.linear on dequantized weights.
  - AWQLinear (INT4 packed mode): forward produces correct output.
  - GPTQLinear (INT4 packed mode): forward produces correct output.
  - Numerical equivalence: pack → unpack → dequant == original fake-quant.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from src.quant.kernels import (
    pack_int4,
    unpack_int4,
    quantize_to_uint4,
    compute_groupwise_qparams,
    int4_dequant_gemm,
)
from src.quant.kernels.int4_packing import dequant_weight_cpu
from src.quant.backends import AWQLinear, GPTQLinear

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_weight(N: int, K: int, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(N, K)


# ---------------------------------------------------------------------------
# pack_int4 / unpack_int4
# ---------------------------------------------------------------------------

class TestInt4PackUnpack:
    def test_roundtrip_all_nibbles(self):
        """pack_int4 followed by unpack_int4 is lossless for values in [0, 15]."""
        N, K = 8, 32
        W_uint = torch.randint(0, 16, (N, K), dtype=torch.int32)
        packed = pack_int4(W_uint)
        recovered = unpack_int4(packed, K)
        assert torch.equal(W_uint, recovered), "pack/unpack round-trip failed"

    def test_packed_shape(self):
        N, K = 16, 64
        W_uint = torch.zeros(N, K, dtype=torch.int32)
        packed = pack_int4(W_uint)
        assert packed.shape == (N, K // 8)

    def test_packed_dtype(self):
        W_uint = torch.zeros(4, 16, dtype=torch.int32)
        assert pack_int4(W_uint).dtype == torch.int32

    def test_nibble_values_preserved(self):
        """Verify each nibble position is correctly packed and unpacked."""
        N, K = 4, 8
        # One distinct value per nibble position
        W_uint = torch.arange(16, dtype=torch.int32).unsqueeze(0).repeat(N, 1)[:, :K]
        packed = pack_int4(W_uint)
        recovered = unpack_int4(packed, K)
        assert torch.equal(W_uint, recovered)

    def test_zero_values(self):
        W_uint = torch.zeros(8, 16, dtype=torch.int32)
        packed = pack_int4(W_uint)
        recovered = unpack_int4(packed, 16)
        assert torch.all(recovered == 0)

    def test_max_values(self):
        W_uint = torch.full((8, 16), 15, dtype=torch.int32)
        packed = pack_int4(W_uint)
        recovered = unpack_int4(packed, 16)
        assert torch.all(recovered == 15)


# ---------------------------------------------------------------------------
# quantize_to_uint4
# ---------------------------------------------------------------------------

class TestQuantizeToUint4:
    def test_output_range_symmetric(self):
        """Symmetric quantized values must lie in [0, 15]."""
        W = _rand_weight(16, 64)
        scales, qzeros = compute_groupwise_qparams(W, group_size=64,
                                                   num_bits=4, symmetric=True)
        W_uint = quantize_to_uint4(W, scales, qzeros, group_size=64)
        assert W_uint.min() >= 0
        assert W_uint.max() <= 15

    def test_output_range_asymmetric(self):
        """Asymmetric quantized values must lie in [0, 15]."""
        W = _rand_weight(16, 64)
        scales, qzeros = compute_groupwise_qparams(W, group_size=64,
                                                   num_bits=4, symmetric=False)
        W_uint = quantize_to_uint4(W, scales, qzeros, group_size=64)
        assert W_uint.min() >= 0
        assert W_uint.max() <= 15

    def test_dtype_is_int32(self):
        W = _rand_weight(8, 32)
        scales, qzeros = compute_groupwise_qparams(W, group_size=32, num_bits=4, symmetric=True)
        W_uint = quantize_to_uint4(W, scales, qzeros, group_size=32)
        assert W_uint.dtype == torch.int32

    def test_shape_preserved(self):
        N, K = 12, 48
        W = _rand_weight(N, K)
        scales, qzeros = compute_groupwise_qparams(W, group_size=48, num_bits=4, symmetric=True)
        W_uint = quantize_to_uint4(W, scales, qzeros, group_size=48)
        assert W_uint.shape == (N, K)


# ---------------------------------------------------------------------------
# compute_groupwise_qparams
# ---------------------------------------------------------------------------

class TestComputeGroupwiseQparams:
    def test_scales_shape_single_group(self):
        N, K = 16, 64
        W = _rand_weight(N, K)
        scales, qzeros = compute_groupwise_qparams(W, group_size=K,
                                                   num_bits=4, symmetric=True)
        assert scales.shape == (1, N)
        assert qzeros.shape == (1, N)

    def test_scales_shape_multi_group(self):
        N, K, gs = 16, 128, 32
        W = _rand_weight(N, K)
        scales, qzeros = compute_groupwise_qparams(W, group_size=gs,
                                                   num_bits=4, symmetric=True)
        n_groups = K // gs
        assert scales.shape == (n_groups, N)
        assert qzeros.shape == (n_groups, N)

    def test_symmetric_qzero_is_eight(self):
        """Symmetric INT4 qzeros should equal 8 (= 2^(bits-1))."""
        W = _rand_weight(8, 32)
        _, qzeros = compute_groupwise_qparams(W, group_size=32,
                                              num_bits=4, symmetric=True)
        assert torch.all(qzeros == 8), f"Expected all-8 qzeros, got {qzeros}"

    def test_asymmetric_qzero_is_non_negative(self):
        W = _rand_weight(8, 32)
        _, qzeros = compute_groupwise_qparams(W, group_size=32,
                                              num_bits=4, symmetric=False)
        assert torch.all(qzeros >= 0)

    def test_scales_positive(self):
        W = _rand_weight(8, 32)
        scales, _ = compute_groupwise_qparams(W, group_size=32,
                                              num_bits=4, symmetric=True)
        assert torch.all(scales > 0)


# ---------------------------------------------------------------------------
# dequant_weight_cpu
# ---------------------------------------------------------------------------

class TestDequantWeightCpu:
    def test_matches_manual_dequant(self):
        """dequant_weight_cpu should reproduce (W_uint - qzero) * scale."""
        N, K, gs = 8, 32, 32
        W = _rand_weight(N, K)
        scales, qzeros = compute_groupwise_qparams(W, group_size=gs,
                                                   num_bits=4, symmetric=True)
        W_uint = quantize_to_uint4(W, scales, qzeros, group_size=gs)
        qweight = pack_int4(W_uint)

        W_dequant = dequant_weight_cpu(qweight, scales, qzeros, group_size=gs)

        # Manual dequant reference
        group_idx  = torch.arange(K) // gs
        scales_exp = scales[group_idx, :].T   # [N, K]
        zeros_exp  = qzeros[group_idx, :].T.float()
        W_ref = (W_uint.float() - zeros_exp) * scales_exp.float()

        assert torch.allclose(W_dequant.float(), W_ref, atol=1e-5)

    def test_output_shape(self):
        N, K, gs = 12, 48, 48
        W = _rand_weight(N, K)
        scales, qzeros = compute_groupwise_qparams(W, group_size=gs, num_bits=4, symmetric=True)
        W_uint  = quantize_to_uint4(W, scales, qzeros, group_size=gs)
        qweight = pack_int4(W_uint)
        W_dq = dequant_weight_cpu(qweight, scales, qzeros, group_size=gs)
        assert W_dq.shape == (N, K)


# ---------------------------------------------------------------------------
# int4_dequant_gemm (CPU path)
# ---------------------------------------------------------------------------

class TestInt4DequantGemm:
    def _quantize_and_pack(self, W: torch.Tensor, group_size: int, symmetric: bool):
        scales, qzeros = compute_groupwise_qparams(W, group_size=group_size,
                                                   num_bits=4, symmetric=symmetric)
        W_uint  = quantize_to_uint4(W, scales, qzeros, group_size=group_size)
        qweight = pack_int4(W_uint)
        return qweight, scales, qzeros

    def test_output_shape(self):
        N, K, M = 16, 64, 8
        W = _rand_weight(N, K)
        x = torch.randn(M, K)
        qweight, scales, qzeros = self._quantize_and_pack(W, K, symmetric=True)
        out = int4_dequant_gemm(x, qweight, scales, qzeros, group_size=K)
        assert out.shape == (M, N)

    def test_3d_input(self):
        """int4_dequant_gemm must handle batched 3-D inputs [B, S, K]."""
        N, K, B, S = 16, 32, 2, 8
        W = _rand_weight(N, K)
        x = torch.randn(B, S, K)
        qweight, scales, qzeros = self._quantize_and_pack(W, K, symmetric=True)
        out = int4_dequant_gemm(x, qweight, scales, qzeros, group_size=K)
        assert out.shape == (B, S, N)

    def test_symmetric_matches_fake_quant(self):
        """
        int4_dequant_gemm output must match F.linear(x, W_q, None) where W_q
        is the fake-quantised float32 weight.
        """
        N, K = 16, 64
        W = _rand_weight(N, K)
        x = torch.randn(4, K)
        gs = K

        scales, qzeros = compute_groupwise_qparams(W, group_size=gs,
                                                   num_bits=4, symmetric=True)
        W_uint  = quantize_to_uint4(W, scales, qzeros, group_size=gs)
        qweight = pack_int4(W_uint)

        # Reference: dequantise and run F.linear
        W_ref = dequant_weight_cpu(qweight, scales, qzeros, group_size=gs)
        out_ref = F.linear(x, W_ref)

        out_kernel = int4_dequant_gemm(x, qweight, scales, qzeros, group_size=gs)
        assert torch.allclose(out_kernel.float(), out_ref.float(), atol=1e-4), (
            f"max abs diff: {(out_kernel.float() - out_ref.float()).abs().max():.6f}"
        )

    def test_asymmetric_matches_fake_quant(self):
        N, K = 16, 64
        W = _rand_weight(N, K, seed=99)
        x = torch.randn(4, K)
        gs = K

        scales, qzeros = compute_groupwise_qparams(W, group_size=gs,
                                                   num_bits=4, symmetric=False)
        W_uint  = quantize_to_uint4(W, scales, qzeros, group_size=gs)
        qweight = pack_int4(W_uint)
        W_ref = dequant_weight_cpu(qweight, scales, qzeros, group_size=gs)
        out_ref    = F.linear(x, W_ref)
        out_kernel = int4_dequant_gemm(x, qweight, scales, qzeros, group_size=gs)
        assert torch.allclose(out_kernel.float(), out_ref.float(), atol=1e-4)

    def test_with_bias(self):
        N, K = 8, 32
        W = _rand_weight(N, K)
        x = torch.randn(3, K)
        bias = torch.randn(N)
        qweight, scales, qzeros = self._quantize_and_pack(W, K, symmetric=True)
        W_ref = dequant_weight_cpu(qweight, scales, qzeros, group_size=K)
        out_ref    = F.linear(x, W_ref, bias)
        out_kernel = int4_dequant_gemm(x, qweight, scales, qzeros,
                                       group_size=K, bias=bias)
        assert torch.allclose(out_kernel.float(), out_ref.float(), atol=1e-4)

    def test_multi_group(self):
        """int4_dequant_gemm must handle multiple quantization groups."""
        N, K, gs = 16, 128, 32  # 4 groups
        W = _rand_weight(N, K)
        x = torch.randn(4, K)
        scales, qzeros = compute_groupwise_qparams(W, group_size=gs,
                                                   num_bits=4, symmetric=True)
        W_uint  = quantize_to_uint4(W, scales, qzeros, group_size=gs)
        qweight = pack_int4(W_uint)
        W_ref = dequant_weight_cpu(qweight, scales, qzeros, group_size=gs)
        out_ref    = F.linear(x, W_ref)
        out_kernel = int4_dequant_gemm(x, qweight, scales, qzeros, group_size=gs)
        assert torch.allclose(out_kernel.float(), out_ref.float(), atol=1e-4)


# ---------------------------------------------------------------------------
# Numerical equivalence: packed INT4 == fake-quant float32
# ---------------------------------------------------------------------------

class TestNumericalEquivalence:
    """
    Verify that the INT4 packed format reproduces the same output as the
    original float32 fake-quantized linear layer.

    This is the key correctness guarantee for the Phase 6 refactor.
    """

    def test_awq_packed_matches_fakequant_symmetric(self):
        """
        AWQLinear (packed INT4) forward ≈ F.linear(x/s, W_q_float, bias)
        where W_q_float is the symmetric fake-quantised weight.
        """
        d_in, d_out = 64, 32
        W = _rand_weight(d_out, d_in)
        s = torch.rand(d_in).clamp(0.5, 2.0)  # non-trivial AWQ scale
        W_scaled = W * s.unsqueeze(0)
        x = torch.randn(4, 8, d_in)
        bias = torch.randn(d_out)
        gs = d_in

        # Build float32 reference: fake-quant then F.linear
        scales, qzeros = compute_groupwise_qparams(W_scaled, group_size=gs,
                                                   num_bits=4, symmetric=True)
        W_uint  = quantize_to_uint4(W_scaled, scales, qzeros, group_size=gs)
        W_ref   = dequant_weight_cpu(pack_int4(W_uint), scales, qzeros, group_size=gs)
        out_ref = F.linear(x / s, W_ref, bias)

        # Build AWQLinear
        awq = AWQLinear.from_float(d_in, d_out, W_scaled, awq_scale=s,
                                   symmetric=True, bias=bias, group_size=gs)
        awq.eval()
        with torch.no_grad():
            out_awq = awq(x)

        assert torch.allclose(out_awq.float(), out_ref.float(), atol=1e-4), (
            f"AWQLinear packed INT4 ≠ float32 reference. "
            f"max diff = {(out_awq.float() - out_ref.float()).abs().max():.6f}"
        )

    def test_awq_packed_matches_fakequant_asymmetric(self):
        d_in, d_out = 64, 32
        W = _rand_weight(d_out, d_in, seed=7)
        s = torch.rand(d_in).clamp(0.5, 2.0)
        W_scaled = W * s.unsqueeze(0)
        x = torch.randn(4, 8, d_in)
        gs = d_in

        scales, qzeros = compute_groupwise_qparams(W_scaled, group_size=gs,
                                                   num_bits=4, symmetric=False)
        W_uint  = quantize_to_uint4(W_scaled, scales, qzeros, group_size=gs)
        W_ref   = dequant_weight_cpu(pack_int4(W_uint), scales, qzeros, group_size=gs)
        out_ref = F.linear(x / s, W_ref)

        awq = AWQLinear.from_float(d_in, d_out, W_scaled, awq_scale=s,
                                   symmetric=False, group_size=gs)
        awq.eval()
        with torch.no_grad():
            out_awq = awq(x)

        assert torch.allclose(out_awq.float(), out_ref.float(), atol=1e-4)

    def test_gptq_packed_matches_fakequant(self):
        """
        GPTQLinear (packed INT4) forward ≈ F.linear(x, W_dequant, bias)
        where W_dequant is reconstructed from the packed weights.
        """
        d_in, d_out = 64, 32
        W = _rand_weight(d_out, d_in, seed=13)
        x = torch.randn(4, 8, d_in)
        bias = torch.randn(d_out)
        gs = d_in

        scales, qzeros = compute_groupwise_qparams(W, group_size=gs,
                                                   num_bits=4, symmetric=True)
        W_uint  = quantize_to_uint4(W, scales, qzeros, group_size=gs)
        qweight = pack_int4(W_uint)
        W_ref   = dequant_weight_cpu(qweight, scales, qzeros, group_size=gs)
        out_ref = F.linear(x, W_ref, bias)

        gptq = GPTQLinear(d_in, d_out,
                          qweight=qweight, scales=scales, qzeros=qzeros,
                          group_size=gs, bias=bias)
        gptq.eval()
        with torch.no_grad():
            out_gptq = gptq(x)

        assert torch.allclose(out_gptq.float(), out_ref.float(), atol=1e-4), (
            f"GPTQLinear packed INT4 ≠ float32 reference. "
            f"max diff = {(out_gptq.float() - out_ref.float()).abs().max():.6f}"
        )


# ---------------------------------------------------------------------------
# Memory footprint: INT4 packed < float32
# ---------------------------------------------------------------------------

class TestMemoryFootprint:
    def test_qweight_4x_smaller_than_float32(self):
        """Packed INT4 qweight is 4× smaller in elements than float32 weight."""
        N, K = 128, 256
        W = _rand_weight(N, K)
        scales, qzeros = compute_groupwise_qparams(W, group_size=K, num_bits=4, symmetric=True)
        W_uint  = quantize_to_uint4(W, scales, qzeros, group_size=K)
        qweight = pack_int4(W_uint)
        # float32: N*K * 4 bytes  = N*K*4
        # int32 packed: N*(K//8) * 4 bytes = N*K//2 bytes  (→ 8× fewer bytes)
        # But element count: N*(K//8) vs N*K  → 8× fewer int32 elements
        assert qweight.numel() * 8 == N * K
        # Byte footprint: qweight uses (N*K//8)*4 = N*K//2 bytes vs N*K*4 bytes
        qweight_bytes = qweight.numel() * qweight.element_size()
        float32_bytes = N * K * 4
        assert qweight_bytes * 8 == float32_bytes, (
            "Packed INT4 should use 8× fewer bytes than float32"
        )
