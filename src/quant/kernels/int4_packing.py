"""
INT4 weight packing / unpacking and per-group quantization utilities.

Wire format (compatible with AutoGPTQ / autoawq / vLLM marlin):
────────────────────────────────────────────────────────────────
  qweight   [out_features, in_features // 8]   int32
            8 consecutive INT4 values packed into one int32.
            Bit layout (low-nibble first):
              bits[ 3: 0] = col 0    bits[ 7: 4] = col 1
              bits[11: 8] = col 2    bits[15:12] = col 3
              bits[19:16] = col 4    bits[23:20] = col 5
              bits[27:24] = col 6    bits[31:28] = col 7

  scales    [n_groups, out_features]            float32
            n_groups = ceil(in_features / group_size)

  qzeros    [n_groups, out_features]            int32
            Per-output-channel zero-point per group.
            Symmetric  INT4 → qzeros = 8   (= 2^(bits-1))
            Asymmetric INT4 → qzeros = computed zero-point

Dequantize formula:
    W_fp[n, k] = (qweight_unpacked[n, k] − qzeros[g, n]) × scales[g, n]
    where g = k // group_size

Group size notes:
    group_size must be divisible by 8 (so all 8 nibbles in one int32 share
    the same group).  Standard values: 32, 64, 128 (default), 256.
    Set group_size = in_features for per-channel quantization (n_groups = 1).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


# ---------------------------------------------------------------------------
# Bit-packing helpers
# ---------------------------------------------------------------------------

def pack_int4(W_uint: torch.Tensor) -> torch.Tensor:
    """
    Pack a [out_features, in_features] uint4 tensor into [out_features, in_features//8]
    int32, 8 INT4 values per int32 (low nibble = column 0).

    Args:
        W_uint: Integer tensor with values in [0, 15].  Shape [N, K].
                K must be divisible by 8.

    Returns:
        qweight: int32 tensor of shape [N, K // 8].
    """
    assert W_uint.shape[1] % 8 == 0, (
        f"in_features ({W_uint.shape[1]}) must be divisible by 8 for INT4 packing"
    )
    N, K = W_uint.shape
    W = W_uint.to(torch.int32)
    packed = torch.zeros(N, K // 8, dtype=torch.int32, device=W.device)
    for i in range(8):
        packed |= (W[:, i::8] & 0xF) << (i * 4)
    return packed


def unpack_int4(qweight: torch.Tensor, K: int) -> torch.Tensor:
    """
    Unpack [out_features, K//8] int32 → [out_features, K] int32 (values 0–15).

    Args:
        qweight: Packed weight tensor of shape [N, K//8].
        K:       Original (unpacked) in_features dimension.

    Returns:
        W_uint: int32 tensor of shape [N, K], values in [0, 15].
    """
    N = qweight.shape[0]
    W = torch.zeros(N, K, dtype=torch.int32, device=qweight.device)
    for i in range(8):
        W[:, i::8] = (qweight >> (i * 4)) & 0xF
    return W


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def quantize_to_uint4(
    W: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
    num_bits: int = 4,
) -> torch.Tensor:
    """
    Quantize float32 weight matrix to unsigned integers [0, 2^bits - 1].

    Args:
        W:          [out_features, in_features] float32.
        scales:     [n_groups, out_features] float32.
        qzeros:     [n_groups, out_features] int32 zero-points.
                    Symmetric: qzeros = 2^(bits-1) = 8.
                    Asymmetric: qzeros = per-channel zero-point.
        group_size: Number of input features per quantization group.
        num_bits:   Bit-width (default 4).

    Returns:
        W_uint: [out_features, in_features] int32, values in [0, 2^bits - 1].
    """
    N, K = W.shape
    n_groups = scales.shape[0]
    quant_max = (1 << num_bits) - 1   # 15 for INT4
    W_uint = torch.empty(N, K, dtype=torch.int32, device=W.device)

    for g in range(n_groups):
        k0 = g * group_size
        k1 = min(k0 + group_size, K)
        s = scales[g, :].to(W.dtype).unsqueeze(1)       # [N, 1]
        z = qzeros[g, :].to(W.dtype).unsqueeze(1)       # [N, 1]
        W_block = W[:, k0:k1]                           # [N, gs]
        # Reconstruct: W_uint = round(W / scale + zero).clamp(0, quant_max)
        W_uint[:, k0:k1] = (W_block / s + z).round().clamp(0, quant_max).to(torch.int32)

    return W_uint


def compute_groupwise_qparams(
    W: torch.Tensor,
    group_size: int,
    num_bits: int,
    symmetric: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-group, per-output-channel quantization parameters.

    Args:
        W:          [out_features, in_features] float32.
        group_size: Number of input features per group.  Use ``in_features``
                    for standard per-channel (single-group) quantization.
        num_bits:   Quantization bit-width (usually 4 or 8).
        symmetric:  True → zero-point = 2^(bits-1); False → asymmetric.

    Returns:
        scales:  [n_groups, out_features] float32.
        qzeros:  [n_groups, out_features] int32.
                 Symmetric: all 8 (= 2^(bits-1)).
                 Asymmetric: computed from per-group min/max.

    Notes:
        ``group_size`` must be divisible by 8 so all nibbles in a packed
        int32 share the same group.
    """
    assert group_size % 8 == 0, (
        f"group_size ({group_size}) must be divisible by 8"
    )
    N, K = W.shape
    n_groups = math.ceil(K / group_size)
    quant_max = (1 << num_bits) - 1               # 15 for INT4
    quant_half = 1 << (num_bits - 1)              # 8  for INT4 (= 2^(bits-1))

    scales = torch.empty(n_groups, N, dtype=torch.float32, device=W.device)
    qzeros = torch.empty(n_groups, N, dtype=torch.int32,   device=W.device)

    W_f = W.float()
    for g in range(n_groups):
        k0 = g * group_size
        k1 = min(k0 + group_size, K)
        W_block = W_f[:, k0:k1]   # [N, gs]

        if symmetric:
            abs_max = W_block.abs().amax(dim=1).clamp(min=1e-8)  # [N]
            s = abs_max / (quant_half - 1)                        # [N]
            z = torch.full((N,), quant_half, dtype=torch.int32, device=W.device)
        else:
            w_min = W_block.amin(dim=1)
            w_max = W_block.amax(dim=1)
            w_min = torch.minimum(w_min, torch.zeros_like(w_min))
            w_max = torch.maximum(w_max, torch.zeros_like(w_max))
            s = ((w_max - w_min) / quant_max).clamp(min=1e-8)    # [N]
            z = (-w_min / s).round().to(torch.int32)              # [N]

        scales[g] = s
        qzeros[g] = z

    return scales, qzeros


# ---------------------------------------------------------------------------
# Vectorised dequantisation (CPU helper for the GEMM wrapper)
# ---------------------------------------------------------------------------

def dequant_weight_cpu(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    Dequantize [N, K//8] int32 packed weights → [N, K] float32.

    Args:
        qweight:    [N, K//8] int32 packed.
        scales:     [n_groups, N] float32.
        qzeros:     [n_groups, N] int32.
        group_size: Number of input features per group.

    Returns:
        W_fp: [N, K] float32.
    """
    N, Kp = qweight.shape
    K = Kp * 8

    # Expand scales / qzeros from [n_groups, N] → [N, K] by repeating per group
    g_idx = torch.arange(K, device=qweight.device) // group_size   # [K]
    # scales[g_idx[k], n] → index as scales.T[:, g_idx] then transpose
    # scales: [G, N], g_idx: [K]
    scales_exp = scales[g_idx, :].T.float()    # [N, K]
    qzeros_exp = qzeros[g_idx, :].T.float()    # [N, K]

    W_uint = unpack_int4(qweight, K).float()   # [N, K]
    return (W_uint - qzeros_exp) * scales_exp  # [N, K]
