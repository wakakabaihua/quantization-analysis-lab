"""
Triton INT4 dequantization kernel and GEMM wrapper.

Architecture
────────────
The INT4 GEMM is split into two steps:

  1. **Dequantize** (Triton kernel on CUDA, pure PyTorch on CPU):
       qweight [N, K//8] int32  →  W_fp16 [N, K] float16
     Each packed int32 holds 8 INT4 nibbles (low-nibble-first).
     Scale and zero-point are applied per-group during unpacking.

  2. **GEMM** (cuBLAS via ``torch.mm`` on CUDA):
       out [M, N] = x [M, K] fp16  ×  W_fp16 [K, N] fp16

This "dequant-then-mm" pattern avoids materialising fp32 weight tensors
(INT4 packed → FP16 is 4× smaller than FP32) while leveraging highly
optimised tensor-core GEMM from cuBLAS.

For production, a fused dequant+GEMM kernel (e.g. ExLlama v2 / Marlin)
eliminates the intermediate FP16 materialisation.  The architecture here
is designed so swapping in a fused kernel requires only replacing the body
of ``int4_dequant_gemm`` when running on CUDA.

Triton kernel design notes
──────────────────────────
  - Program grid: (ceil(N / BLOCK_N), ceil(K_packed / BLOCK_KP))
    where K_packed = K // 8 and BLOCK_KP = BLOCK_K // 8.
  - Constraint: BLOCK_K must divide group_size and BLOCK_K % 8 == 0.
    With the default BLOCK_K = 64 and group_size = 128 this holds.
  - Because group_size % 8 == 0, all 8 nibbles in a packed int32 belong
    to the same quantization group, so one scalar (scale, zero) pair
    covers the entire [BLOCK_N, BLOCK_KP] tile.
  - Inner nibble loop: ``tl.static_range(8)`` is unrolled at compile time;
    each iteration extracts one nibble column-slice and stores to output.
  - Output dtype is float16 to feed directly into ``torch.mm``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .int4_packing import dequant_weight_cpu

# ---------------------------------------------------------------------------
# Optional Triton import
# ---------------------------------------------------------------------------

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Triton dequantisation kernel
# ---------------------------------------------------------------------------

if _HAS_TRITON:
    @triton.jit
    def _dequant_int4_kernel(
        # Packed weight  [N, K_packed]  int32
        qw_ptr,
        # Scales          [G, N]         float32
        sc_ptr,
        # Zero-points     [G, N]         int32
        zp_ptr,
        # Output          [N, K]         float16
        out_ptr,
        # Dimensions
        N, K_packed,
        # Strides for qweight (row-major)
        stride_qwn, stride_qwk,
        # Strides for scales / zeros (row = group, col = out-channel)
        stride_sg, stride_sn,
        stride_zg, stride_zn,
        # Strides for output
        stride_on, stride_ok,
        # Compile-time constants
        GROUP_SIZE_PACKED: tl.constexpr,   # group_size // 8
        BLOCK_N:           tl.constexpr,
        BLOCK_KP:          tl.constexpr,   # BLOCK_K // 8; must be <= GROUP_SIZE_PACKED
    ):
        """
        Each program instance writes a [BLOCK_N, BLOCK_K] tile of the
        dequantized float16 output matrix.

        Invariant: BLOCK_KP <= GROUP_SIZE_PACKED so every (n, kp) pair in the
        tile shares a single (scale, zero) pair.
        """
        pid_n  = tl.program_id(0)
        pid_kp = tl.program_id(1)

        offs_n  = pid_n  * BLOCK_N  + tl.arange(0, BLOCK_N)   # [BLOCK_N]
        offs_kp = pid_kp * BLOCK_KP + tl.arange(0, BLOCK_KP)  # [BLOCK_KP]

        n_mask  = offs_n  < N
        kp_mask = offs_kp < K_packed

        # Group index: all kp in this tile share the same group because
        #   BLOCK_KP <= GROUP_SIZE_PACKED and pid_kp * BLOCK_KP is
        #   GROUP_SIZE_PACKED-aligned (tile boundaries align with groups).
        g = pid_kp * BLOCK_KP // GROUP_SIZE_PACKED

        # Load scale and zero-point: [BLOCK_N]
        scale = tl.load(
            sc_ptr + g * stride_sg + offs_n * stride_sn,
            mask=n_mask, other=1.0,
        ).to(tl.float32)
        zero = tl.load(
            zp_ptr + g * stride_zg + offs_n * stride_zn,
            mask=n_mask, other=0.0,
        ).to(tl.float32)

        # Load packed weights: [BLOCK_N, BLOCK_KP]
        qw = tl.load(
            qw_ptr + offs_n[:, None] * stride_qwn + offs_kp[None, :] * stride_qwk,
            mask=n_mask[:, None] & kp_mask[None, :],
            other=0,
        ).to(tl.int32)

        # Unpack 8 nibbles and dequantize each nibble-slice.
        # Nibble j corresponds to output columns (kp * 8 + j).
        for j in tl.static_range(8):
            w_int = (qw >> (j * 4)) & 0xF              # [BLOCK_N, BLOCK_KP]
            w_fp  = (w_int.to(tl.float32) - zero[:, None]) * scale[:, None]
            # Output column indices for this nibble: offs_kp * 8 + j
            offs_k = offs_kp * 8 + j                   # [BLOCK_KP]
            k_mask = offs_k < (K_packed * 8)
            tl.store(
                out_ptr + offs_n[:, None] * stride_on + offs_k[None, :] * stride_ok,
                w_fp.to(tl.float16),
                mask=n_mask[:, None] & k_mask[None, :],
            )


def _triton_dequant(
    qweight: torch.Tensor,
    scales:  torch.Tensor,
    qzeros:  torch.Tensor,
    K:       int,
    group_size: int,
) -> torch.Tensor:
    """
    Run the Triton dequant kernel.  Returns [N, K] float16 on the same device.
    """
    assert qweight.is_cuda, "Triton dequant requires a CUDA tensor"
    N, K_packed = qweight.shape
    G = scales.shape[0]
    GROUP_SIZE_PACKED = group_size // 8

    out = torch.empty(N, K, dtype=torch.float16, device=qweight.device)

    # BLOCK_KP: largest power-of-2 that fits within GROUP_SIZE_PACKED and
    # satisfies Triton's minimum tl.store width (≥ 1).
    BLOCK_KP = min(GROUP_SIZE_PACKED, 16)   # 16 → BLOCK_K = 128 (matches typical group_size)
    BLOCK_N  = 16

    grid = (
        triton.cdiv(N,       BLOCK_N),
        triton.cdiv(K_packed, BLOCK_KP),
    )

    _dequant_int4_kernel[grid](
        qweight, scales, qzeros, out,
        N, K_packed,
        qweight.stride(0), qweight.stride(1),
        scales.stride(0),  scales.stride(1),
        qzeros.stride(0),  qzeros.stride(1),
        out.stride(0),     out.stride(1),
        GROUP_SIZE_PACKED=GROUP_SIZE_PACKED,
        BLOCK_N=BLOCK_N,
        BLOCK_KP=BLOCK_KP,
    )
    return out   # [N, K] float16


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dequant_int4(
    qweight:    torch.Tensor,
    scales:     torch.Tensor,
    qzeros:     torch.Tensor,
    K:          int,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Materialise a dequantised float weight matrix from packed INT4 storage.

    On CUDA with Triton installed → runs the Triton dequant kernel → float16.
    Otherwise → pure-PyTorch CPU path → float32.

    Args:
        qweight:    [N, K//8] int32 packed weight.
        scales:     [G, N]    float32 per-group scales.
        qzeros:     [G, N]    int32  per-group zero-points.
        K:          Unpacked in_features dimension.
        group_size: Number of input features per quantization group.

    Returns:
        W_fp: [N, K] float16 (CUDA) or float32 (CPU).
    """
    if qweight.is_cuda and _HAS_TRITON:
        return _triton_dequant(qweight, scales, qzeros, K, group_size)
    else:
        return dequant_weight_cpu(qweight, scales, qzeros, group_size)


def int4_dequant_gemm(
    x:          torch.Tensor,
    qweight:    torch.Tensor,
    scales:     torch.Tensor,
    qzeros:     torch.Tensor,
    group_size: int = 128,
    bias:       torch.Tensor | None = None,
) -> torch.Tensor:
    """
    INT4 weight dequantization + matrix multiplication.

    Computes:  out = x  @  dequant(qweight)ᵀ  +  bias

    CUDA path (Triton available):
        1. Triton kernel dequantizes qweight → W_fp16 [N, K].
        2. x cast to float16; torch.mm(x_fp16, W_fp16.T) via cuBLAS.
        3. Result cast back to x.dtype.

    CPU / no-Triton path:
        Unpack and dequantize with PyTorch, then F.linear in float32.

    Args:
        x:          [..., K] float32 or float16 input activations.
        qweight:    [N, K//8] int32 packed weight matrix.
        scales:     [G, N] float32 per-group scales.
        qzeros:     [G, N] int32 per-group zero-points.
        group_size: Quantization group size (default 128).
        bias:       [N] optional bias tensor.

    Returns:
        out: [..., N] same dtype as x.
    """
    N, Kp = qweight.shape
    K = Kp * 8
    orig_shape = x.shape
    orig_dtype = x.dtype
    x_2d = x.reshape(-1, K)     # [M, K]

    if x.is_cuda and _HAS_TRITON:
        # --- Triton dequant + cuBLAS GEMM ---
        W_fp16 = _triton_dequant(qweight, scales, qzeros, K, group_size)  # [N, K] fp16
        x_fp16 = x_2d.to(torch.float16)
        out = torch.mm(x_fp16, W_fp16.T)                                  # [M, N] fp16
        out = out.to(orig_dtype)
    else:
        # --- Pure-PyTorch CPU fallback ---
        W_fp32 = dequant_weight_cpu(qweight, scales, qzeros, group_size)  # [N, K] fp32
        out = F.linear(x_2d.float(), W_fp32).to(orig_dtype)               # [M, N]

    if bias is not None:
        out = out + bias.to(orig_dtype)

    return out.reshape(*orig_shape[:-1], N)
