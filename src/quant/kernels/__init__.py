"""
INT4 kernel utilities for GPTQ and AWQ backends.

Exports:
    pack_int4              — Pack [N, K] uint4 values → [N, K//8] int32.
    unpack_int4            — Inverse of pack_int4.
    quantize_to_uint4      — Float weight → unsigned INT4 grid values.
    compute_groupwise_qparams — Per-group (scale, qzeros) for a weight matrix.
    int4_dequant_gemm      — INT4 dequant + GEMM (Triton on CUDA, PyTorch on CPU).
    dequant_int4           — Materialise [N, K] float16 from packed int4.
"""

from .int4_packing import (
    pack_int4,
    unpack_int4,
    quantize_to_uint4,
    compute_groupwise_qparams,
)
from .triton_int4_gemm import int4_dequant_gemm, dequant_int4

__all__ = [
    "pack_int4",
    "unpack_int4",
    "quantize_to_uint4",
    "compute_groupwise_qparams",
    "int4_dequant_gemm",
    "dequant_int4",
]
