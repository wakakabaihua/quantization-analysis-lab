"""
Attention block for quantization analysis.

Implements multi-head self-attention with separate Q, K, V projections
and an output projection. This isolates attention-specific quantization
sensitivity from the MLP block.

The module also returns intermediate tensors (Q, K, V, attention weights)
to enable detailed error analysis at each stage.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """
    Multi-head self-attention block.

    Projection structure: q_proj, k_proj, v_proj -> scaled dot-product -> out_proj.
    All four projections are nn.Linear and are therefore visible to the PTQ pipeline.

    Returns both the output and a dict of intermediate tensors so that
    quantization error can be decomposed across Q, K, V, and attention scores.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x:    (batch, seq_len, embed_dim)
            mask: (batch, seq_len, seq_len) optional boolean mask; True = keep

        Returns:
            output:        (batch, seq_len, embed_dim)
            intermediates: dict with keys q, k, v, attn_weights for error analysis
        """
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, heads, seq_len, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(~mask, float("-inf"))
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights_dropped = self.attn_drop(attn_weights)

        out = torch.matmul(attn_weights_dropped, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)

        intermediates = {
            "q": q.detach(),
            "k": k.detach(),
            "v": v.detach(),
            "attn_weights": attn_weights.detach(),
        }
        return out, intermediates
