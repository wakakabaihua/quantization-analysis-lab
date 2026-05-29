"""
Transformer-style MLP (feed-forward) block for quantization analysis.

Provides a compact but realistic two-layer FFN that mirrors the structure
used in transformer models (e.g., GPT-style feed-forward network).
Keeping the workload small makes quantization error sources clearly visible.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


_ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class MLPBlock(nn.Module):
    """
    Two-layer feed-forward block as used in transformer architectures.

    Structure: Linear -> activation -> Linear -> (optional dropout)

    Default expansion ratio: hidden_dim = 4 * input_dim, matching the
    standard transformer FFN design.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else 4 * input_dim

        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {activation!r}. "
                f"Choose from: {list(_ACTIVATIONS)}"
            )

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.act = _ACTIVATIONS[activation]()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, input_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., input_dim)
        Returns:
            (..., input_dim)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
