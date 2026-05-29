"""
Calibrators compute quantization scale and zero-point from observer statistics.

Supported strategies:
  - MinMaxCalibrator:       Uses raw observed min/max. Fast; can be noisy.
  - PercentileCalibrator:   Clips at a user-specified percentile before computing
                            scale/zp. Reduces the impact of statistical outliers.
  - KLCalibrator:           Searches over clip thresholds to minimize KL divergence
                            between the original and quantized distributions.

MinMaxCalibrator accepts both MinMax and histogram stats.
PercentileCalibrator and KLCalibrator require histogram stats (from HistogramObserver).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Scale / zero-point helpers
# ---------------------------------------------------------------------------

def _symmetric_scale_zp(
    abs_max: torch.Tensor, num_bits: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric signed integer quantization: zero_point = 0."""
    quant_max = 2 ** (num_bits - 1) - 1
    scale = (abs_max / quant_max).clamp(min=1e-8)
    zero_point = torch.zeros_like(scale, dtype=torch.int32)
    return scale, zero_point


def _asymmetric_scale_zp(
    x_min: torch.Tensor, x_max: torch.Tensor, num_bits: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Asymmetric unsigned integer quantization with non-zero zero_point."""
    quant_max = 2**num_bits - 1
    # Ensure the range always covers zero
    x_min = torch.minimum(x_min, torch.zeros_like(x_min))
    x_max = torch.maximum(x_max, torch.zeros_like(x_max))
    scale = ((x_max - x_min) / quant_max).clamp(min=1e-8)
    zero_point = (-x_min / scale).round().to(torch.int32).clamp(0, quant_max)
    return scale, zero_point


# ---------------------------------------------------------------------------
# Calibrator classes
# ---------------------------------------------------------------------------

class BaseCalibrator(ABC):
    """Abstract base class for calibrators."""

    def __init__(self, num_bits: int = 8, symmetric: bool = True) -> None:
        self.num_bits = num_bits
        self.symmetric = symmetric

    @abstractmethod
    def compute(self, stats: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (scale, zero_point) from observer stats."""


class MinMaxCalibrator(BaseCalibrator):
    """Computes scale/zp directly from observed min and max values."""

    def compute(self, stats: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        x_min = stats["min"]
        x_max = stats["max"]
        if not isinstance(x_min, torch.Tensor):
            x_min = torch.tensor(float(x_min), dtype=torch.float32)
            x_max = torch.tensor(float(x_max), dtype=torch.float32)
        else:
            x_min = x_min.float()
            x_max = x_max.float()

        if self.symmetric:
            abs_max = torch.maximum(x_min.abs(), x_max.abs())
            return _symmetric_scale_zp(abs_max, self.num_bits)
        return _asymmetric_scale_zp(x_min, x_max, self.num_bits)


class PercentileCalibrator(BaseCalibrator):
    """
    Clips the observed distribution at a specified percentile before calibrating.

    For example, percentile=99.99 clips the top and bottom 0.005% of values,
    preventing a handful of outliers from inflating the quantization range.

    Requires histogram stats (HistogramObserver).
    """

    def __init__(
        self,
        num_bits: int = 8,
        symmetric: bool = True,
        percentile: float = 99.99,
    ) -> None:
        super().__init__(num_bits, symmetric)
        self.percentile = percentile

    def compute(self, stats: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        if "hist" not in stats:
            raise ValueError(
                "PercentileCalibrator requires histogram stats. "
                "Use HistogramObserver instead of MinMaxObserver."
            )
        hist = stats["hist"]
        bin_edges = stats["bin_edges"]

        total = hist.sum()
        if total == 0:
            x_min = torch.tensor(float(bin_edges[0]), dtype=torch.float32)
            x_max = torch.tensor(float(bin_edges[-1]), dtype=torch.float32)
        else:
            cdf = np.cumsum(hist) / total
            tail = (1.0 - self.percentile / 100.0) / 2.0
            lower_idx = int(np.clip(np.searchsorted(cdf, tail), 0, len(bin_edges) - 2))
            upper_idx = int(np.clip(np.searchsorted(cdf, 1.0 - tail), 0, len(bin_edges) - 2))
            x_min = torch.tensor(float(bin_edges[lower_idx]), dtype=torch.float32)
            x_max = torch.tensor(float(bin_edges[upper_idx + 1]), dtype=torch.float32)

        if self.symmetric:
            abs_max = torch.maximum(x_min.abs(), x_max.abs())
            return _symmetric_scale_zp(abs_max, self.num_bits)
        return _asymmetric_scale_zp(x_min, x_max, self.num_bits)


class KLCalibrator(BaseCalibrator):
    """
    Minimizes KL divergence between the original and quantized distributions.

    Searches over candidate clip thresholds by coarsening the histogram to
    num_quant_bins and expanding back, measuring how much information is lost
    at each threshold. The threshold with the lowest KL divergence is chosen.

    This is an approximation of TensorRT's entropy calibration. The key
    insight is that clipping too conservatively wastes representable range,
    while clipping too aggressively truncates important signal.

    Requires histogram stats (HistogramObserver).
    """

    def compute(self, stats: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        if "hist" not in stats:
            raise ValueError(
                "KLCalibrator requires histogram stats. "
                "Use HistogramObserver instead of MinMaxObserver."
            )
        hist = stats["hist"]
        bin_edges = stats["bin_edges"]
        num_bins = len(hist)
        num_quant_bins = 2 ** (self.num_bits - 1) if self.symmetric else 2 ** self.num_bits

        total = hist.sum()
        if total == 0:
            x_min = torch.tensor(float(bin_edges[0]), dtype=torch.float32)
            x_max = torch.tensor(float(bin_edges[-1]), dtype=torch.float32)
        else:
            hist_norm = hist / total
            best_kl = float("inf")
            best_i = num_bins

            step = max(1, num_bins // 200)
            for i in range(num_quant_bins, num_bins + 1, step):
                # Clip histogram at bin i, accumulating tail mass
                p = hist_norm[:i].copy()
                p[-1] += hist_norm[i:].sum()
                if p.sum() == 0:
                    continue
                p /= p.sum()

                # Coarsen p to num_quant_bins bins
                q_coarse = np.zeros(num_quant_bins, dtype=np.float64)
                for j in range(num_quant_bins):
                    start = int(j * i / num_quant_bins)
                    end = int((j + 1) * i / num_quant_bins)
                    q_coarse[j] = p[start:end].sum()

                # Expand coarse histogram back to i bins
                q_fine = np.zeros(i, dtype=np.float64)
                for j in range(num_quant_bins):
                    start = int(j * i / num_quant_bins)
                    end = int((j + 1) * i / num_quant_bins)
                    n_bins = end - start
                    if n_bins > 0:
                        q_fine[start:end] = q_coarse[j] / n_bins

                q_sum = q_fine.sum()
                if q_sum == 0:
                    continue
                q_fine /= q_sum

                # KL(p || q) over bins where p > 0
                mask = p > 0
                kl = float(np.sum(p[mask] * np.log(p[mask] / (q_fine[mask] + 1e-10))))
                if kl < best_kl:
                    best_kl = kl
                    best_i = i

            clip_idx = min(best_i, len(bin_edges) - 1)
            x_min = torch.tensor(float(bin_edges[0]), dtype=torch.float32)
            x_max = torch.tensor(float(bin_edges[clip_idx]), dtype=torch.float32)

        if self.symmetric:
            abs_max = torch.maximum(x_min.abs(), x_max.abs())
            return _symmetric_scale_zp(abs_max, self.num_bits)
        return _asymmetric_scale_zp(x_min, x_max, self.num_bits)


def get_calibrator(method: str, **kwargs) -> BaseCalibrator:
    """Factory: return a calibrator by name."""
    method = method.lower()
    dispatch = {
        "minmax": MinMaxCalibrator,
        "percentile": PercentileCalibrator,
        "kl": KLCalibrator,
        "histogram": KLCalibrator,
    }
    if method not in dispatch:
        raise ValueError(
            f"Unknown calibration method {method!r}. "
            f"Choose from: {list(dispatch)}"
        )
    return dispatch[method](**kwargs)
