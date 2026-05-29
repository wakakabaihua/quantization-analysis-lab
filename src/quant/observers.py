"""
Observers for collecting activation and weight statistics during calibration.

An observer is attached to a module via a forward hook. It accumulates
running statistics that calibrators later use to compute scale and zero-point.

Two observer types are provided:
  - MinMaxObserver:   O(1) memory per update; fast but sensitive to outliers.
  - HistogramObserver: Collects raw samples (up to max_samples) and computes
                       a histogram on demand; required for percentile and KL
                       calibration strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch


class BaseObserver(ABC):
    """Abstract base class for activation/weight observers."""

    def __init__(self, per_channel: bool = False, channel_axis: int = 0) -> None:
        self.per_channel = per_channel
        self.channel_axis = channel_axis

    @abstractmethod
    def update(self, x: torch.Tensor) -> None:
        """Update internal statistics with a new batch of values."""

    @property
    @abstractmethod
    def stats(self) -> dict:
        """Return collected statistics as a dictionary."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated statistics."""


class MinMaxObserver(BaseObserver):
    """
    Tracks running min and max values.

    Memory: O(1) per call — only stores current min/max tensors.
    Limitation: A single outlier value can expand the observed range
                significantly, wasting representable range.
    """

    def __init__(self, per_channel: bool = False, channel_axis: int = 0) -> None:
        super().__init__(per_channel, channel_axis)
        self._min: Optional[torch.Tensor] = None
        self._max: Optional[torch.Tensor] = None

    def update(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            if self.per_channel:
                reduce_dims = [i for i in range(x.ndim) if i != self.channel_axis]
                x_min = x.amin(dim=reduce_dims)
                x_max = x.amax(dim=reduce_dims)
            else:
                x_min = x.amin()
                x_max = x.amax()

            if self._min is None:
                self._min = x_min.clone()
                self._max = x_max.clone()
            else:
                self._min = torch.minimum(self._min, x_min)
                self._max = torch.maximum(self._max, x_max)

    @property
    def stats(self) -> dict:
        if self._min is None:
            raise RuntimeError(
                "Observer has not seen any data. Call update() before accessing stats."
            )
        return {"min": self._min, "max": self._max}

    def reset(self) -> None:
        self._min = None
        self._max = None


class HistogramObserver(BaseObserver):
    """
    Collects a histogram of activation values for percentile or KL calibration.

    Accumulates raw samples (up to max_samples) and computes the full
    histogram lazily when stats is accessed. This avoids the dynamic-range
    problem of incremental histogramming while keeping memory bounded.

    Note: per_channel is accepted for API compatibility but histogram
    collection is always global (per-tensor). Per-channel calibration
    with histogram stats requires separate observer instances per channel.
    """

    def __init__(
        self,
        num_bins: int = 2048,
        max_samples: int = 500_000,
        per_channel: bool = False,
        channel_axis: int = 0,
    ) -> None:
        super().__init__(per_channel, channel_axis)
        self.num_bins = num_bins
        self.max_samples = max_samples
        self._chunks: list[np.ndarray] = []
        self._sample_count: int = 0

    def update(self, x: torch.Tensor) -> None:
        remaining = self.max_samples - self._sample_count
        if remaining <= 0:
            return
        flat = x.detach().float().cpu().numpy().ravel()
        take = min(len(flat), remaining)
        self._chunks.append(flat[:take])
        self._sample_count += take

    @property
    def stats(self) -> dict:
        if not self._chunks:
            raise RuntimeError(
                "Observer has not seen any data. Call update() before accessing stats."
            )
        all_data = np.concatenate(self._chunks)
        hist, bin_edges = np.histogram(all_data, bins=self.num_bins)
        return {
            "hist": hist.astype(np.float64),
            "bin_edges": bin_edges,
            "min": float(all_data.min()),
            "max": float(all_data.max()),
        }

    def reset(self) -> None:
        self._chunks = []
        self._sample_count = 0
