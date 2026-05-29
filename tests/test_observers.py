"""Tests for observer correctness."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.quant.observers import HistogramObserver, MinMaxObserver


class TestMinMaxObserver:
    def test_basic_min_max(self):
        obs = MinMaxObserver()
        obs.update(torch.tensor([-2.0, 0.0, 3.0]))
        stats = obs.stats
        assert float(stats["min"]) == pytest.approx(-2.0)
        assert float(stats["max"]) == pytest.approx(3.0)

    def test_running_updates(self):
        obs = MinMaxObserver()
        obs.update(torch.tensor([0.0, 1.0]))
        obs.update(torch.tensor([-1.5, 0.5]))
        obs.update(torch.tensor([2.0, 0.0]))
        assert float(obs.stats["min"]) == pytest.approx(-1.5)
        assert float(obs.stats["max"]) == pytest.approx(2.0)

    def test_stats_raises_before_update(self):
        obs = MinMaxObserver()
        with pytest.raises(RuntimeError):
            _ = obs.stats

    def test_reset_clears_state(self):
        obs = MinMaxObserver()
        obs.update(torch.tensor([1.0, 2.0]))
        obs.reset()
        with pytest.raises(RuntimeError):
            _ = obs.stats

    def test_per_channel_shape(self):
        """Per-channel min/max should have one value per channel."""
        obs = MinMaxObserver(per_channel=True, channel_axis=0)
        x = torch.randn(4, 16)  # 4 output channels
        obs.update(x)
        assert obs.stats["min"].shape == (4,)
        assert obs.stats["max"].shape == (4,)

    def test_per_channel_values_correct(self):
        obs = MinMaxObserver(per_channel=True, channel_axis=0)
        x = torch.tensor([[1.0, 2.0], [-3.0, 0.0]])  # 2 channels
        obs.update(x)
        stats = obs.stats
        assert float(stats["min"][0]) == pytest.approx(1.0)
        assert float(stats["max"][0]) == pytest.approx(2.0)
        assert float(stats["min"][1]) == pytest.approx(-3.0)
        assert float(stats["max"][1]) == pytest.approx(0.0)

    def test_multidim_tensor(self):
        """Observer should work with arbitrary input shapes."""
        obs = MinMaxObserver()
        obs.update(torch.randn(2, 4, 8, 16))
        stats = obs.stats
        assert "min" in stats
        assert "max" in stats


class TestHistogramObserver:
    def test_histogram_populated(self):
        obs = HistogramObserver(num_bins=64)
        obs.update(torch.randn(1000))
        stats = obs.stats
        assert stats["hist"].sum() > 0
        assert len(stats["bin_edges"]) == 65  # num_bins + 1

    def test_range_coverage(self):
        obs = HistogramObserver(num_bins=32)
        obs.update(torch.tensor([-5.0, 0.0, 5.0]))
        stats = obs.stats
        assert stats["min"] == pytest.approx(-5.0)
        assert stats["max"] == pytest.approx(5.0)

    def test_multiple_updates_accumulate(self):
        obs = HistogramObserver(num_bins=64, max_samples=100_000)
        for _ in range(5):
            obs.update(torch.randn(200))
        stats = obs.stats
        # Should have accumulated 5*200 = 1000 samples
        assert stats["hist"].sum() == pytest.approx(1000.0, abs=1.0)

    def test_max_samples_cap(self):
        obs = HistogramObserver(num_bins=32, max_samples=100)
        obs.update(torch.randn(500))
        obs.update(torch.randn(500))  # should be capped at 100
        stats = obs.stats
        assert stats["hist"].sum() == pytest.approx(100.0, abs=1.0)

    def test_stats_raises_before_update(self):
        obs = HistogramObserver()
        with pytest.raises(RuntimeError):
            _ = obs.stats

    def test_reset_clears_state(self):
        obs = HistogramObserver()
        obs.update(torch.randn(100))
        obs.reset()
        with pytest.raises(RuntimeError):
            _ = obs.stats
