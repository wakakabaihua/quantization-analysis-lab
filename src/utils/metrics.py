"""
Evaluation metrics for quantization analysis.

Provides signal-level metrics (SNR, PSNR), task-level accuracy helpers,
and a function to aggregate per-layer error dicts into summary statistics.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


def compute_snr(signal: torch.Tensor, noise: torch.Tensor) -> float:
    """
    Signal-to-noise ratio in dB.

    Args:
        signal: Reference (baseline) tensor.
        noise:  Error tensor (baseline − quantized).
    """
    signal_power = float((signal.float() ** 2).mean())
    noise_power = float((noise.float() ** 2).mean())
    if noise_power < 1e-12:
        return float("inf")
    return 10.0 * float(np.log10(signal_power / noise_power))


def compute_psnr(
    reference: torch.Tensor,
    distorted: torch.Tensor,
    max_val: Optional[float] = None,
) -> float:
    """
    Peak signal-to-noise ratio in dB.

    Args:
        reference: Baseline tensor.
        distorted: Quantized tensor.
        max_val:   Peak value; defaults to the max absolute value in reference.
    """
    if max_val is None:
        max_val = float(reference.float().abs().max())
    mse = float(((reference.float() - distorted.float()) ** 2).mean())
    if mse < 1e-12:
        return float("inf")
    return 20.0 * float(np.log10(max_val / np.sqrt(mse)))


def accuracy(predictions: Sequence, targets: Sequence) -> float:
    """Top-1 classification accuracy."""
    if not targets:
        return 0.0
    correct = sum(int(p == t) for p, t in zip(predictions, targets))
    return correct / len(targets)


def aggregate_errors(per_layer_errors: Dict[str, dict]) -> dict:
    """
    Aggregate per-layer error dicts into summary statistics.

    For each metric found across layers, computes mean, max, and min
    values, prefixed as 'mean_*', 'max_*', 'min_*'.

    Args:
        per_layer_errors: Dict mapping layer name → error metric dict.

    Returns:
        Flat dict of summary statistics.
    """
    if not per_layer_errors:
        return {}

    all_metrics: Dict[str, List[float]] = {}
    for layer_metrics in per_layer_errors.values():
        for key, val in layer_metrics.items():
            all_metrics.setdefault(key, []).append(float(val))

    summary: dict = {}
    for metric, values in all_metrics.items():
        summary[f"mean_{metric}"] = float(np.mean(values))
        summary[f"max_{metric}"] = float(np.max(values))
        summary[f"min_{metric}"] = float(np.min(values))
    return summary
