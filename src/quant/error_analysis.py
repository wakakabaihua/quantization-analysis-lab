"""
Error analysis utilities for quantization experiments.

Provides functions and a context-manager tracker for comparing quantized
model outputs to a floating-point baseline, at both the global and per-layer
level.

Key metrics:
  - cosine_similarity:      Direction alignment; 1.0 = perfect.
  - max_absolute_error:     Worst-case deviation; sensitive to outliers.
  - mean_absolute_error:    Average per-element deviation.
  - root_mean_squared_error: Penalises large errors more than MAE.
  - relative_error:         Error normalized by baseline magnitude.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Pointwise metrics
# ---------------------------------------------------------------------------

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two tensors (flattened). Range [−1, 1]."""
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    norm = torch.norm(a_f) * torch.norm(b_f)
    if norm < 1e-10:
        return 1.0
    return float(torch.dot(a_f, b_f) / norm)


def max_absolute_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


def mean_absolute_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().mean())


def root_mean_squared_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a.float() - b.float()) ** 2).mean().sqrt())


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean |a − b| / (|b| + eps)."""
    return float(((a.float() - b.float()).abs() / (b.float().abs() + 1e-8)).mean())


def compute_output_error(
    baseline: torch.Tensor, quantized: torch.Tensor
) -> dict:
    """Return a dict of all error metrics for a single pair of outputs."""
    return {
        "cosine_similarity": cosine_similarity(baseline, quantized),
        "max_absolute_error": max_absolute_error(baseline, quantized),
        "mean_absolute_error": mean_absolute_error(baseline, quantized),
        "root_mean_squared_error": root_mean_squared_error(baseline, quantized),
        "relative_error": relative_error(baseline, quantized),
    }


# ---------------------------------------------------------------------------
# Layer-wise tracker
# ---------------------------------------------------------------------------

class LayerwiseErrorTracker:
    """
    Captures intermediate layer outputs by attaching forward hooks.

    Use as a context manager so hooks are cleaned up automatically::

        tracker_fp = LayerwiseErrorTracker(model_fp)
        tracker_q  = LayerwiseErrorTracker(model_q, target_types=(QuantizedLinear,))

        with tracker_fp, tracker_q:
            with torch.no_grad():
                model_fp(x)
                model_q(x)

        per_layer = compute_layerwise_errors(tracker_fp, tracker_q)

    The dotted module names in ``outputs`` match those produced by
    ``model.named_modules()``, so baseline and quantized trackers with
    structurally equivalent models share the same key namespace.
    """

    def __init__(
        self,
        model: nn.Module,
        target_types: Tuple[Type[nn.Module], ...] = (nn.Linear,),
    ) -> None:
        self.model = model
        self.target_types = target_types
        self.outputs: Dict[str, torch.Tensor] = {}
        self._hooks: List[Any] = []

    def _attach(self) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, self.target_types):
                def _make_hook(n: str):
                    def hook(mod: nn.Module, inp: Tuple, out: torch.Tensor) -> None:
                        self.outputs[n] = out.detach()
                    return hook
                self._hooks.append(module.register_forward_hook(_make_hook(name)))

    def _detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self) -> "LayerwiseErrorTracker":
        self.outputs.clear()
        self._attach()
        return self

    def __exit__(self, *args: Any) -> None:
        self._detach()


def compute_layerwise_errors(
    baseline_tracker: LayerwiseErrorTracker,
    quantized_tracker: LayerwiseErrorTracker,
) -> Dict[str, dict]:
    """
    Compare per-layer outputs captured by two trackers.

    Only layers present in both trackers are compared. Layers that exist
    in one but not the other are silently skipped (this can happen when
    target_types differ between the two trackers).
    """
    errors: Dict[str, dict] = {}
    for name, b_out in baseline_tracker.outputs.items():
        if name not in quantized_tracker.outputs:
            continue
        q_out = quantized_tracker.outputs[name]
        errors[name] = compute_output_error(b_out, q_out)
    return errors
