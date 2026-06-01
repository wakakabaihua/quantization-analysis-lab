"""
Fake-quantization backend.

Implements the QuantBackend interface using pure float32 fake quantization:
all arithmetic stays in float32 while quantization error is simulated via
the round → clamp → dequantize cycle. No integer compute kernels are used.

This is the original PTQ logic extracted from ptq_pipeline.py and adapted
to the QuantBackend interface so it can be swapped for other backends.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..calibrators import BaseCalibrator, get_calibrator
from ..fake_quant import FakeQuantize
from ..observers import BaseObserver, HistogramObserver, MinMaxObserver
from .base import QuantBackend


# ---------------------------------------------------------------------------
# QuantizedLinear
# ---------------------------------------------------------------------------

class QuantizedLinear(nn.Module):
    """
    A Linear layer with fake quantization applied to weights and optionally
    to input activations.

    Exposed as a public class so that LayerwiseErrorTracker can be pointed
    at QuantizedLinear instances in a quantized model.
    """

    def __init__(
        self,
        original: nn.Linear,
        weight_fq: FakeQuantize,
        act_fq: Optional[FakeQuantize] = None,
    ) -> None:
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.weight = original.weight
        self.bias = original.bias
        self.weight_fq = weight_fq
        self.act_fq = act_fq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_fq is not None:
            x = self.act_fq(x)
        w = self.weight_fq(self.weight)
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"act_fq={self.act_fq is not None}"
        )


# ---------------------------------------------------------------------------
# FakeQuantBackend
# ---------------------------------------------------------------------------

class FakeQuantBackend(QuantBackend):
    """
    Post-training quantization backend based on float32 fake quantization.

    Supports:
        - INT4 / INT8 (and arbitrary bit-widths)
        - Symmetric and asymmetric quantization
        - Per-tensor and per-channel weight quantization
        - Weight-only and weight+activation quantization
        - MinMax, Percentile, and KL calibration
    """

    def __init__(
        self,
        num_bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
        weight_only: bool = False,
        calibration_method: str = "minmax",
        calibration_percentile: float = 99.99,
    ) -> None:
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.weight_only = weight_only
        self.calibration_method = calibration_method
        self.calibration_percentile = calibration_percentile

        self._activation_stats: Dict[str, dict] = {}
        self._weight_stats: Dict[str, dict] = {}
        self._hooks: List[Any] = []

    @property
    def name(self) -> str:
        return "fake_quant"

    # ------------------------------------------------------------------
    # Observer / calibrator factories
    # ------------------------------------------------------------------

    def _make_weight_observer(self) -> BaseObserver:
        if self.calibration_method in ("percentile", "kl", "histogram"):
            return HistogramObserver(per_channel=self.per_channel)
        return MinMaxObserver(per_channel=self.per_channel)

    def _make_act_observer(self) -> BaseObserver:
        if self.calibration_method in ("percentile", "kl", "histogram"):
            return HistogramObserver(per_channel=False)
        return MinMaxObserver(per_channel=False)

    def _make_calibrator(self) -> BaseCalibrator:
        kwargs: dict = dict(num_bits=self.num_bits, symmetric=self.symmetric)
        if self.calibration_method == "percentile":
            kwargs["percentile"] = self.calibration_percentile
        return get_calibrator(self.calibration_method, **kwargs)

    # ------------------------------------------------------------------
    # QuantBackend interface
    # ------------------------------------------------------------------

    def calibrate(
        self,
        model: nn.Module,
        calibration_fn: Callable[[], None],
    ) -> None:
        """
        Attach observers to all Linear layers, run calibration_fn, then
        detach hooks and store the collected statistics.
        """
        activation_observers: Dict[str, BaseObserver] = {}

        def make_hook(obs: BaseObserver) -> Callable:
            def hook(module: nn.Module, inp: Tuple, out: torch.Tensor) -> None:
                obs.update(inp[0].detach())
            return hook

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                obs = self._make_act_observer()
                activation_observers[name] = obs
                self._hooks.append(module.register_forward_hook(make_hook(obs)))

        model.eval()
        with torch.no_grad():
            calibration_fn()

        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        for name, obs in activation_observers.items():
            try:
                self._activation_stats[name] = obs.stats
            except RuntimeError:
                pass

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                w_obs = self._make_weight_observer()
                w_obs.update(module.weight.detach())
                self._weight_stats[name] = w_obs.stats

    def convert(self, model: nn.Module) -> nn.Module:
        """
        Return a deep copy of the model with Linear layers replaced by
        QuantizedLinear (fake quantization applied to weights and activations).

        The original model is not mutated.
        """
        model_q = copy.deepcopy(model)
        calibrator = self._make_calibrator()

        for name, module in list(model_q.named_modules()):
            if not isinstance(module, nn.Linear):
                continue

            # --- Weight quantization ---
            w_stats = self._weight_stats.get(name)
            if w_stats is None:
                w_obs = self._make_weight_observer()
                w_obs.update(module.weight.detach())
                w_stats = w_obs.stats

            w_scale, w_zp = calibrator.compute(w_stats)
            weight_fq = FakeQuantize(
                num_bits=self.num_bits,
                symmetric=self.symmetric,
                per_channel=self.per_channel,
                channel_axis=0,
            )
            weight_fq.set_qparams(w_scale, w_zp)

            # --- Activation quantization (skipped for weight-only mode) ---
            act_fq: Optional[FakeQuantize] = None
            if not self.weight_only:
                a_stats = self._activation_stats.get(name)
                if a_stats is not None:
                    a_scale, a_zp = calibrator.compute(a_stats)
                    act_fq = FakeQuantize(
                        num_bits=self.num_bits,
                        symmetric=self.symmetric,
                        per_channel=False,
                        channel_axis=0,
                    )
                    act_fq.set_qparams(a_scale, a_zp)

            q_linear = QuantizedLinear(module, weight_fq, act_fq)
            _set_module(model_q, name, q_linear)

        return model_q


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _set_module(model: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    """Replace a submodule identified by a dotted attribute path."""
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)
