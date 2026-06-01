"""
Abstract base class for quantization backends.

A backend encapsulates the full quantization workflow:

  1. calibrate() — observe statistics from representative data.
  2. convert()   — return a new quantized model.

Subclasses implement the concrete strategy (fake quant, torch.ao,
bitsandbytes, GPTQ, AWQ, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch.nn as nn


class QuantBackend(ABC):
    """
    Abstract interface for a post-training quantization backend.

    All backends share the same two-phase protocol so that PTQPipeline
    can orchestrate them uniformly regardless of the underlying engine.
    """

    @abstractmethod
    def calibrate(self, model: nn.Module, calibration_fn: Callable[[], None]) -> None:
        """
        Collect quantization statistics by running calibration_fn.

        Args:
            model:          Float model to observe (must not be mutated).
            calibration_fn: Zero-argument callable that runs forward passes
                            with representative calibration data.
        """
        ...

    @abstractmethod
    def convert(self, model: nn.Module) -> nn.Module:
        """
        Return a new quantized model based on previously collected statistics.

        The original model must not be mutated.

        Args:
            model: Float model that was previously passed to calibrate().

        Returns:
            A new model with quantized layers.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable backend identifier."""
        return self.__class__.__name__
