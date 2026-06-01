"""
Quantization backend registry.

Usage::

    from src.quant.backends import get_backend, QuantBackend, FakeQuantBackend

    backend = get_backend("fake", num_bits=8, symmetric=True)
"""

from __future__ import annotations

from .base import QuantBackend
from .fake_quant_backend import FakeQuantBackend, QuantizedLinear
from .torch_ao_backend import TorchAOBackend
from .bnb_backend import BitsAndBytesBackend
from .gptq_backend import GPTQBackend, GPTQLinear
from .awq_backend import AWQBackend, AWQLinear
from .mixed_precision_backend import MixedPrecisionBackend

# Registry mapping name → backend class.
# New backends (gptq, awq, …) are added here.
_REGISTRY: dict[str, type[QuantBackend]] = {
    "fake": FakeQuantBackend,
    "fake_quant": FakeQuantBackend,
    "torch_ao": TorchAOBackend,
    "torch": TorchAOBackend,
    "bitsandbytes": BitsAndBytesBackend,
    "bnb": BitsAndBytesBackend,
    "gptq": GPTQBackend,
    "awq": AWQBackend,
    "mixed_precision": MixedPrecisionBackend,
    "mp": MixedPrecisionBackend,
}


def get_backend(name: str, **kwargs) -> QuantBackend:
    """
    Instantiate a quantization backend by name.

    Args:
        name:   Backend identifier (e.g. ``"fake"``).
        **kwargs: Constructor arguments forwarded to the backend class.

    Returns:
        A configured QuantBackend instance.

    Raises:
        ValueError: If the backend name is not registered.
    """
    key = name.lower()
    if key not in _REGISTRY:
        available = list(_REGISTRY)
        raise ValueError(
            f"Unknown quantization backend {name!r}. "
            f"Available backends: {available}"
        )
    return _REGISTRY[key](**kwargs)


def register_backend(name: str, cls: type[QuantBackend]) -> None:
    """
    Register a new backend class under the given name.

    This allows third-party or future backends to be plugged in without
    modifying this file::

        from src.quant.backends import register_backend
        from my_pkg import MyBackend

        register_backend("my_backend", MyBackend)

    Args:
        name: Identifier string (case-insensitive).
        cls:  A QuantBackend subclass.
    """
    _REGISTRY[name.lower()] = cls


__all__ = [
    "QuantBackend",
    "FakeQuantBackend",
    "TorchAOBackend",
    "BitsAndBytesBackend",
    "GPTQBackend",
    "AWQBackend",
    "AWQLinear",
    "MixedPrecisionBackend",
    "QuantizedLinear",
    "get_backend",
    "register_backend",
]
