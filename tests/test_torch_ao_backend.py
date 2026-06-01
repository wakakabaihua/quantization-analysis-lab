"""
Tests for TorchAOBackend.

Verifies that:
  - TorchAOBackend integrates through PTQPipeline without error.
  - Output shapes are preserved.
  - Cosine similarity is acceptable (dynamic INT8 ≈ float).
  - Per-channel weight quantization produces valid models.
  - Calibration data is not required (calibrate() is a no-op).
  - Rejects unsupported bit-widths at construction.
  - from_config() picks up backend: torch_ao.
  - Resulting model contains DynamicQuantizedLinear modules (real INT8).
  - FakeQuantBackend and TorchAOBackend produce numerically similar
    results (both are INT8 approximations of the same float model).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.mlp_block import MLPBlock
from src.quant.backends import TorchAOBackend, get_backend
from src.quant.backends.fake_quant_backend import QuantizedLinear
from src.quant.error_analysis import compute_output_error
from src.quant.ptq_pipeline import PTQPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mlp(input_dim: int = 64, hidden_dim: int = 128) -> MLPBlock:
    return MLPBlock(input_dim=input_dim, hidden_dim=hidden_dim).eval()


def _run_torch_ao(pipeline: PTQPipeline, model: nn.Module) -> nn.Module:
    """Calibrate (no-op for torch_ao) and convert."""
    # No calibration data needed; calibrate() is a no-op.
    pipeline.calibrate(model, lambda: None)
    return pipeline.quantize(model).eval()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestTorchAOBackendConstruction:
    def test_default_construction(self):
        backend = TorchAOBackend()
        assert backend.num_bits == 8
        assert backend.name == "torch_ao"

    def test_rejects_non_int8(self):
        with pytest.raises(ValueError, match="only supports 8-bit"):
            TorchAOBackend(num_bits=4)

    def test_get_backend_aliases(self):
        assert isinstance(get_backend("torch_ao"), TorchAOBackend)
        assert isinstance(get_backend("torch"), TorchAOBackend)

    def test_from_config_torch_ao(self):
        config = {
            "quantization": {"enabled": True, "dtype": "int8", "symmetric": True, "weight_only": False},
            "calibration": {"method": "minmax", "num_samples": 128, "percentile": None},
            "granularity": "per_tensor",
            "backend": "torch_ao",
        }
        pipeline = PTQPipeline.from_config(config)
        assert isinstance(pipeline._backend, TorchAOBackend)


# ---------------------------------------------------------------------------
# Calibrate (no-op)
# ---------------------------------------------------------------------------

class TestTorchAOCalibration:
    def test_calibrate_accepts_none_fn(self):
        """calibrate() must not raise even with a trivial no-op fn."""
        model = _make_mlp()
        backend = TorchAOBackend()
        backend.calibrate(model, lambda: None)  # must not raise

    def test_calibrate_does_not_mutate_model(self):
        """calibrate() must leave the model untouched."""
        model = _make_mlp()
        original_params = {n: p.clone() for n, p in model.named_parameters()}
        backend = TorchAOBackend()
        backend.calibrate(model, lambda: None)
        for name, param in model.named_parameters():
            assert torch.equal(param, original_params[name]), (
                f"Parameter {name} was mutated by calibrate()"
            )


# ---------------------------------------------------------------------------
# Convert — per-tensor
# ---------------------------------------------------------------------------

class TestTorchAOPerTensor:
    def test_output_shape_preserved(self):
        model = _make_mlp()
        pipeline = PTQPipeline(num_bits=8, symmetric=True, per_channel=False, backend="torch_ao")
        model_q = _run_torch_ao(pipeline, model)
        x = torch.randn(2, 4, 64)
        with torch.no_grad():
            out = model_q(x)
        assert out.shape == x.shape

    def test_cosine_similarity_acceptable(self):
        """Dynamic INT8 should stay close to float."""
        model = _make_mlp()
        pipeline = PTQPipeline(num_bits=8, symmetric=True, per_channel=False, backend="torch_ao")
        model_q = _run_torch_ao(pipeline, model)
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            ref = model(x)
            q = model_q(x)
        metrics = compute_output_error(ref, q)
        assert metrics["cosine_similarity"] > 0.95

    def test_model_contains_dynamic_quantized_linear(self):
        """Converted model must have real torch.ao quantized Linear layers."""
        model = _make_mlp()
        pipeline = PTQPipeline(num_bits=8, per_channel=False, backend="torch_ao")
        model_q = _run_torch_ao(pipeline, model)
        # torch.ao dynamic quantization replaces nn.Linear with
        # torch.ao.nn.quantized.dynamic.Linear (or torch.nn.quantized.dynamic.Linear
        # depending on PyTorch version).
        dynamic_linears = [
            m for m in model_q.modules()
            if "quantized" in type(m).__module__ or "dynamic" in type(m).__name__.lower()
        ]
        assert len(dynamic_linears) > 0, (
            "Expected quantized linear modules in converted model, "
            f"but only found: {[type(m).__name__ for m in model_q.modules()]}"
        )

    def test_original_model_not_mutated(self):
        """convert() must not touch the original model."""
        model = _make_mlp()
        original_params = {n: p.clone() for n, p in model.named_parameters()}
        pipeline = PTQPipeline(num_bits=8, per_channel=False, backend="torch_ao")
        _run_torch_ao(pipeline, model)
        for name, param in model.named_parameters():
            assert torch.equal(param, original_params[name])


# ---------------------------------------------------------------------------
# Convert — per-channel
# ---------------------------------------------------------------------------

class TestTorchAOPerChannel:
    def test_output_shape_preserved(self):
        model = _make_mlp()
        pipeline = PTQPipeline(num_bits=8, symmetric=True, per_channel=True, backend="torch_ao")
        model_q = _run_torch_ao(pipeline, model)
        x = torch.randn(1, 4, 64)
        with torch.no_grad():
            out = model_q(x)
        assert out.shape == x.shape

    def test_cosine_similarity_acceptable(self):
        model = _make_mlp()
        pipeline = PTQPipeline(num_bits=8, symmetric=True, per_channel=True, backend="torch_ao")
        model_q = _run_torch_ao(pipeline, model)
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            metrics = compute_output_error(model(x), model_q(x))
        assert metrics["cosine_similarity"] > 0.95

    def test_per_channel_not_worse_than_per_tensor(self):
        """Per-channel weight quantization should achieve >= cosine sim."""
        torch.manual_seed(42)
        model = _make_mlp()
        x = torch.randn(2, 8, 64)

        def sim(per_channel: bool) -> float:
            p = PTQPipeline(num_bits=8, symmetric=True, per_channel=per_channel, backend="torch_ao")
            p.calibrate(model, lambda: None)
            mq = p.quantize(model).eval()
            with torch.no_grad():
                return compute_output_error(model(x), mq(x))["cosine_similarity"]

        assert sim(per_channel=True) >= sim(per_channel=False) - 0.01


# ---------------------------------------------------------------------------
# PTQPipeline disabled
# ---------------------------------------------------------------------------

class TestTorchAODisabled:
    def test_disabled_returns_plain_copy(self):
        model = _make_mlp()
        pipeline = PTQPipeline(enabled=False, backend="torch_ao")
        model_q = pipeline.quantize(model)
        # Disabled pipeline → deep copy, no quantized layers.
        dynamic_linears = [
            m for m in model_q.modules()
            if "quantized" in type(m).__module__
        ]
        assert len(dynamic_linears) == 0
        x = torch.randn(1, 4, 64)
        with torch.no_grad():
            assert torch.equal(model(x), model_q(x))


# ---------------------------------------------------------------------------
# Backend comparison: FakeQuant vs TorchAO
# ---------------------------------------------------------------------------

class TestFakeVsTorchAO:
    def test_both_close_to_float(self):
        """FakeQuantBackend and TorchAOBackend should both be close to fp model."""
        torch.manual_seed(0)
        model = _make_mlp()
        x = torch.randn(4, 8, 64)

        # FakeQuant
        p_fake = PTQPipeline(num_bits=8, symmetric=True, per_channel=True, backend="fake")
        calib_inputs = [torch.randn(2, 4, 64) for _ in range(8)]
        p_fake.calibrate(model, lambda: [model(xi) for xi in calib_inputs])
        mq_fake = p_fake.quantize(model).eval()

        # TorchAO
        p_torch = PTQPipeline(num_bits=8, symmetric=True, per_channel=True, backend="torch_ao")
        p_torch.calibrate(model, lambda: None)
        mq_torch = p_torch.quantize(model).eval()

        with torch.no_grad():
            ref = model(x)
            sim_fake = compute_output_error(ref, mq_fake(x))["cosine_similarity"]
            sim_torch = compute_output_error(ref, mq_torch(x))["cosine_similarity"]

        assert sim_fake > 0.95, f"FakeQuant cosine sim too low: {sim_fake:.4f}"
        assert sim_torch > 0.95, f"TorchAO cosine sim too low: {sim_torch:.4f}"
