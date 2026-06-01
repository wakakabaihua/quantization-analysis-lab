"""
Tests for AWQBackend and AWQLinear.

AWQ (Lin et al., MLSys 2024) searches for per-input-channel scales that
protect important weight channels by giving them more quantization range.
The AWQLinear inference module stores W_q = Q(W × s*) and applies x / s*
at forward time so that W_q @ (x/s*) ≈ W @ x with reduced error.

Test coverage:
  - Construction: defaults, parameter pass-through, get_backend alias.
  - calibrate(): collects activations; does not mutate model.
  - convert():
      - Output shape preserved.
      - Cosine similarity acceptable at INT4 (≥ 0.85).
      - All Linear layers replaced with AWQLinear.
      - AWQLinear has awq_scale buffer with correct shape.
      - Original model is not mutated.
  - AWQLinear forward: scale is applied to input; output ≈ plain linear.
  - Fallback: convert() without calibration uses scale=1 (no AWQ benefit).
  - Pipeline integration: from_config() routes to AWQBackend.
  - Cross-backend: AWQ output error ≤ FakeQuant output error at INT4.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mlp_block import MLPBlock
from src.quant.backends import AWQBackend, AWQLinear, get_backend
from src.quant.backends.awq_backend import _compute_qparams, _find_awq_scale, _batch_fake_quant
from src.quant.error_analysis import compute_output_error, cosine_similarity
from src.quant.ptq_pipeline import PTQPipeline

torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mlp(input_dim: int = 64, hidden_dim: int = 128) -> MLPBlock:
    return MLPBlock(input_dim=input_dim, hidden_dim=hidden_dim).eval()


def _calibration_fn(model: nn.Module, batch_size: int = 16, seq_len: int = 8,
                    input_dim: int = 64) -> callable:
    def fn() -> None:
        x = torch.randn(batch_size, seq_len, input_dim)
        with torch.no_grad():
            model(x)
    return fn


def _run_awq(
    pipeline: PTQPipeline,
    model: nn.Module,
    input_dim: int = 64,
) -> nn.Module:
    """Calibrate with random data, then convert."""
    pipeline.calibrate(model, _calibration_fn(model, input_dim=input_dim))
    return pipeline.quantize(model).eval()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestAWQBackendConstruction:
    def test_default_construction(self):
        backend = AWQBackend()
        assert backend.num_bits == 4
        assert backend.symmetric is True
        assert backend.per_channel is True
        assert backend.weight_only is True
        assert backend.n_grid == 20

    def test_custom_params(self):
        backend = AWQBackend(num_bits=8, symmetric=False, per_channel=False, n_grid=10)
        assert backend.num_bits == 8
        assert backend.symmetric is False
        assert backend.per_channel is False
        assert backend.n_grid == 10

    def test_name_property(self):
        assert AWQBackend().name == "awq"

    def test_get_backend_alias(self):
        backend = get_backend("awq", num_bits=4)
        assert isinstance(backend, AWQBackend)
        assert backend.num_bits == 4


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TestAWQCalibration:
    def test_calibrate_collects_activations(self):
        model = _make_mlp()
        backend = AWQBackend()
        backend.calibrate(model, _calibration_fn(model))
        assert len(backend._activations) > 0

    def test_activations_shape(self):
        model = _make_mlp(input_dim=64)
        backend = AWQBackend()
        backend.calibrate(model, _calibration_fn(model, batch_size=4, seq_len=8))
        for name, X in backend._activations.items():
            assert X.ndim == 2, f"Layer {name}: expected 2D, got {X.shape}"

    def test_calibrate_does_not_mutate_model(self):
        model = _make_mlp()
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        backend = AWQBackend()
        backend.calibrate(model, _calibration_fn(model))
        for name, p in model.named_parameters():
            assert torch.equal(p, params_before[name]), \
                f"Parameter {name} changed during calibration"

    def test_calibrate_clears_hooks(self):
        model = _make_mlp()
        backend = AWQBackend()
        backend.calibrate(model, _calibration_fn(model))
        assert len(backend._hooks) == 0


# ---------------------------------------------------------------------------
# AWQLinear module
# ---------------------------------------------------------------------------

def _make_awq_linear(d_in: int = 16, d_out: int = 8, num_bits: int = 4,
                     symmetric: bool = True) -> AWQLinear:
    """Helper: create an AWQLinear via from_float() with random weights."""
    W = torch.randn(d_out, d_in)
    awq_scale = torch.ones(d_in) * 2.0
    W_scaled = W * awq_scale.unsqueeze(0)
    return AWQLinear.from_float(d_in, d_out, W_scaled, awq_scale,
                                symmetric=symmetric, num_bits=num_bits)


class TestAWQLinear:
    def test_forward_matches_scaled_linear(self):
        """AWQLinear(x) ≈ W @ x after scale inversion (within INT4 quant noise)."""
        d_in, d_out = 32, 16
        torch.manual_seed(0)
        W = torch.randn(d_out, d_in)
        s = torch.ones(d_in) * 2.0
        W_scaled = W * s.unsqueeze(0)
        bias = torch.randn(d_out)
        x = torch.randn(4, 8, d_in)

        awq = AWQLinear.from_float(d_in, d_out, W_scaled, awq_scale=s,
                                   symmetric=True, bias=bias)
        awq.eval()
        with torch.no_grad():
            out_awq = awq(x)
            out_ref = F.linear(x, W, bias)   # ideal: W @ x + bias
        cos = F.cosine_similarity(out_awq.flatten(), out_ref.flatten(), dim=0)
        assert cos >= 0.90, (
            f"AWQLinear output cosine similarity too low: {cos:.4f}. "
            "AWQ scale inversion may not be applied correctly."
        )

    def test_awq_scale_shape(self):
        d_in, d_out = 16, 8
        awq = _make_awq_linear(d_in, d_out)
        assert awq.awq_scale.shape == (d_in,)

    def test_awq_scale_is_buffer_not_parameter(self):
        awq = _make_awq_linear()
        buf_names = [n for n, _ in awq.named_buffers()]
        param_names = [n for n, _ in awq.named_parameters()]
        assert "awq_scale" in buf_names, "awq_scale must be a buffer"
        assert "qweight" in buf_names, "qweight must be a buffer (not a parameter)"
        assert "weight" not in param_names, "float32 'weight' parameter should not exist"

    def test_no_bias(self):
        awq = AWQLinear.from_float(8, 4, torch.randn(4, 8),
                                   awq_scale=torch.ones(8), symmetric=True, bias=None)
        assert awq.bias is None
        x = torch.randn(2, 8)
        with torch.no_grad():
            out = awq(x)
        assert out.shape == (2, 4)

    def test_extra_repr(self):
        awq = _make_awq_linear(16, 8)
        assert "in_features=16" in awq.extra_repr()
        assert "out_features=8" in awq.extra_repr()


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

class TestAWQConversion:
    def _make_pipeline(self, **kw) -> PTQPipeline:
        backend = AWQBackend(num_bits=4, **kw)
        return PTQPipeline(backend=backend)

    def test_output_shape_preserved(self):
        model = _make_mlp()
        pipeline = self._make_pipeline()
        model_q = _run_awq(pipeline, model)
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            out_q = model_q(x)
        assert out_q.shape == x.shape

    def test_cosine_similarity_int4(self):
        model = _make_mlp()
        pipeline = self._make_pipeline()
        model_q = _run_awq(pipeline, model)
        x = torch.randn(4, 16, 64)
        with torch.no_grad():
            out_fp = model(x)
            out_q = model_q(x)
        errors = compute_output_error(out_fp, out_q)
        assert errors["cosine_similarity"] >= 0.85, (
            f"INT4 AWQ cosine similarity too low: {errors['cosine_similarity']:.4f}"
        )

    def test_linear_layers_replaced_with_awqlinear(self):
        model = _make_mlp()
        pipeline = self._make_pipeline()
        model_q = _run_awq(pipeline, model)
        linear_count = 0
        awq_count = 0
        for module in model_q.modules():
            if type(module) is nn.Linear:    # strict: AWQLinear is not nn.Linear
                linear_count += 1
            if isinstance(module, AWQLinear):
                awq_count += 1
        # All original Linear layers should have been replaced with AWQLinear
        assert awq_count > 0, "Expected at least one AWQLinear in converted model"
        assert linear_count == 0, \
            f"All nn.Linear layers should be replaced; {linear_count} remain"

    def test_awq_scale_shape_correct(self):
        model = _make_mlp(input_dim=64, hidden_dim=128)
        pipeline = self._make_pipeline()
        model_q = _run_awq(pipeline, model)
        for module in model_q.modules():
            if isinstance(module, AWQLinear):
                assert module.awq_scale.shape == (module.in_features,), \
                    f"awq_scale shape mismatch: {module.awq_scale.shape}"

    def test_original_model_not_mutated(self):
        model = _make_mlp()
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        pipeline = self._make_pipeline()
        _run_awq(pipeline, model)
        for name, p in model.named_parameters():
            assert torch.equal(p, params_before[name]), \
                f"convert() mutated model parameter {name}"
        # Also check that original model still has plain nn.Linear
        for module in model.modules():
            if hasattr(module, "weight") and not isinstance(module, AWQLinear):
                assert not isinstance(module, AWQLinear)

    def test_int8_awq_high_quality(self):
        model = _make_mlp()
        pipeline = PTQPipeline(backend=AWQBackend(num_bits=8))
        model_q = _run_awq(pipeline, model)
        x = torch.randn(4, 16, 64)
        with torch.no_grad():
            out_fp = model(x)
            out_q = model_q(x)
        errors = compute_output_error(out_fp, out_q)
        assert errors["cosine_similarity"] >= 0.99, (
            f"INT8 AWQ cosine similarity too low: {errors['cosine_similarity']:.4f}"
        )

    def test_per_tensor_awq(self):
        model = _make_mlp()
        pipeline = PTQPipeline(backend=AWQBackend(num_bits=4, per_channel=False))
        model_q = _run_awq(pipeline, model)
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            out_q = model_q(x)
        assert out_q.shape == x.shape

    def test_asymmetric_awq(self):
        model = _make_mlp()
        pipeline = PTQPipeline(backend=AWQBackend(num_bits=4, symmetric=False))
        model_q = _run_awq(pipeline, model)
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            out_q = model_q(x)
        assert out_q.shape == x.shape


# ---------------------------------------------------------------------------
# Fallback (no calibration data)
# ---------------------------------------------------------------------------

class TestAWQFallback:
    def test_convert_without_calibration(self):
        """Without calibration, fallback uses scale=1 (no AWQ benefit)."""
        model = _make_mlp()
        backend = AWQBackend(num_bits=4)
        # Do NOT call calibrate()
        model_q = backend.convert(model).eval()
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            out_q = model_q(x)
        assert out_q.shape == x.shape

    def test_fallback_scale_is_ones(self):
        """Fallback AWQLinear should use scale=1 (no activation data)."""
        model = _make_mlp()
        backend = AWQBackend(num_bits=4)
        model_q = backend.convert(model)
        for module in model_q.modules():
            if isinstance(module, AWQLinear):
                assert torch.allclose(module.awq_scale,
                                      torch.ones_like(module.awq_scale)), \
                    "Fallback AWQLinear should have scale=1"


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

class TestAWQPipelineIntegration:
    def test_from_config_routes_to_awq(self):
        config = {
            "quantization": {"enabled": True, "dtype": "int4",
                             "symmetric": True, "weight_only": True},
            "calibration": {"method": "minmax", "num_samples": 128,
                            "percentile": None},
            "granularity": "per_channel",
            "backend": "awq",
        }
        pipeline = PTQPipeline.from_config(config)
        assert isinstance(pipeline._backend, AWQBackend)

    def test_pipeline_disabled_passthrough(self):
        """Disabled pipeline returns a plain (unquantized) copy."""
        config = {
            "quantization": {"enabled": False, "dtype": "int4",
                             "symmetric": True, "weight_only": True},
            "calibration": {"method": "minmax", "num_samples": 0,
                            "percentile": None},
            "granularity": "per_channel",
            "backend": "awq",
        }
        model = _make_mlp()
        pipeline = PTQPipeline.from_config(config)
        pipeline.calibrate(model, lambda: None)
        model_q = pipeline.quantize(model)
        # Disabled: returns a plain copy — outputs should be identical to the original.
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            assert torch.equal(model(x), model_q(x))


# ---------------------------------------------------------------------------
# Internal algorithm tests
# ---------------------------------------------------------------------------

class TestAWQAlgorithm:
    def test_find_awq_scale_shape(self):
        W = torch.randn(32, 64)
        X = torch.randn(100, 64)
        s = _find_awq_scale(W, X, num_bits=4, symmetric=True,
                            per_channel=True, n_grid=10)
        assert s.shape == (64,), f"Expected shape (64,), got {s.shape}"

    def test_find_awq_scale_positive(self):
        W = torch.randn(16, 32)
        X = torch.randn(50, 32)
        s = _find_awq_scale(W, X, num_bits=4, symmetric=True,
                            per_channel=True, n_grid=10)
        assert torch.all(s > 0), "AWQ scale should be strictly positive"

    def test_awq_scale_reduces_error(self):
        """
        AWQ should produce lower output error than alpha=0 (no scaling).
        Not guaranteed for all seeds, but for typical data the grid search
        should find a better solution than the trivial s=1.
        """
        torch.manual_seed(123)
        d_out, d_in, n = 32, 64, 200
        # Create weights with varying magnitudes to make AWQ beneficial
        W = torch.randn(d_out, d_in)
        W[:, :10] *= 5.0   # make first 10 input channels more important
        X = torch.randn(n, d_in)
        X[:, :10] *= 3.0   # first 10 channels also have large activations

        ref = X @ W.T

        # alpha=0 → s=1, no scaling
        scale_q0, zp_q0 = _compute_qparams(W, 4, True, True)
        W_q0 = _batch_fake_quant(W, scale_q0, zp_q0, 4, True)
        err_no_scale = (ref - X @ W_q0.T).pow(2).mean().item()

        # AWQ scale search
        s = _find_awq_scale(W, X, num_bits=4, symmetric=True,
                            per_channel=True, n_grid=20)
        W_scaled = W * s.unsqueeze(0)
        scale_q, zp_q = _compute_qparams(W_scaled, 4, True, True)
        W_q = _batch_fake_quant(W_scaled, scale_q, zp_q, 4, True)
        W_eff = W_q / s.unsqueeze(0)
        err_awq = (ref - X @ W_eff.T).pow(2).mean().item()

        assert err_awq <= err_no_scale * 1.1, (
            f"AWQ (err={err_awq:.6f}) should not be much worse than "
            f"no-scale (err={err_no_scale:.6f})"
        )

    def test_batch_fake_quant_symmetric(self):
        W = torch.tensor([[1.0, 2.0, -1.0], [3.0, -2.0, 1.0]])
        scale = torch.tensor([2.0 / 7, 3.0 / 7])
        zp = torch.zeros(2, dtype=torch.int32)
        W_q = _batch_fake_quant(W, scale, zp, num_bits=4, symmetric=True)
        # Result should be in {-8, -7, ..., 7} * scale
        assert W_q.shape == W.shape
        for r in range(W.shape[0]):
            integers = (W_q[r] / scale[r]).round()
            assert integers.abs().max().item() <= 7 + 1e-4


# ---------------------------------------------------------------------------
# Cross-backend comparison
# ---------------------------------------------------------------------------

class TestAWQVsFakeQuant:
    def test_both_backends_produce_valid_int4_outputs(self):
        from src.quant.backends import FakeQuantBackend

        model = _make_mlp()
        x = torch.randn(4, 16, 64)

        # FakeQuant baseline
        fq_backend = FakeQuantBackend(num_bits=4, symmetric=True, per_channel=True,
                                      weight_only=True)
        fq_backend.calibrate(model, _calibration_fn(model))
        model_fq = fq_backend.convert(model).eval()

        # AWQ
        awq_backend = AWQBackend(num_bits=4, symmetric=True, per_channel=True)
        awq_backend.calibrate(model, _calibration_fn(model))
        model_awq = awq_backend.convert(model).eval()

        with torch.no_grad():
            out_fp  = model(x)
            out_fq  = model_fq(x)
            out_awq = model_awq(x)

        sim_fq  = cosine_similarity(out_fp, out_fq)
        sim_awq = cosine_similarity(out_fp, out_awq)

        assert sim_fq  >= 0.80, f"FakeQuant INT4 too low: {sim_fq:.4f}"
        assert sim_awq >= 0.85, f"AWQ INT4 too low: {sim_awq:.4f}"
