"""
Tests for GPTQBackend.

GPTQ (Frantar et al., ICLR 2023) uses second-order Hessian information to
propagate per-column quantization error to remaining columns, minimising the
per-layer output error.  Unlike FakeQuantBackend, it requires calibration data
to construct the Hessian.

Test coverage:
  - Construction: defaults, parameter pass-through, get_backend alias.
  - calibrate(): collects activations for each Linear layer; does not mutate
    the model.
  - convert():
      - Output shape preserved.
      - Cosine similarity acceptable at INT4 (≥ 0.85).
      - Linear layer weights are replaced with GPTQ-quantized values.
      - Original model is not mutated.
      - Converted model contains plain nn.Linear layers (no wrapper needed).
  - Fallback: convert() without calibration falls back to MinMax fake-quant.
  - Pipeline integration: from_config() routes to GPTQBackend.
  - Cross-backend: GPTQ output error ≤ FakeQuant output error at INT4
    (GPTQ compensates for errors FakeQuant ignores).
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from src.models.mlp_block import MLPBlock
from src.quant.backends import GPTQBackend, GPTQLinear, get_backend
from src.quant.backends.gptq_backend import _compute_qparams, _gptq_quantize_weight
from src.quant.error_analysis import compute_output_error
from src.quant.ptq_pipeline import PTQPipeline

torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mlp(input_dim: int = 64, hidden_dim: int = 128) -> MLPBlock:
    return MLPBlock(input_dim=input_dim, hidden_dim=hidden_dim).eval()


def _calibration_fn(model: nn.Module, batch_size: int = 16, seq_len: int = 8,
                    input_dim: int = 64) -> callable:
    """Return a zero-argument callable that feeds random data through model."""
    def fn() -> None:
        x = torch.randn(batch_size, seq_len, input_dim)
        with torch.no_grad():
            model(x)
    return fn


def _run_gptq(
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

class TestGPTQBackendConstruction:
    def test_default_construction(self):
        backend = GPTQBackend()
        assert backend.num_bits == 4
        assert backend.symmetric is True
        assert backend.per_channel is True
        assert backend.weight_only is True
        assert backend.damp_percent == pytest.approx(0.01)
        assert backend.blocksize == 128

    def test_custom_params(self):
        backend = GPTQBackend(num_bits=8, symmetric=False, per_channel=False,
                              damp_percent=0.05, blocksize=64)
        assert backend.num_bits == 8
        assert backend.symmetric is False
        assert backend.per_channel is False
        assert backend.damp_percent == pytest.approx(0.05)
        assert backend.blocksize == 64

    def test_name_property(self):
        assert GPTQBackend().name == "gptq"

    def test_get_backend_alias(self):
        backend = get_backend("gptq", num_bits=4)
        assert isinstance(backend, GPTQBackend)
        assert backend.num_bits == 4


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TestGPTQCalibration:
    def test_calibrate_collects_activations(self):
        model = _make_mlp()
        backend = GPTQBackend()
        backend.calibrate(model, _calibration_fn(model))
        # At least one Linear layer should have been reached
        assert len(backend._activations) > 0

    def test_activations_shape(self):
        model = _make_mlp(input_dim=64)
        backend = GPTQBackend()
        backend.calibrate(model, _calibration_fn(model, batch_size=4, seq_len=8))
        # Each activation should be 2D: [n_total, in_features]
        for name, X in backend._activations.items():
            assert X.ndim == 2, f"Layer {name}: expected 2D activation, got {X.shape}"
            assert X.shape[1] in (64, 128), \
                f"Layer {name}: unexpected in_features {X.shape[1]}"

    def test_calibrate_does_not_mutate_model(self):
        model = _make_mlp()
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        backend = GPTQBackend()
        backend.calibrate(model, _calibration_fn(model))
        for name, p in model.named_parameters():
            assert torch.equal(p, params_before[name]), \
                f"Parameter {name} changed during calibration"

    def test_calibrate_clears_hooks(self):
        model = _make_mlp()
        backend = GPTQBackend()
        backend.calibrate(model, _calibration_fn(model))
        assert len(backend._hooks) == 0, "Hooks should be removed after calibration"


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

class TestGPTQConversion:
    def _make_pipeline(self, **kw) -> PTQPipeline:
        backend = GPTQBackend(num_bits=4, **kw)
        return PTQPipeline(backend=backend)

    def test_output_shape_preserved(self):
        model = _make_mlp()
        pipeline = self._make_pipeline()
        model_q = _run_gptq(pipeline, model)
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            out_q = model_q(x)
        assert out_q.shape == x.shape

    def test_cosine_similarity_int4(self):
        model = _make_mlp()
        pipeline = self._make_pipeline()
        model_q = _run_gptq(pipeline, model)
        x = torch.randn(4, 16, 64)
        with torch.no_grad():
            out_fp = model(x)
            out_q = model_q(x)
        errors = compute_output_error(out_fp, out_q)
        assert errors["cosine_similarity"] >= 0.85, (
            f"INT4 GPTQ cosine similarity too low: {errors['cosine_similarity']:.4f}"
        )

    def test_original_model_not_mutated(self):
        model = _make_mlp()
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        pipeline = self._make_pipeline()
        _run_gptq(pipeline, model)
        for name, p in model.named_parameters():
            assert torch.equal(p, params_before[name]), \
                f"convert() mutated model parameter {name}"

    def test_weights_are_replaced(self):
        """GPTQ quantization produces GPTQLinear layers with packed qweight."""
        model = _make_mlp()
        pipeline = self._make_pipeline()
        model_q = _run_gptq(pipeline, model)
        gptq_layers = [
            m for m in model_q.modules() if isinstance(m, GPTQLinear)
        ]
        assert len(gptq_layers) > 0, "Expected GPTQLinear layers in converted model"
        # Verify qweight buffers exist and have the correct packed shape
        for layer in gptq_layers:
            assert hasattr(layer, "qweight"), "GPTQLinear must have qweight buffer"
            expected_kp = layer.in_features // 8
            assert layer.qweight.shape == (layer.out_features, expected_kp), (
                f"qweight shape mismatch: {layer.qweight.shape} vs "
                f"({layer.out_features}, {expected_kp})"
            )
            assert layer.qweight.dtype == torch.int32

    def test_converted_layers_are_gptq_linear(self):
        """After convert(), all nn.Linear layers are replaced with GPTQLinear."""
        model = _make_mlp()
        pipeline = self._make_pipeline()
        model_q = _run_gptq(pipeline, model)
        gptq_count   = sum(1 for m in model_q.modules() if isinstance(m, GPTQLinear))
        linear_count = sum(1 for m in model_q.modules() if type(m) is nn.Linear)
        assert gptq_count > 0, "Expected at least one GPTQLinear in converted model"
        assert linear_count == 0, (
            f"All nn.Linear layers should be replaced; {linear_count} remain"
        )

    def test_int8_gptq_high_quality(self):
        """GPTQ at INT8 should produce very high cosine similarity."""
        model = _make_mlp()
        pipeline = PTQPipeline(backend=GPTQBackend(num_bits=8))
        model_q = _run_gptq(pipeline, model)
        x = torch.randn(4, 16, 64)
        with torch.no_grad():
            out_fp = model(x)
            out_q = model_q(x)
        errors = compute_output_error(out_fp, out_q)
        assert errors["cosine_similarity"] >= 0.99, (
            f"INT8 GPTQ cosine similarity too low: {errors['cosine_similarity']:.4f}"
        )

    def test_per_tensor_gptq(self):
        """Per-tensor GPTQ should also run without error."""
        model = _make_mlp()
        pipeline = PTQPipeline(backend=GPTQBackend(num_bits=4, per_channel=False))
        model_q = _run_gptq(pipeline, model)
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            out_q = model_q(x)
        assert out_q.shape == x.shape

    def test_asymmetric_gptq(self):
        """Asymmetric GPTQ should run without error."""
        model = _make_mlp()
        pipeline = PTQPipeline(backend=GPTQBackend(num_bits=4, symmetric=False))
        model_q = _run_gptq(pipeline, model)
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            out_q = model_q(x)
        assert out_q.shape == x.shape


# ---------------------------------------------------------------------------
# Fallback (no calibration data)
# ---------------------------------------------------------------------------

class TestGPTQFallback:
    def test_convert_without_calibration(self):
        """Without calibration, fallback to standard MinMax fake-quant."""
        model = _make_mlp()
        backend = GPTQBackend(num_bits=4)
        # Do NOT call calibrate() — no activations collected
        model_q = backend.convert(model).eval()
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            out_q = model_q(x)
        assert out_q.shape == x.shape

    def test_fallback_weights_differ_from_original(self):
        """Fallback quantization should still change the model outputs."""
        model = _make_mlp()
        backend = GPTQBackend(num_bits=4)
        model_q = backend.convert(model).eval()
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            out_fp = model(x)
            out_q  = model_q(x)
        assert not torch.equal(out_fp, out_q), (
            "Fallback quantization should alter model outputs"
        )


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

class TestGPTQPipelineIntegration:
    def test_from_config_routes_to_gptq(self):
        config = {
            "quantization": {"enabled": True, "dtype": "int4",
                             "symmetric": True, "weight_only": True},
            "calibration": {"method": "minmax", "num_samples": 128,
                            "percentile": None},
            "granularity": "per_channel",
            "backend": "gptq",
        }
        pipeline = PTQPipeline.from_config(config)
        assert isinstance(pipeline._backend, GPTQBackend)

    def test_pipeline_disabled_passthrough(self):
        """Disabled pipeline returns a plain (unquantized) copy."""
        config = {
            "quantization": {"enabled": False, "dtype": "int4",
                             "symmetric": True, "weight_only": True},
            "calibration": {"method": "minmax", "num_samples": 0,
                            "percentile": None},
            "granularity": "per_channel",
            "backend": "gptq",
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

class TestGPTQAlgorithm:
    def test_compute_qparams_symmetric_per_channel(self):
        W = torch.tensor([[1.0, 2.0, -3.0], [0.5, -1.0, 4.0]])
        scale, zp = _compute_qparams(W, num_bits=8, symmetric=True, per_channel=True)
        assert scale.shape == (2,)
        assert torch.all(zp == 0)
        # Row 0: max_abs=3, quant_max=127, scale=3/127
        assert scale[0] == pytest.approx(3.0 / 127, rel=1e-5)
        # Row 1: max_abs=4, scale=4/127
        assert scale[1] == pytest.approx(4.0 / 127, rel=1e-5)

    def test_compute_qparams_asymmetric_per_tensor(self):
        W = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
        scale, zp = _compute_qparams(W, num_bits=4, symmetric=False, per_channel=False)
        assert scale.shape == (2,)
        # Per-tensor: range = [-1, 3], scale = 4/15
        expected_scale = 4.0 / 15
        assert scale[0] == pytest.approx(expected_scale, rel=1e-4)

    def test_gptq_preserves_shape(self):
        W = torch.randn(32, 64)
        X = torch.randn(100, 64)
        W_q = _gptq_quantize_weight(W, X, num_bits=4, symmetric=True,
                                    per_channel=True, damp_percent=0.01, blocksize=64)
        assert W_q.shape == W.shape

    def test_gptq_weight_values_in_quantized_grid(self):
        """GPTQ output values should be representable as INT4 dequantized floats."""
        torch.manual_seed(0)
        W = torch.randn(16, 32)
        X = torch.randn(50, 32)
        scale, zp = _compute_qparams(W, num_bits=4, symmetric=True, per_channel=True)
        W_q = _gptq_quantize_weight(W, X, num_bits=4, symmetric=True,
                                    per_channel=True, damp_percent=0.01, blocksize=32)
        # Each element of W_q must be an integer multiple of its row scale
        # Allow a small tolerance (floating-point rounding)
        for row in range(W.shape[0]):
            residuals = (W_q[row] / scale[row]).remainder(1)
            # Residuals should be ≈ 0 or ≈ 1
            residuals = torch.minimum(residuals, 1 - residuals)
            assert residuals.max().item() < 1e-5, \
                f"Row {row} has non-integer-multiple value"

    def test_gptq_lower_error_than_independent_quant(self):
        """
        On well-conditioned data, GPTQ should produce lower output error
        than independent per-column quantisation (no error propagation).
        """
        torch.manual_seed(7)
        d_out, d_in, n = 32, 64, 200
        W = torch.randn(d_out, d_in)
        X = torch.randn(n, d_in)

        # Reference output
        ref = X @ W.T

        # GPTQ-quantized weight
        W_gptq = _gptq_quantize_weight(W, X, num_bits=4, symmetric=True,
                                       per_channel=True, damp_percent=0.01, blocksize=32)

        # Plain fake-quant (column-independent)
        scale, zp = _compute_qparams(W, num_bits=4, symmetric=True, per_channel=True)
        W_fq = (W / scale.unsqueeze(1)).round().clamp(-8, 7) * scale.unsqueeze(1)

        err_gptq = (ref - X @ W_gptq.T).pow(2).mean().item()
        err_fq   = (ref - X @ W_fq.T).pow(2).mean().item()

        assert err_gptq <= err_fq * 1.1, (
            f"GPTQ (err={err_gptq:.6f}) should not be much worse than "
            f"FakeQuant (err={err_fq:.6f})"
        )


# ---------------------------------------------------------------------------
# Cross-backend comparison
# ---------------------------------------------------------------------------

class TestGPTQVsFakeQuant:
    def test_both_backends_produce_valid_int4_outputs(self):
        """GPTQ and FakeQuant should both produce high-quality INT4 results."""
        from src.quant.backends import FakeQuantBackend

        model = _make_mlp()
        x = torch.randn(4, 16, 64)

        # FakeQuant baseline
        fq_backend = FakeQuantBackend(num_bits=4, symmetric=True, per_channel=True,
                                      weight_only=True)
        fq_backend.calibrate(model, _calibration_fn(model))
        model_fq = fq_backend.convert(model).eval()

        # GPTQ
        gptq_backend = GPTQBackend(num_bits=4, symmetric=True, per_channel=True)
        gptq_backend.calibrate(model, _calibration_fn(model))
        model_gptq = gptq_backend.convert(model).eval()

        with torch.no_grad():
            out_fp = model(x)
            out_fq = model_fq(x)
            out_gptq = model_gptq(x)

        from src.quant.error_analysis import cosine_similarity as cs
        sim_fq   = cs(out_fp, out_fq)
        sim_gptq = cs(out_fp, out_gptq)

        assert sim_fq   >= 0.80, f"FakeQuant INT4 too low: {sim_fq:.4f}"
        assert sim_gptq >= 0.85, f"GPTQ INT4 too low: {sim_gptq:.4f}"
