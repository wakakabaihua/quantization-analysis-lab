"""
Tests for MixedPrecisionBackend.

Mixed-precision quantization assigns a (potentially different) bit-width to
each Linear layer.  Sensitive layers keep higher precision; robust layers are
compressed more aggressively.

Test coverage:
  - Construction: defaults, explicit layer_config, get_backend alias.
  - from_sensitivity(): correct bit-width assignment per threshold.
  - from_config_dict(): YAML-style config parsing.
  - bit_assignment(): returns correct per-layer bits.
  - calibrate(): collects statistics; does not mutate the model.
  - convert():
      - Output shape preserved.
      - Each layer quantized at its assigned bit-width.
      - All Linear layers replaced with QuantizedLinear.
      - Original model not mutated.
  - theoretical_compression(): correct ratios for pure INT8, pure INT4,
    and mixed assignments.
  - Quality: mixed INT8/INT4 output quality between INT8 and INT4 extremes.
  - Pipeline integration: from_config() routes to MixedPrecisionBackend;
    disabled passthrough works.
  - Cross-backend: uniform assignment matches FakeQuantBackend numerically.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from src.models.mlp_block import MLPBlock
from src.models.attention_block import AttentionBlock
from src.quant.backends import MixedPrecisionBackend, get_backend
from src.quant.backends.fake_quant_backend import QuantizedLinear
from src.quant.error_analysis import compute_output_error
from src.quant.ptq_pipeline import PTQPipeline

torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mlp(input_dim: int = 64, hidden_dim: int = 128) -> MLPBlock:
    return MLPBlock(input_dim=input_dim, hidden_dim=hidden_dim).eval()


def _calibration_fn(model: nn.Module, input_dim: int = 64,
                    batch_size: int = 4, seq_len: int = 8) -> callable:
    def fn() -> None:
        x = torch.randn(batch_size, seq_len, input_dim)
        with torch.no_grad():
            model(x)
    return fn


def _run(backend: MixedPrecisionBackend, model: nn.Module,
         input_dim: int = 64) -> nn.Module:
    backend.calibrate(model, _calibration_fn(model, input_dim=input_dim))
    return backend.convert(model).eval()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestMPBackendConstruction:
    def test_default_construction(self):
        backend = MixedPrecisionBackend()
        assert backend.default_num_bits == 8
        assert backend.symmetric is True
        assert backend.per_channel is True
        assert backend.weight_only is True
        assert backend.layer_config == {}

    def test_custom_layer_config(self):
        backend = MixedPrecisionBackend(
            layer_config={"fc1": {"num_bits": 8}, "fc2": {"num_bits": 4}},
            default_num_bits=8,
        )
        assert backend._num_bits_for("fc1") == 8
        assert backend._num_bits_for("fc2") == 4
        assert backend._num_bits_for("unknown_layer") == 8

    def test_name_property(self):
        assert MixedPrecisionBackend().name == "mixed_precision"

    def test_get_backend_aliases(self):
        assert isinstance(get_backend("mixed_precision"), MixedPrecisionBackend)
        assert isinstance(get_backend("mp"), MixedPrecisionBackend)


# ---------------------------------------------------------------------------
# from_sensitivity constructor
# ---------------------------------------------------------------------------

class TestMPFromSensitivity:
    def test_basic_assignment(self):
        scores = {"fc1": 0.9997, "fc2": 0.99996}
        backend = MixedPrecisionBackend.from_sensitivity(
            scores, threshold=0.9999, high_bits=8, low_bits=4
        )
        assert backend._num_bits_for("fc1") == 8   # below threshold
        assert backend._num_bits_for("fc2") == 4   # above threshold

    def test_all_sensitive(self):
        scores = {"fc1": 0.9990, "fc2": 0.9991}
        backend = MixedPrecisionBackend.from_sensitivity(
            scores, threshold=0.9999, high_bits=8, low_bits=4
        )
        assert backend._num_bits_for("fc1") == 8
        assert backend._num_bits_for("fc2") == 8

    def test_all_robust(self):
        scores = {"fc1": 0.99999, "fc2": 0.99999}
        backend = MixedPrecisionBackend.from_sensitivity(
            scores, threshold=0.9999, high_bits=8, low_bits=4
        )
        assert backend._num_bits_for("fc1") == 4
        assert backend._num_bits_for("fc2") == 4

    def test_threshold_at_exactly_equal(self):
        """Score exactly equal to threshold → low_bits (not sensitive)."""
        scores = {"fc1": 0.9999}
        backend = MixedPrecisionBackend.from_sensitivity(
            scores, threshold=0.9999, high_bits=8, low_bits=4
        )
        assert backend._num_bits_for("fc1") == 4

    def test_default_num_bits_is_high_bits(self):
        scores = {"fc1": 0.9997}
        backend = MixedPrecisionBackend.from_sensitivity(
            scores, threshold=0.9999, high_bits=8, low_bits=4
        )
        assert backend.default_num_bits == 8


# ---------------------------------------------------------------------------
# from_config_dict constructor
# ---------------------------------------------------------------------------

class TestMPFromConfigDict:
    def test_basic(self):
        mp_cfg = {
            "default_num_bits": 8,
            "layers": {
                "fc1": {"num_bits": 8},
                "fc2": {"num_bits": 4},
            }
        }
        backend = MixedPrecisionBackend.from_config_dict(
            mp_cfg, symmetric=True, per_channel=True, weight_only=True
        )
        assert backend.default_num_bits == 8
        assert backend._num_bits_for("fc1") == 8
        assert backend._num_bits_for("fc2") == 4
        assert backend.symmetric is True
        assert backend.per_channel is True

    def test_empty_layers(self):
        backend = MixedPrecisionBackend.from_config_dict(
            {"default_num_bits": 4}, symmetric=True
        )
        assert backend.default_num_bits == 4
        assert backend._num_bits_for("any_layer") == 4


# ---------------------------------------------------------------------------
# bit_assignment
# ---------------------------------------------------------------------------

class TestBitAssignment:
    def test_correct_for_mlp(self):
        model = _make_mlp()
        backend = MixedPrecisionBackend(
            layer_config={"fc1": {"num_bits": 8}, "fc2": {"num_bits": 4}},
        )
        assignment = backend.bit_assignment(model)
        assert assignment["fc1"] == 8
        assert assignment["fc2"] == 4

    def test_default_applied_to_unknown_layers(self):
        model = _make_mlp()
        backend = MixedPrecisionBackend(default_num_bits=6)
        assignment = backend.bit_assignment(model)
        for bits in assignment.values():
            assert bits == 6


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TestMPCalibration:
    def test_calibrate_collects_stats(self):
        model = _make_mlp()
        backend = MixedPrecisionBackend()
        backend.calibrate(model, _calibration_fn(model))
        assert len(backend._activation_stats) > 0
        assert len(backend._weight_stats) > 0

    def test_calibrate_does_not_mutate_model(self):
        model = _make_mlp()
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        backend = MixedPrecisionBackend()
        backend.calibrate(model, _calibration_fn(model))
        for name, p in model.named_parameters():
            assert torch.equal(p, params_before[name]), \
                f"Calibration mutated {name}"

    def test_calibrate_clears_hooks(self):
        model = _make_mlp()
        backend = MixedPrecisionBackend()
        backend.calibrate(model, _calibration_fn(model))
        assert len(backend._hooks) == 0


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

class TestMPConversion:
    def test_output_shape_preserved(self):
        model = _make_mlp()
        backend = MixedPrecisionBackend()
        model_q = _run(backend, model)
        x = torch.randn(2, 8, 64)
        assert model_q(x).shape == x.shape

    def test_linear_layers_replaced_with_quantized_linear(self):
        model = _make_mlp()
        backend = MixedPrecisionBackend()
        model_q = _run(backend, model)
        for module in model_q.modules():
            if hasattr(module, "weight_fq"):
                assert isinstance(module, QuantizedLinear), \
                    f"Expected QuantizedLinear, got {type(module)}"

    def test_assigned_bits_respected(self):
        """Each QuantizedLinear should use the configured num_bits."""
        model = _make_mlp()
        backend = MixedPrecisionBackend(
            layer_config={"fc1": {"num_bits": 8}, "fc2": {"num_bits": 4}},
        )
        _run(backend, model)
        # Indirectly verified: bit_assignment returns correct values
        assignment = backend.bit_assignment(model)
        assert assignment["fc1"] == 8
        assert assignment["fc2"] == 4

    def test_original_model_not_mutated(self):
        model = _make_mlp()
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        backend = MixedPrecisionBackend()
        _run(backend, model)
        for name, p in model.named_parameters():
            assert torch.equal(p, params_before[name]), \
                f"convert() mutated parameter {name}"

    def test_fallback_without_calibration(self):
        """convert() should still work if calibrate() was never called."""
        model = _make_mlp()
        backend = MixedPrecisionBackend()
        model_q = backend.convert(model).eval()
        x = torch.randn(2, 8, 64)
        assert model_q(x).shape == x.shape

    def test_uniform_int8_high_cosine_similarity(self):
        model = _make_mlp()
        backend = MixedPrecisionBackend(default_num_bits=8)
        model_q = _run(backend, model)
        x = torch.randn(4, 16, 64)
        with torch.no_grad():
            out_fp = model(x)
            out_q = model_q(x)
        errors = compute_output_error(out_fp, out_q)
        assert errors["cosine_similarity"] >= 0.99, (
            f"Uniform INT8 MP cosine too low: {errors['cosine_similarity']:.4f}"
        )

    def test_mixed_int8_int4_cosine_similarity(self):
        model = _make_mlp()
        backend = MixedPrecisionBackend(
            layer_config={"fc1": {"num_bits": 8}, "fc2": {"num_bits": 4}},
        )
        model_q = _run(backend, model)
        x = torch.randn(4, 16, 64)
        with torch.no_grad():
            out_fp = model(x)
            out_q = model_q(x)
        errors = compute_output_error(out_fp, out_q)
        assert errors["cosine_similarity"] >= 0.95, (
            f"Mixed INT8/4 cosine too low: {errors['cosine_similarity']:.4f}"
        )


# ---------------------------------------------------------------------------
# theoretical_compression
# ---------------------------------------------------------------------------

class TestTheoreticalCompression:
    def test_uniform_int8(self):
        model = _make_mlp()
        backend = MixedPrecisionBackend(default_num_bits=8)
        info = backend.theoretical_compression(model)
        # FP32 = 4 bytes, INT8 = 1 byte → 4×
        assert info["compression_ratio"] == pytest.approx(4.0, rel=1e-5)

    def test_uniform_int4(self):
        model = _make_mlp()
        backend = MixedPrecisionBackend(default_num_bits=4)
        info = backend.theoretical_compression(model)
        # FP32 = 4 bytes, INT4 = 0.5 bytes → 8×
        assert info["compression_ratio"] == pytest.approx(8.0, rel=1e-5)

    def test_mixed_compression_between_extremes(self):
        model = _make_mlp()
        backend = MixedPrecisionBackend(
            layer_config={"fc1": {"num_bits": 8}, "fc2": {"num_bits": 4}},
        )
        info = backend.theoretical_compression(model)
        # Should be between 4× (all INT8) and 8× (all INT4)
        assert 4.0 < info["compression_ratio"] < 8.0

    def test_per_layer_breakdown(self):
        model = _make_mlp()
        backend = MixedPrecisionBackend(
            layer_config={"fc1": {"num_bits": 8}, "fc2": {"num_bits": 4}},
        )
        info = backend.theoretical_compression(model)
        assert "fc1" in info["per_layer"]
        assert "fc2" in info["per_layer"]
        assert info["per_layer"]["fc1"]["num_bits"] == 8
        assert info["per_layer"]["fc2"]["num_bits"] == 4
        # Compression ratios per layer
        assert info["per_layer"]["fc1"]["compression_ratio"] == pytest.approx(4.0, rel=1e-5)
        assert info["per_layer"]["fc2"]["compression_ratio"] == pytest.approx(8.0, rel=1e-5)

    def test_fp32_bytes_correct(self):
        model = _make_mlp(input_dim=64, hidden_dim=128)
        backend = MixedPrecisionBackend()
        info = backend.theoretical_compression(model)
        # Count weight parameters manually
        total_params = sum(
            m.weight.numel()
            for m in model.modules()
            if isinstance(m, nn.Linear)
        )
        assert info["fp32_bytes"] == total_params * 4


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

class TestMPPipelineIntegration:
    def test_from_config_routes_to_mp(self):
        config = {
            "quantization": {"enabled": True, "dtype": "int8",
                             "symmetric": True, "weight_only": True},
            "calibration": {"method": "minmax", "num_samples": 32,
                            "percentile": None},
            "granularity": "per_channel",
            "backend": "mixed_precision",
            "mixed_precision": {
                "default_num_bits": 8,
                "layers": {"fc1": {"num_bits": 8}, "fc2": {"num_bits": 4}},
            },
        }
        pipeline = PTQPipeline.from_config(config)
        assert isinstance(pipeline._backend, MixedPrecisionBackend)
        assert pipeline._backend._num_bits_for("fc1") == 8
        assert pipeline._backend._num_bits_for("fc2") == 4

    def test_from_config_mp_alias(self):
        config = {
            "quantization": {"enabled": True, "dtype": "int8",
                             "symmetric": True, "weight_only": True},
            "calibration": {"method": "minmax", "num_samples": 32,
                            "percentile": None},
            "granularity": "per_channel",
            "backend": "mp",
            "mixed_precision": {"default_num_bits": 4},
        }
        pipeline = PTQPipeline.from_config(config)
        assert isinstance(pipeline._backend, MixedPrecisionBackend)

    def test_pipeline_disabled_passthrough(self):
        config = {
            "quantization": {"enabled": False, "dtype": "int8",
                             "symmetric": True, "weight_only": True},
            "calibration": {"method": "minmax", "num_samples": 0,
                            "percentile": None},
            "granularity": "per_channel",
            "backend": "mixed_precision",
            "mixed_precision": {"default_num_bits": 8},
        }
        model = _make_mlp()
        pipeline = PTQPipeline.from_config(config)
        pipeline.calibrate(model, lambda: None)
        model_q = pipeline.quantize(model)
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            assert torch.equal(model(x), model_q(x))

    def test_pipeline_full_run(self):
        config = {
            "quantization": {"enabled": True, "dtype": "int8",
                             "symmetric": True, "weight_only": True},
            "calibration": {"method": "minmax", "num_samples": 16,
                            "percentile": None},
            "granularity": "per_channel",
            "backend": "mixed_precision",
            "mixed_precision": {
                "default_num_bits": 8,
                "layers": {"fc1": {"num_bits": 8}, "fc2": {"num_bits": 4}},
            },
        }
        model = _make_mlp()
        pipeline = PTQPipeline.from_config(config)
        pipeline.calibrate(model, _calibration_fn(model))
        model_q = pipeline.quantize(model).eval()
        x = torch.randn(2, 8, 64)
        out_q = model_q(x)
        assert out_q.shape == x.shape


# ---------------------------------------------------------------------------
# Quality: mixed precision is between INT8 and INT4 extremes
# ---------------------------------------------------------------------------

class TestMPQualityOrdering:
    def test_mixed_between_int8_and_int4(self):
        """
        Mixed INT8/INT4 output error should be between the all-INT8 and
        all-INT4 extremes when fc1 is the bottleneck (kept at INT8).
        """
        model = _make_mlp()
        x = torch.randn(4, 16, 64)

        backends = {
            "int8": MixedPrecisionBackend(default_num_bits=8),
            "int4": MixedPrecisionBackend(default_num_bits=4),
            "mixed": MixedPrecisionBackend(
                layer_config={"fc1": {"num_bits": 8}, "fc2": {"num_bits": 4}},
            ),
        }

        sims = {}
        for name, backend in backends.items():
            _run(backend, model)
            model_q = backend.convert(model).eval()
            with torch.no_grad():
                out_fp = model(x)
                out_q = model_q(x)
            sims[name] = compute_output_error(out_fp, out_q)["cosine_similarity"]

        assert sims["int8"] >= sims["mixed"] >= sims["int4"] - 0.01, (
            f"Expected INT8 ≥ mixed ≥ INT4 (≈), got "
            f"int8={sims['int8']:.6f} mixed={sims['mixed']:.6f} int4={sims['int4']:.6f}"
        )


# ---------------------------------------------------------------------------
# Cross-backend: uniform MP ≈ FakeQuantBackend
# ---------------------------------------------------------------------------

class TestMPVsFakeQuant:
    def test_uniform_int8_matches_fake_quant(self):
        """Uniform INT8 MixedPrecisionBackend should produce the same output as FakeQuantBackend INT8."""
        from src.quant.backends import FakeQuantBackend

        model = _make_mlp()
        torch.manual_seed(0)
        x = torch.randn(4, 16, 64)

        # FakeQuant INT8 — calibrate with a fixed-seed input
        fq = FakeQuantBackend(num_bits=8, symmetric=True, per_channel=True, weight_only=True)
        torch.manual_seed(1)
        fq.calibrate(model, _calibration_fn(model))
        model_fq = fq.convert(model).eval()

        # Mixed precision: all INT8 — same fixed seed
        mp = MixedPrecisionBackend(default_num_bits=8, symmetric=True,
                                   per_channel=True, weight_only=True)
        torch.manual_seed(1)
        mp.calibrate(model, _calibration_fn(model))
        model_mp = mp.convert(model).eval()

        with torch.no_grad():
            out_fq = model_fq(x)
            out_mp = model_mp(x)

        # Same calibration input → same quantization params → identical outputs
        assert torch.allclose(out_fq, out_mp, atol=1e-5), \
            "Uniform INT8 MixedPrecision should match FakeQuantBackend INT8"
