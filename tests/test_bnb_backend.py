"""
Tests for BitsAndBytesBackend.

All tests are skipped automatically when bitsandbytes is not installed.
INT4/NF4 tests are additionally skipped when CUDA is not available.

Test coverage:
  - Construction: INT8, INT4, unsupported bit-widths, aliases, from_config.
  - calibrate(): must be a no-op that doesn't mutate the model.
  - convert() INT8: layer replacement, output shape, cosine similarity,
    model contains Linear8bitLt, original not mutated.
  - convert() INT4: layer replacement on CPU (construction only),
    forward pass on CUDA when available.
  - Pipeline integration: disabled passthrough, backend_field routing.
  - Cross-backend: FakeQuant vs BNB both close to float.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

bnb = pytest.importorskip("bitsandbytes", reason="bitsandbytes not installed")

from src.models.mlp_block import MLPBlock
from src.quant.backends import BitsAndBytesBackend, get_backend
from src.quant.error_analysis import compute_output_error
from src.quant.ptq_pipeline import PTQPipeline

CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mlp(input_dim: int = 64, hidden_dim: int = 128) -> MLPBlock:
    return MLPBlock(input_dim=input_dim, hidden_dim=hidden_dim).eval()


def _convert(pipeline: PTQPipeline, model: nn.Module) -> nn.Module:
    """Calibrate (no-op) and convert via pipeline."""
    pipeline.calibrate(model, lambda: None)
    return pipeline.quantize(model).eval()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestBNBBackendConstruction:
    def test_default_is_int8(self):
        backend = BitsAndBytesBackend()
        assert backend.num_bits == 8
        assert backend.name == "bitsandbytes"

    def test_int4_construction(self):
        backend = BitsAndBytesBackend(num_bits=4)
        assert backend.num_bits == 4
        assert backend.bnb_4bit_quant_type == "nf4"

    def test_rejects_unsupported_bits(self):
        with pytest.raises(ValueError, match="4-bit and 8-bit only"):
            BitsAndBytesBackend(num_bits=2)

    def test_get_backend_aliases(self):
        assert isinstance(get_backend("bitsandbytes"), BitsAndBytesBackend)
        assert isinstance(get_backend("bnb"), BitsAndBytesBackend)

    def test_from_config_int8_bnb(self):
        config = {
            "quantization": {"enabled": True, "dtype": "int8", "symmetric": True, "weight_only": True},
            "calibration": {"method": "minmax", "num_samples": 0, "percentile": None},
            "granularity": "per_tensor",
            "backend": "bitsandbytes",
        }
        pipeline = PTQPipeline.from_config(config)
        assert isinstance(pipeline._backend, BitsAndBytesBackend)

    def test_from_config_int4_bnb(self):
        config = {
            "quantization": {"enabled": True, "dtype": "int4", "symmetric": True, "weight_only": True},
            "calibration": {"method": "minmax", "num_samples": 0, "percentile": None},
            "granularity": "per_channel",
            "backend": "bnb",
        }
        pipeline = PTQPipeline.from_config(config)
        assert isinstance(pipeline._backend, BitsAndBytesBackend)
        assert pipeline._backend.num_bits == 4

    def test_threshold_kwarg(self):
        backend = BitsAndBytesBackend(num_bits=8, threshold=0.0)
        assert backend.threshold == 0.0

    def test_double_quant_kwarg(self):
        backend = BitsAndBytesBackend(num_bits=4, bnb_4bit_use_double_quant=True)
        assert backend.bnb_4bit_use_double_quant is True


# ---------------------------------------------------------------------------
# Calibration (no-op)
# ---------------------------------------------------------------------------

class TestBNBCalibration:
    def test_calibrate_accepts_noop_fn(self):
        backend = BitsAndBytesBackend()
        model = _make_mlp()
        backend.calibrate(model, lambda: None)  # must not raise

    def test_calibrate_does_not_mutate_model(self):
        model = _make_mlp()
        original_params = {n: p.clone() for n, p in model.named_parameters()}
        backend = BitsAndBytesBackend()
        backend.calibrate(model, lambda: None)
        for name, param in model.named_parameters():
            assert torch.equal(param, original_params[name]), (
                f"Parameter {name} was mutated by calibrate()"
            )


# ---------------------------------------------------------------------------
# convert() — INT8
# ---------------------------------------------------------------------------

class TestBNBInt8Convert:
    def test_model_contains_linear8bitlt(self):
        """All nn.Linear layers must be replaced with Linear8bitLt."""
        model = _make_mlp()
        pipeline = PTQPipeline(num_bits=8, backend="bitsandbytes")
        model_q = _convert(pipeline, model)
        bnb_layers = [m for m in model_q.modules() if isinstance(m, bnb.nn.Linear8bitLt)]
        plain_linears = [m for m in model_q.modules() if type(m) is nn.Linear]
        assert len(bnb_layers) == 2, f"Expected 2 Linear8bitLt, got {len(bnb_layers)}"
        assert len(plain_linears) == 0

    def test_output_shape_preserved(self):
        model = _make_mlp()
        pipeline = PTQPipeline(num_bits=8, backend="bitsandbytes")
        model_q = _convert(pipeline, model)
        x = torch.randn(2, 4, 64)
        with torch.no_grad():
            out = model_q(x)
        assert out.shape == x.shape

    def test_cosine_similarity_acceptable(self):
        """INT8 (LLM.int8()) should stay close to float on CPU."""
        model = _make_mlp()
        pipeline = PTQPipeline(num_bits=8, backend="bitsandbytes")
        model_q = _convert(pipeline, model)
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            ref = model(x)
            q = model_q(x)
        metrics = compute_output_error(ref, q)
        assert metrics["cosine_similarity"] > 0.95

    def test_original_model_not_mutated(self):
        model = _make_mlp()
        original_params = {n: p.clone() for n, p in model.named_parameters()}
        pipeline = PTQPipeline(num_bits=8, backend="bitsandbytes")
        _convert(pipeline, model)
        for name, param in model.named_parameters():
            assert torch.equal(param, original_params[name])

    def test_weight_is_int8params(self):
        """Weights in Linear8bitLt must be wrapped in Int8Params.

        On CPU, Int8Params stores data in float32 and only converts to
        int8 at the first CUDA forward pass. The type check here is the
        correct CPU-safe assertion; dtype==torch.int8 requires CUDA.
        """
        model = _make_mlp()
        pipeline = PTQPipeline(num_bits=8, backend="bitsandbytes")
        model_q = _convert(pipeline, model)
        for m in model_q.modules():
            if isinstance(m, bnb.nn.Linear8bitLt):
                assert isinstance(m.weight, bnb.nn.Int8Params), (
                    f"Expected Int8Params, got {type(m.weight)}"
                )

    @requires_cuda
    def test_weight_dtype_is_int8_on_cuda(self):
        """On CUDA, Linear8bitLt weight dtype must be int8."""
        model = _make_mlp().cuda()
        pipeline = PTQPipeline(num_bits=8, backend="bitsandbytes")
        model_q = _convert(pipeline, model).cuda()
        # Trigger quantization by running a forward pass
        with torch.no_grad():
            model_q(torch.randn(1, 4, 64, device="cuda"))
        for m in model_q.modules():
            if isinstance(m, bnb.nn.Linear8bitLt):
                assert m.weight.dtype == torch.int8, (
                    f"Expected int8 weight on CUDA, got {m.weight.dtype}"
                )

    def test_threshold_zero_disables_outlier_handling(self):
        """threshold=0.0 means all features go through int8 (no outlier decomposition)."""
        model = _make_mlp()
        pipeline = PTQPipeline(num_bits=8, backend="bitsandbytes")
        # Override threshold via backend directly
        pipeline._backend.threshold = 0.0
        model_q = _convert(pipeline, model)
        x = torch.randn(1, 4, 64)
        with torch.no_grad():
            out = model_q(x)  # must not raise
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# convert() — INT4 (construction only on CPU; forward needs CUDA)
# ---------------------------------------------------------------------------

class TestBNBInt4Convert:
    def test_model_contains_linear4bit(self):
        """All nn.Linear layers must be replaced with Linear4bit."""
        model = _make_mlp()
        pipeline = PTQPipeline(num_bits=4, backend="bitsandbytes")
        model_q = _convert(pipeline, model)
        bnb_layers = [m for m in model_q.modules() if isinstance(m, bnb.nn.Linear4bit)]
        assert len(bnb_layers) == 2, f"Expected 2 Linear4bit, got {len(bnb_layers)}"

    def test_original_model_not_mutated(self):
        model = _make_mlp()
        original_params = {n: p.clone() for n, p in model.named_parameters()}
        pipeline = PTQPipeline(num_bits=4, backend="bitsandbytes")
        _convert(pipeline, model)
        for name, param in model.named_parameters():
            assert torch.equal(param, original_params[name])

    def test_quant_type_nf4(self):
        """Default quant type for INT4 is NF4."""
        model = _make_mlp()
        backend = BitsAndBytesBackend(num_bits=4, bnb_4bit_quant_type="nf4")
        backend.calibrate(model, lambda: None)
        model_q = backend.convert(model)
        for m in model_q.modules():
            if isinstance(m, bnb.nn.Linear4bit):
                assert m.weight.quant_type == "nf4"

    @requires_cuda
    def test_int4_forward_on_cuda(self):
        """INT4 NF4 forward pass on CUDA device."""
        model = _make_mlp().cuda()
        pipeline = PTQPipeline(num_bits=4, backend="bitsandbytes")
        model_q = _convert(pipeline, model).cuda()
        x = torch.randn(2, 4, 64, device="cuda")
        with torch.no_grad():
            out = model_q(x)
        assert out.shape == x.shape

    @requires_cuda
    def test_int4_cosine_similarity_on_cuda(self):
        model = _make_mlp().cuda()
        pipeline = PTQPipeline(num_bits=4, backend="bitsandbytes")
        model_q = _convert(pipeline, model).cuda()
        x = torch.randn(2, 8, 64, device="cuda")
        with torch.no_grad():
            ref = model(x)
            q = model_q(x)
        metrics = compute_output_error(ref.cpu(), q.cpu())
        assert metrics["cosine_similarity"] > 0.90


# ---------------------------------------------------------------------------
# Disabled pipeline
# ---------------------------------------------------------------------------

class TestBNBDisabled:
    def test_disabled_returns_plain_copy(self):
        model = _make_mlp()
        pipeline = PTQPipeline(enabled=False, backend="bitsandbytes")
        model_q = pipeline.quantize(model)
        bnb_layers = [
            m for m in model_q.modules()
            if isinstance(m, (bnb.nn.Linear8bitLt, bnb.nn.Linear4bit))
        ]
        assert len(bnb_layers) == 0
        x = torch.randn(1, 4, 64)
        with torch.no_grad():
            assert torch.equal(model(x), model_q(x))


# ---------------------------------------------------------------------------
# Cross-backend: FakeQuant vs BNB (INT8)
# ---------------------------------------------------------------------------

class TestFakeVsBNB:
    def test_both_close_to_float(self):
        """FakeQuantBackend and BNBBackend (INT8) both stay close to float."""
        torch.manual_seed(1)
        model = _make_mlp()
        x = torch.randn(4, 8, 64)

        # FakeQuant INT8
        p_fake = PTQPipeline(num_bits=8, symmetric=True, per_channel=True, backend="fake")
        calib_inputs = [torch.randn(2, 4, 64) for _ in range(8)]
        p_fake.calibrate(model, lambda: [model(xi) for xi in calib_inputs])
        mq_fake = p_fake.quantize(model).eval()

        # BNB INT8
        p_bnb = PTQPipeline(num_bits=8, backend="bitsandbytes")
        p_bnb.calibrate(model, lambda: None)
        mq_bnb = p_bnb.quantize(model).eval()

        with torch.no_grad():
            ref = model(x)
            sim_fake = compute_output_error(ref, mq_fake(x))["cosine_similarity"]
            sim_bnb = compute_output_error(ref, mq_bnb(x))["cosine_similarity"]

        assert sim_fake > 0.95, f"FakeQuant cosine sim too low: {sim_fake:.4f}"
        assert sim_bnb > 0.95, f"BNB cosine sim too low: {sim_bnb:.4f}"
