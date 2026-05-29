"""
Integration tests for the PTQ pipeline.

Covers end-to-end correctness: calibrate → quantize → measure error.
Key invariants tested:
  - Disabled pipeline is a passthrough.
  - Per-channel quantization achieves higher or equal cosine similarity
    compared to per-tensor for INT8.
  - Output shapes are always preserved.
  - Weight-only quantization does not add activation FakeQuantize.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.mlp_block import MLPBlock
from src.quant.error_analysis import compute_output_error
from src.quant.ptq_pipeline import PTQPipeline, QuantizedLinear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mlp(input_dim: int = 64, hidden_dim: int = 128) -> MLPBlock:
    return MLPBlock(input_dim=input_dim, hidden_dim=hidden_dim).eval()


def _calib_fn(model: nn.Module, inputs: list):
    def fn() -> None:
        for x in inputs:
            with torch.no_grad():
                model(x)
    return fn


def _run(pipeline: PTQPipeline, model: nn.Module, n_calib: int = 8):
    inputs = [torch.randn(2, 4, 64) for _ in range(n_calib)]
    pipeline.calibrate(model, _calib_fn(model, inputs))
    model_q = pipeline.quantize(model)
    model_q.eval()
    return model_q


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPTQPipelineDisabled:
    def test_passthrough_output_identical(self):
        """With enabled=False the quantized model output must match exactly."""
        model = _make_mlp()
        pipeline = PTQPipeline(enabled=False)
        model_q = _run(pipeline, model)
        x = torch.randn(1, 4, 64)
        with torch.no_grad():
            assert torch.equal(model(x), model_q(x))

    def test_no_quantized_linear_modules(self):
        """Disabled pipeline should not introduce any QuantizedLinear."""
        model = _make_mlp()
        model_q = _run(PTQPipeline(enabled=False), model)
        q_linears = [m for m in model_q.modules() if isinstance(m, QuantizedLinear)]
        assert len(q_linears) == 0


class TestINT8PerTensor:
    def test_output_shape_preserved(self):
        model = _make_mlp()
        pipeline = PTQPipeline(num_bits=8, symmetric=True, per_channel=False)
        model_q = _run(pipeline, model)
        x = torch.randn(2, 4, 64)
        with torch.no_grad():
            out = model_q(x)
        assert out.shape == x.shape

    def test_high_cosine_similarity(self):
        model = _make_mlp()
        pipeline = PTQPipeline(num_bits=8, symmetric=True, per_channel=False)
        model_q = _run(pipeline, model)
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            ref = model(x)
            q = model_q(x)
        metrics = compute_output_error(ref, q)
        assert metrics["cosine_similarity"] > 0.95

    def test_quantized_linears_present(self):
        model = _make_mlp()
        model_q = _run(PTQPipeline(num_bits=8, per_channel=False), model)
        q_linears = [m for m in model_q.modules() if isinstance(m, QuantizedLinear)]
        assert len(q_linears) == 2  # fc1 and fc2


class TestINT8PerChannel:
    def test_higher_similarity_than_per_tensor(self):
        """Per-channel should achieve >= cosine similarity than per-tensor."""
        torch.manual_seed(7)
        model = _make_mlp()
        inputs = [torch.randn(4, 8, 64) for _ in range(8)]
        x = torch.randn(2, 8, 64)

        def sim(per_channel: bool) -> float:
            p = PTQPipeline(num_bits=8, symmetric=True, per_channel=per_channel)
            p.calibrate(model, _calib_fn(model, inputs))
            mq = p.quantize(model).eval()
            with torch.no_grad():
                return compute_output_error(model(x), mq(x))["cosine_similarity"]

        assert sim(per_channel=True) >= sim(per_channel=False) - 0.01

    def test_per_channel_cosine_similarity(self):
        model = _make_mlp()
        pipeline = PTQPipeline(num_bits=8, symmetric=True, per_channel=True)
        model_q = _run(pipeline, model)
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            metrics = compute_output_error(model(x), model_q(x))
        assert metrics["cosine_similarity"] > 0.97


class TestWeightOnlyQuantization:
    def test_no_activation_fq(self):
        """Weight-only mode must not add act_fq to any QuantizedLinear."""
        model = _make_mlp()
        pipeline = PTQPipeline(
            num_bits=4, symmetric=True, per_channel=True, weight_only=True
        )
        model_q = _run(pipeline, model)
        for m in model_q.modules():
            if isinstance(m, QuantizedLinear):
                assert m.act_fq is None, "act_fq should be None in weight-only mode"

    def test_output_shape_int4(self):
        model = _make_mlp()
        pipeline = PTQPipeline(num_bits=4, weight_only=True, per_channel=True)
        model_q = _run(pipeline, model)
        x = torch.randn(1, 4, 64)
        with torch.no_grad():
            assert model_q(x).shape == x.shape


class TestFromConfig:
    def test_from_config_int8_per_channel(self):
        config = {
            "quantization": {"enabled": True, "dtype": "int8", "symmetric": True, "weight_only": False},
            "calibration": {"method": "minmax", "num_samples": 8, "percentile": None},
            "granularity": "per_channel",
        }
        pipeline = PTQPipeline.from_config(config)
        assert pipeline.num_bits == 8
        assert pipeline.per_channel is True
        assert pipeline.symmetric is True

    def test_from_config_fp16_disabled(self):
        config = {
            "quantization": {"enabled": False, "dtype": "fp16"},
            "calibration": {"method": "none", "num_samples": 0},
            "granularity": "none",
        }
        pipeline = PTQPipeline.from_config(config)
        assert pipeline.enabled is False
