#!/usr/bin/env python3
"""
Layer-wise sensitivity analysis.

Answers: which individual layers are most sensitive to quantization?

Strategy — for each Linear layer in the model:
  1. Quantize ONLY that one layer (INT8 per-channel, the best single config).
  2. Run the model on the test input.
  3. Measure output cosine similarity and MAE vs. full-float baseline.

Additionally runs two cross-cutting experiments:
  A. Weight-only vs. full (weight+activation) quantization for all layers.
  B. Sensitivity to calibration strategy: minmax vs. percentile on all layers.

Results are saved to:
  results/raw/sensitivity_layerwise.json
  results/raw/sensitivity_weightonly_vs_full.json
  results/raw/sensitivity_calibration.json

Usage:
    uv run python scripts/sensitivity_analysis.py
    uv run python scripts/sensitivity_analysis.py --output_dir results/raw --device cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.attention_block import AttentionBlock
from src.models.mlp_block import MLPBlock
from src.quant.error_analysis import compute_output_error
from src.quant.fake_quant import FakeQuantize
from src.quant.observers import MinMaxObserver, HistogramObserver
from src.quant.calibrators import MinMaxCalibrator, PercentileCalibrator
from src.quant.ptq_pipeline import PTQPipeline, QuantizedLinear, _set_module


# ---------------------------------------------------------------------------
# Helpers shared with run_experiments.py
# ---------------------------------------------------------------------------

def _forward(model_name: str, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


def _make_models(device: torch.device) -> dict[str, nn.Module]:
    return {
        "mlp": MLPBlock(input_dim=512, hidden_dim=2048).eval().to(device),
        "attention": AttentionBlock(embed_dim=512, num_heads=8).eval().to(device),
    }


def _calib_inputs(model_name: str, device: torch.device, n: int = 32) -> list:
    torch.manual_seed(42)
    if model_name == "mlp":
        return [torch.randn(8, 32, 512, device=device) for _ in range(n)]
    return [torch.randn(4, 16, 512, device=device) for _ in range(n)]


def _test_input(model_name: str, device: torch.device) -> torch.Tensor:
    torch.manual_seed(0)
    if model_name == "mlp":
        return torch.randn(4, 16, 512, device=device)
    return torch.randn(2, 8, 512, device=device)


# ---------------------------------------------------------------------------
# Single-layer quantization helper
# ---------------------------------------------------------------------------

def _quantize_single_layer(
    model: nn.Module,
    layer_name: str,
    calib_inputs: list[torch.Tensor],
    model_name: str,
    num_bits: int = 8,
    symmetric: bool = True,
    weight_only: bool = False,
    calibration_method: str = "minmax",
    percentile: float = 99.99,
) -> nn.Module:
    """Return a copy of model with exactly one Linear layer quantized."""
    model_q = copy.deepcopy(model)

    # Find the target module in the copy
    target: nn.Linear | None = None
    for n, m in model_q.named_modules():
        if n == layer_name and isinstance(m, nn.Linear):
            target = m
            break
    if target is None:
        raise ValueError(f"Layer {layer_name!r} not found or not Linear")

    # Build weight observer & calibrator
    if calibration_method == "percentile":
        w_obs = HistogramObserver(per_channel=True)
        w_cal = PercentileCalibrator(num_bits=num_bits, symmetric=symmetric, percentile=percentile)
    else:
        w_obs = MinMaxObserver(per_channel=True)
        w_cal = MinMaxCalibrator(num_bits=num_bits, symmetric=symmetric)

    w_obs.update(target.weight.detach())
    w_scale, w_zp = w_cal.compute(w_obs.stats)

    weight_fq = FakeQuantize(
        num_bits=num_bits, symmetric=symmetric, per_channel=True, channel_axis=0
    )
    weight_fq.set_qparams(w_scale, w_zp)

    # Activation FQ (weight_only → skip)
    act_fq = None
    if not weight_only:
        if calibration_method == "percentile":
            a_obs = HistogramObserver(per_channel=False)
            a_cal = PercentileCalibrator(num_bits=num_bits, symmetric=symmetric, percentile=percentile)
        else:
            a_obs = MinMaxObserver(per_channel=False)
            a_cal = MinMaxCalibrator(num_bits=num_bits, symmetric=symmetric)

        with torch.no_grad():
            for x in calib_inputs:
                # Forward until we reach the target layer's input
                _collect_act(model_name, model, x, layer_name, a_obs)

        a_scale, a_zp = a_cal.compute(a_obs.stats)
        act_fq = FakeQuantize(num_bits=num_bits, symmetric=symmetric, per_channel=False)
        act_fq.set_qparams(a_scale, a_zp)

    q_layer = QuantizedLinear(target, weight_fq, act_fq)
    _set_module(model_q, layer_name, q_layer)
    return model_q.eval()


def _collect_act(
    model_name: str,
    model: nn.Module,
    x: torch.Tensor,
    layer_name: str,
    observer,
) -> None:
    """Hook-based activation collection for a single named layer."""
    handles = []

    def hook(mod, inp, out):
        observer.update(inp[0].detach())

    for n, m in model.named_modules():
        if n == layer_name:
            handles.append(m.register_forward_hook(hook))
            break

    with torch.no_grad():
        out = model(x)

    for h in handles:
        h.remove()


# ---------------------------------------------------------------------------
# Experiment A: layer-by-layer sensitivity
# ---------------------------------------------------------------------------

def run_layerwise_sensitivity(
    models: dict[str, nn.Module],
    device: torch.device,
    num_bits: int = 8,
) -> list[dict]:
    records = []
    for model_name, model in models.items():
        print(f"  [sensitivity-layerwise] {model_name}")
        calib = _calib_inputs(model_name, device)
        x_test = _test_input(model_name, device)

        with torch.no_grad():
            fp_out = _forward(model_name, model, x_test)

        linear_layers = [
            n for n, m in model.named_modules() if isinstance(m, nn.Linear)
        ]

        for layer_name in linear_layers:
            model_q = _quantize_single_layer(
                model, layer_name, calib, model_name,
                num_bits=num_bits, symmetric=True, weight_only=False,
                calibration_method="minmax",
            )
            with torch.no_grad():
                q_out = _forward(model_name, model_q, x_test)

            metrics = compute_output_error(fp_out, q_out)
            records.append({
                "experiment": "sensitivity_layerwise",
                "model": model_name,
                "layer": layer_name,
                "num_bits": num_bits,
                "weight_only": False,
                "calibration": "minmax",
                **metrics,
            })
            sim = metrics["cosine_similarity"]
            print(f"    {layer_name:<20} cosine_sim={sim:.6f}  MAE={metrics['mean_absolute_error']:.6f}")

    return records


# ---------------------------------------------------------------------------
# Experiment B: weight-only vs. full quantization per layer
# ---------------------------------------------------------------------------

def run_weightonly_vs_full(
    models: dict[str, nn.Module],
    device: torch.device,
) -> list[dict]:
    records = []
    for model_name, model in models.items():
        print(f"  [sensitivity-weight_vs_full] {model_name}")
        calib = _calib_inputs(model_name, device)
        x_test = _test_input(model_name, device)

        with torch.no_grad():
            fp_out = _forward(model_name, model, x_test)

        linear_layers = [
            n for n, m in model.named_modules() if isinstance(m, nn.Linear)
        ]

        for layer_name in linear_layers:
            for weight_only, label in [(True, "weight_only"), (False, "weight+act")]:
                model_q = _quantize_single_layer(
                    model, layer_name, calib, model_name,
                    num_bits=8, symmetric=True, weight_only=weight_only,
                    calibration_method="minmax",
                )
                with torch.no_grad():
                    q_out = _forward(model_name, model_q, x_test)

                metrics = compute_output_error(fp_out, q_out)
                records.append({
                    "experiment": "sensitivity_weightonly_vs_full",
                    "model": model_name,
                    "layer": layer_name,
                    "mode": label,
                    **metrics,
                })
    return records


# ---------------------------------------------------------------------------
# Experiment C: calibration strategy sensitivity
# ---------------------------------------------------------------------------

def run_calibration_sensitivity(
    models: dict[str, nn.Module],
    device: torch.device,
) -> list[dict]:
    records = []
    for model_name, model in models.items():
        print(f"  [sensitivity-calibration] {model_name}")
        calib = _calib_inputs(model_name, device)
        x_test = _test_input(model_name, device)

        with torch.no_grad():
            fp_out = _forward(model_name, model, x_test)

        # Full-model quantization under minmax vs percentile (all layers at once)
        for cal_method, label in [("minmax", "minmax"), ("percentile", "percentile_99.99")]:
            pipeline = PTQPipeline(
                num_bits=8, symmetric=True, per_channel=True, weight_only=False,
                calibration_method=cal_method, calibration_percentile=99.99,
            )

            def _run():
                for x in calib:
                    _forward(model_name, model, x)

            pipeline.calibrate(model, _run)
            model_q = pipeline.quantize(model).eval()

            with torch.no_grad():
                q_out = _forward(model_name, model_q, x_test)

            metrics = compute_output_error(fp_out, q_out)
            records.append({
                "experiment": "sensitivity_calibration",
                "model": model_name,
                "calibration": label,
                "granularity": "per_channel",
                **metrics,
            })
            print(f"    {label:<20} cosine_sim={metrics['cosine_similarity']:.6f}")

    return records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Layer-wise sensitivity analysis")
    parser.add_argument("--output_dir", default="results/raw")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"\nSensitivity Analysis  (device={args.device})")
    print("=" * 60)

    models = _make_models(device)

    t0 = time.perf_counter()

    records_lw = run_layerwise_sensitivity(models, device)
    records_wf = run_weightonly_vs_full(models, device)
    records_cal = run_calibration_sensitivity(models, device)

    # Save
    for fname, data in [
        ("sensitivity_layerwise.json", records_lw),
        ("sensitivity_weightonly_vs_full.json", records_wf),
        ("sensitivity_calibration.json", records_cal),
    ]:
        path = output_dir / fname
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n  Saved → {path}")

    print(f"\nDone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
