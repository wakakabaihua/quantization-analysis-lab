#!/usr/bin/env python3
"""
Run a single quantization experiment from a YAML config file.

For each model layer type specified in the config (mlp, attention):
  1. Build a baseline float model.
  2. Calibrate the PTQ pipeline on synthetic data.
  3. Produce a quantized copy of the model.
  4. Measure output-level and layer-wise quantization error.
  5. Save results as JSON to --output_dir.

Usage:
    python scripts/run_experiments.py --config configs/ptq_int8_per_tensor.yaml
    python scripts/run_experiments.py --config configs/ptq_fp16.yaml --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.attention_block import AttentionBlock
from src.models.mlp_block import MLPBlock
from src.quant.error_analysis import (
    LayerwiseErrorTracker,
    compute_layerwise_errors,
    compute_output_error,
)
from src.quant.ptq_pipeline import PTQPipeline, QuantizedLinear
from src.utils.logging import ExperimentLogger
from src.utils.metrics import aggregate_errors


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _build_models(config: dict, device: torch.device) -> dict:
    layers = config.get("experiment", {}).get("layers", ["mlp", "attention"])
    models = {}
    if "mlp" in layers:
        models["mlp"] = MLPBlock(input_dim=512, hidden_dim=2048).eval().to(device)
    if "attention" in layers:
        models["attention"] = AttentionBlock(embed_dim=512, num_heads=8).eval().to(device)
    return models


def _calib_inputs(model_name: str, device: torch.device, n: int) -> list:
    """Synthetic calibration inputs with fixed seed for reproducibility."""
    torch.manual_seed(42)
    if model_name == "mlp":
        return [torch.randn(8, 32, 512, device=device) for _ in range(n)]
    if model_name == "attention":
        return [torch.randn(4, 16, 512, device=device) for _ in range(n)]
    raise ValueError(f"Unknown model name: {model_name!r}")


def _test_input(model_name: str, device: torch.device) -> torch.Tensor:
    torch.manual_seed(0)
    if model_name == "mlp":
        return torch.randn(4, 16, 512, device=device)
    if model_name == "attention":
        return torch.randn(2, 8, 512, device=device)
    raise ValueError(f"Unknown model name: {model_name!r}")


# ---------------------------------------------------------------------------
# Forward helpers that handle tuple returns from AttentionBlock
# ---------------------------------------------------------------------------

def _forward(model_name: str, model, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    return out


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(config: dict, output_dir: str, device: torch.device) -> None:
    config_name = config.get("name", "unnamed")
    n_calib = int(config.get("calibration", {}).get("num_samples", 32))

    logger = ExperimentLogger(output_dir, config_name)
    pipeline = PTQPipeline.from_config(config)
    models = _build_models(config, device)

    for model_name, model in models.items():
        print(f"  [{config_name}] {model_name} ...", flush=True)

        # --- Calibration ---
        calib_inputs = _calib_inputs(model_name, device, n_calib)

        def run_calib() -> None:
            for x in calib_inputs:
                _forward(model_name, model, x)

        pipeline.calibrate(model, run_calib)

        # --- Quantize ---
        model_q = pipeline.quantize(model)
        model_q.eval()

        # --- Output-level error ---
        x_test = _test_input(model_name, device)
        with torch.no_grad():
            fp_out = _forward(model_name, model, x_test)
            q_out = _forward(model_name, model_q, x_test)

        output_metrics = compute_output_error(fp_out, q_out)

        # --- Layer-wise error ---
        # Baseline tracks nn.Linear; quantized model tracks QuantizedLinear
        bt = LayerwiseErrorTracker(model, target_types=(torch.nn.Linear,))
        qt = LayerwiseErrorTracker(model_q, target_types=(QuantizedLinear,))
        with bt, qt:
            with torch.no_grad():
                _forward(model_name, model, x_test)
                _forward(model_name, model_q, x_test)

        layerwise = compute_layerwise_errors(bt, qt)
        layer_summary = aggregate_errors(layerwise)

        # --- Log ---
        record = {
            "model": model_name,
            "config": config_name,
            **output_metrics,
            **{f"layer_{k}": v for k, v in layer_summary.items()},
        }
        logger.log(record)

        for k, v in output_metrics.items():
            print(f"    {k}: {v:.6f}")

    logger.save()
    print(f"  Saved → {output_dir}/{config_name}.json\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a PTQ experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output_dir", default="results/raw")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Experiment : {config.get('name', args.config)}")
    print(f"Description: {config.get('description', '').strip()}")
    print(f"Device     : {args.device}\n")

    run_experiment(config, args.output_dir, torch.device(args.device))


if __name__ == "__main__":
    main()
