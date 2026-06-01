#!/usr/bin/env python3
"""
Mixed-Precision Quantization Search.

Reads per-layer sensitivity scores from ``results/raw/sensitivity_layerwise.json``
(produced by ``scripts/sensitivity_analysis.py``) and proposes an optimal
per-layer bit-width assignment that balances compression and accuracy.

The script performs three steps:

1. **Sensitivity table** — Print per-layer cosine similarity when each layer
   is quantized alone at INT8.  Lower scores → higher sensitivity.

2. **Threshold sweep** — For thresholds T ∈ {0.9999, 0.99995, 0.99998, 1.0},
   show how many layers would be assigned INT8 vs. INT4 and the estimated
   compression ratio of the resulting mixed-precision model.

3. **Evaluation** — For the default threshold (0.9999), evaluate:
     a. Full INT8 (all layers at 8-bit).
     b. Full INT4 (all layers at 4-bit).
     c. Mixed precision (best assignment per the threshold sweep).
   Print cosine similarity and compression ratio for each.

Usage::

    python scripts/mixed_precision_search.py
    python scripts/mixed_precision_search.py --sensitivity results/raw/sensitivity_layerwise.json
    python scripts/mixed_precision_search.py --model mlp --threshold 0.9999
    python scripts/mixed_precision_search.py --output_dir results/raw

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.attention_block import AttentionBlock
from src.models.mlp_block import MLPBlock
from src.quant.backends.mixed_precision_backend import MixedPrecisionBackend
from src.quant.backends.fake_quant_backend import FakeQuantBackend
from src.quant.error_analysis import compute_output_error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model(model_name: str) -> torch.nn.Module:
    if model_name == "mlp":
        return MLPBlock(input_dim=512, hidden_dim=2048).eval()
    if model_name == "attention":
        return AttentionBlock(embed_dim=512, num_heads=8).eval()
    raise ValueError(f"Unknown model: {model_name!r}")


def _calib_inputs(model_name: str, n: int = 32) -> list:
    torch.manual_seed(42)
    if model_name == "mlp":
        return [torch.randn(8, 32, 512) for _ in range(n)]
    return [torch.randn(4, 16, 512) for _ in range(n)]


def _test_input(model_name: str) -> torch.Tensor:
    torch.manual_seed(0)
    if model_name == "mlp":
        return torch.randn(4, 16, 512)
    return torch.randn(2, 8, 512)


def _forward(model_name: str, model, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


def _calibrate(backend, model_name: str, model, n: int = 32) -> None:
    calib_inputs = _calib_inputs(model_name, n)

    def run_calib() -> None:
        for x in calib_inputs:
            _forward(model_name, model, x)

    backend.calibrate(model, run_calib)


def _compression_label(ratio: float) -> str:
    return f"{ratio:.2f}×"


# ---------------------------------------------------------------------------
# Sensitivity table
# ---------------------------------------------------------------------------

def load_sensitivity(path: Path) -> Dict[str, Dict[str, float]]:
    """
    Load sensitivity_layerwise.json.

    Returns:
        {model_name: {layer_name: cosine_similarity}}
    """
    with open(path) as f:
        records = json.load(f)

    result: Dict[str, Dict[str, float]] = {}
    for rec in records:
        model = rec.get("model", "unknown")
        layer = rec.get("layer", "unknown")
        sim = rec.get("cosine_similarity", 1.0)
        result.setdefault(model, {})[layer] = sim
    return result


def print_sensitivity_table(scores_by_model: Dict[str, Dict[str, float]]) -> None:
    print("\n" + "=" * 60)
    print("  Layer Sensitivity (cosine_similarity when quantized alone)")
    print("  Lower score = higher sensitivity = needs more bits")
    print("=" * 60)
    for model_name, scores in sorted(scores_by_model.items()):
        print(f"\n  Model: {model_name}")
        print(f"  {'Layer':<18} {'Cosine Sim':>12}  {'Assignment hint':>20}")
        print(f"  {'-'*18}  {'-'*12}  {'-'*20}")
        for layer, sim in sorted(scores.items(), key=lambda kv: kv[1]):
            hint = "sensitive → INT8" if sim < 0.9999 else "robust   → INT4"
            print(f"  {layer:<18}  {sim:>12.8f}  {hint:>20}")


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def threshold_sweep(
    scores_by_model: Dict[str, Dict[str, float]],
    model_name: str,
    thresholds: List[float],
    high_bits: int = 8,
    low_bits: int = 4,
) -> None:
    print(f"\n{'='*60}")
    print(f"  Threshold sweep — model: {model_name}")
    print(f"  (INT{high_bits} for sensitive layers, INT{low_bits} for robust)")
    print(f"{'='*60}")
    print(f"  {'Threshold':>10}  {'INT8 layers':>12}  {'INT4 layers':>12}  {'Avg bits':>10}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*10}")

    scores = scores_by_model.get(model_name, {})
    if not scores:
        print(f"  (no sensitivity data for model '{model_name}')")
        return

    for threshold in thresholds:
        n_high = sum(1 for s in scores.values() if s < threshold)
        n_low = len(scores) - n_high
        total = len(scores)
        avg_bits = (n_high * high_bits + n_low * low_bits) / total if total else 0
        print(f"  {threshold:>10.6f}  {n_high:>12}  {n_low:>12}  {avg_bits:>10.2f}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_backends(
    model_name: str,
    scores: Dict[str, float],
    threshold: float,
    high_bits: int = 8,
    low_bits: int = 4,
    symmetric: bool = True,
    per_channel: bool = True,
) -> None:
    print(f"\n{'='*60}")
    print(f"  Evaluation — model: {model_name}  (threshold={threshold})")
    print(f"{'='*60}")

    model = _build_model(model_name)
    x_test = _test_input(model_name)

    with torch.no_grad():
        ref = _forward(model_name, model, x_test)

    def _run(backend) -> dict:
        _calibrate(backend, model_name, model)
        model_q = backend.convert(model).eval()
        with torch.no_grad():
            q_out = _forward(model_name, model_q, x_test)
        return compute_output_error(ref, q_out)

    configs = [
        (f"INT{high_bits} (all layers)",
         FakeQuantBackend(num_bits=high_bits, symmetric=symmetric,
                          per_channel=per_channel, weight_only=True)),
        (f"INT{low_bits} (all layers)",
         FakeQuantBackend(num_bits=low_bits, symmetric=symmetric,
                          per_channel=per_channel, weight_only=True)),
        (f"Mixed (T={threshold})",
         MixedPrecisionBackend.from_sensitivity(
             scores, threshold=threshold,
             high_bits=high_bits, low_bits=low_bits,
             symmetric=symmetric, per_channel=per_channel, weight_only=True
         )),
    ]

    print(f"\n  {'Config':<22}  {'Cosine Sim':>12}  {'MAE':>10}  {'Compression':>12}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*10}  {'-'*12}")

    for label, backend in configs:
        errors = _run(backend)
        sim = errors["cosine_similarity"]
        mae = errors["mean_absolute_error"]
        comp_info = backend.theoretical_compression(model) if hasattr(backend, "theoretical_compression") else None
        if comp_info:
            ratio = comp_info["compression_ratio"]
            comp_label = _compression_label(ratio) + " vs FP32"
        else:
            # FakeQuantBackend: infer from num_bits
            nb = backend.num_bits
            comp_label = _compression_label(32 / nb) + " vs FP32"
        print(f"  {label:<22}  {sim:>12.8f}  {mae:>10.6f}  {comp_label:>12}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Mixed-precision quantization search")
    parser.add_argument(
        "--sensitivity",
        default="results/raw/sensitivity_layerwise.json",
        help="Path to sensitivity_layerwise.json",
    )
    parser.add_argument(
        "--model",
        choices=["mlp", "attention", "both"],
        default="both",
        help="Which model to evaluate (default: both)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9999,
        help="Cosine-similarity threshold for INT8 vs INT4 assignment (default: 0.9999)",
    )
    parser.add_argument(
        "--high_bits", type=int, default=8,
        help="Bits for sensitive layers (default: 8)"
    )
    parser.add_argument(
        "--low_bits", type=int, default=4,
        help="Bits for robust layers (default: 4)"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="If set, save the proposed mixed-precision config as JSON to this directory",
    )
    args = parser.parse_args()

    sensitivity_path = Path(args.sensitivity)
    if not sensitivity_path.exists():
        print(
            f"[warn] Sensitivity file not found: {sensitivity_path}\n"
            "       Run `python scripts/sensitivity_analysis.py` first to generate it.\n"
            "       Proceeding with synthetic scores for demonstration.\n"
        )
        # Fallback demo scores so the script is still useful on a fresh clone
        scores_by_model = {
            "mlp": {"fc1": 0.99979, "fc2": 0.99973},
            "attention": {
                "q_proj": 0.99999, "k_proj": 0.99999,
                "v_proj": 0.99999, "out_proj": 0.99997,
            },
        }
    else:
        scores_by_model = load_sensitivity(sensitivity_path)

    print_sensitivity_table(scores_by_model)

    models_to_eval = (
        [args.model] if args.model != "both" else ["mlp", "attention"]
    )
    thresholds = [0.9998, 0.9999, 0.99995, 0.99998, 1.0]

    for model_name in models_to_eval:
        threshold_sweep(scores_by_model, model_name, thresholds,
                        args.high_bits, args.low_bits)

    for model_name in models_to_eval:
        scores = scores_by_model.get(model_name, {})
        if not scores:
            print(f"\n[skip] No sensitivity data for model '{model_name}'")
            continue
        evaluate_backends(
            model_name, scores,
            threshold=args.threshold,
            high_bits=args.high_bits,
            low_bits=args.low_bits,
        )

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for model_name in models_to_eval:
            scores = scores_by_model.get(model_name, {})
            if not scores:
                continue
            layer_config = {
                name: {"num_bits": args.high_bits if s < args.threshold else args.low_bits}
                for name, s in scores.items()
            }
            out_path = out_dir / f"mixed_precision_{model_name}.json"
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "model": model_name,
                        "threshold": args.threshold,
                        "high_bits": args.high_bits,
                        "low_bits": args.low_bits,
                        "layer_config": layer_config,
                    },
                    f,
                    indent=2,
                )
            print(f"\n[saved] {out_path}")

    print()


if __name__ == "__main__":
    main()
