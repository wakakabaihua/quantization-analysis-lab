#!/usr/bin/env python3
"""
Evaluate quantization accuracy across all saved experiment results.

Reads JSON result files from a results directory and prints a formatted
accuracy comparison table grouped by model type.

Usage:
    python scripts/evaluate_accuracy.py --results_dir results/raw
    python scripts/evaluate_accuracy.py --results_dir results/raw --metric cosine_similarity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def load_results(results_dir: str) -> pd.DataFrame:
    records = []
    for path in sorted(Path(results_dir).glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        records.extend(data)
    return pd.DataFrame(records) if records else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate quantization accuracy")
    parser.add_argument("--results_dir", default="results/raw")
    parser.add_argument(
        "--metric",
        default="cosine_similarity",
        help="Primary metric to highlight",
    )
    args = parser.parse_args()

    df = load_results(args.results_dir)
    if df.empty:
        print("No results found in", args.results_dir, "— run experiments first.")
        sys.exit(0)

    display_metrics = [
        "cosine_similarity",
        "max_absolute_error",
        "mean_absolute_error",
        "root_mean_squared_error",
        "relative_error",
    ]
    available = [m for m in display_metrics if m in df.columns]

    print(f"\nQuantization Accuracy Evaluation — {args.results_dir}")
    print("=" * 80)

    for model_name, group in df.groupby("model"):
        print(f"\nModel: {model_name}")
        cols = ["config"] + available
        present = [c for c in cols if c in group.columns]
        display = group[present].set_index("config").sort_index()

        # Highlight the primary metric column
        if args.metric in display.columns:
            print(f"  (ranked by {args.metric})")
            display = display.sort_values(args.metric, ascending=False)

        print(display.to_string(float_format=lambda x: f"{x:.6f}"))

    print()


if __name__ == "__main__":
    main()
