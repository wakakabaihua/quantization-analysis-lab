#!/usr/bin/env python3
"""
Summarize all experiment results into comparison tables.

Reads JSON result files from --results_dir and produces:
  - results/tables/summary.csv           Full metric table
  - results/tables/summary_ranked.csv    Sorted by cosine similarity (desc)
  - results/tables/delta_vs_baseline.csv Error delta relative to FP16 baseline
  - results/tables/memory_footprint.csv  Estimated weight memory by dtype
  - results/tables/sensitivity_*.csv     Layer-wise sensitivity tables

Also prints a formatted console summary.

Usage:
    uv run python scripts/summarize_results.py
    uv run python scripts/summarize_results.py --results_dir results/raw --output results/tables/summary.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_glob(results_dir: str, pattern: str) -> pd.DataFrame:
    records = []
    for path in sorted(Path(results_dir).glob(pattern)):
        with open(path) as f:
            data = json.load(f)
        records.extend(data)
    return pd.DataFrame(records) if records else pd.DataFrame()


def load_experiment_results(results_dir: str) -> pd.DataFrame:
    """Load main PTQ experiment results (excludes sensitivity files)."""
    records = []
    skip_prefixes = ("sensitivity_",)
    for path in sorted(Path(results_dir).glob("*.json")):
        if any(path.name.startswith(p) for p in skip_prefixes):
            continue
        with open(path) as f:
            data = json.load(f)
        records.extend(data)
    return pd.DataFrame(records) if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# Memory footprint estimation
# ---------------------------------------------------------------------------

_BITS = {
    "fp16": 16, "fp32": 32,
    "int8": 8, "int4": 4,
}

_CONFIG_DTYPE_MAP = {
    "fp16_baseline":       ("fp16",  False),
    "int8_per_tensor_sym": ("int8",  False),
    "int8_per_channel_sym":("int8",  False),
    "int8_per_tensor_asym":("int8",  False),
    "int8_per_channel_asym":("int8", False),
    "int4_weight_only":    ("int4",  True),   # weight_only
}

# MLP: fc1(512→2048) + fc2(2048→512) in float32 = (512*2048 + 2048*512)*4 bytes
_MODEL_FP32_BYTES = {
    "mlp":       (512 * 2048 + 2048 * 512) * 4,
    "attention": (512 * 512 * 4) * 4,          # q+k+v+o projections
}


def compute_memory_footprint(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.drop_duplicates(["config", "model"]).iterrows():
        config = row["config"]
        model = row["model"]
        dtype, weight_only = _CONFIG_DTYPE_MAP.get(config, ("fp32", False))
        bits = _BITS.get(dtype, 32)
        fp32_bytes = _MODEL_FP32_BYTES.get(model, 0)
        q_bytes = fp32_bytes * bits / 32 if fp32_bytes else 0
        reduction = fp32_bytes / q_bytes if q_bytes > 0 else 1.0
        rows.append({
            "config": config,
            "model": model,
            "dtype": dtype,
            "weight_only": weight_only,
            "fp32_weight_mb": round(fp32_bytes / 1e6, 3),
            "quantized_weight_mb": round(q_bytes / 1e6, 3),
            "memory_reduction_x": round(reduction, 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Delta vs. FP16 baseline
# ---------------------------------------------------------------------------

def compute_delta_vs_baseline(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "cosine_similarity", "max_absolute_error",
        "mean_absolute_error", "root_mean_squared_error",
    ]
    present = [c for c in metric_cols if c in df.columns]

    baseline = df[df["config"] == "fp16_baseline"].copy()
    if baseline.empty:
        return pd.DataFrame()

    baseline = baseline.set_index("model")[present].rename(
        columns={c: f"baseline_{c}" for c in present}
    )
    merged = df[df["config"] != "fp16_baseline"].copy()
    merged = merged.join(baseline, on="model")

    for c in present:
        bc = f"baseline_{c}"
        if bc in merged.columns:
            merged[f"delta_{c}"] = merged[c] - merged[bc]

    drop_cols = [f"baseline_{c}" for c in present if f"baseline_{c}" in merged.columns]
    return merged.drop(columns=drop_cols)


# ---------------------------------------------------------------------------
# Sensitivity tables
# ---------------------------------------------------------------------------

def summarize_sensitivity(results_dir: str, output_dir: Path) -> None:
    # Layer-wise
    lw = _load_glob(results_dir, "sensitivity_layerwise.json")
    if not lw.empty:
        cols = [c for c in ["model","layer","cosine_similarity","max_absolute_error",
                             "mean_absolute_error","root_mean_squared_error"] if c in lw.columns]
        lw_out = lw[cols].sort_values(["model", "cosine_similarity"])
        path = output_dir / "sensitivity_layerwise.csv"
        lw_out.to_csv(path, index=False)
        print(f"\nLayer-wise sensitivity (lowest cosine_similarity = most fragile):")
        print(lw_out.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Weight-only vs full
    wf = _load_glob(results_dir, "sensitivity_weightonly_vs_full.json")
    if not wf.empty:
        cols = [c for c in ["model","layer","mode","cosine_similarity","mean_absolute_error"] if c in wf.columns]
        wf_out = wf[cols].sort_values(["model","layer","mode"])
        path = output_dir / "sensitivity_weightonly_vs_full.csv"
        wf_out.to_csv(path, index=False)

    # Calibration strategy
    cal = _load_glob(results_dir, "sensitivity_calibration.json")
    if not cal.empty:
        cols = [c for c in ["model","calibration","cosine_similarity","mean_absolute_error"] if c in cal.columns]
        cal_out = cal[cols].sort_values(["model","calibration"])
        path = output_dir / "sensitivity_calibration.csv"
        cal_out.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_KEEP_COLS = [
    "config", "model",
    "cosine_similarity", "max_absolute_error",
    "mean_absolute_error", "root_mean_squared_error",
    "relative_error",
    "layer_mean_cosine_similarity",
    "layer_mean_mean_absolute_error",
]

_CONFIG_ORDER = [
    "fp16_baseline",
    "int8_per_tensor_sym",
    "int8_per_channel_sym",
    "int8_per_tensor_asym",
    "int8_per_channel_asym",
    "int4_weight_only",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment results")
    parser.add_argument("--results_dir", default="results/raw")
    parser.add_argument("--output", default="results/tables/summary.csv")
    args = parser.parse_args()

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_experiment_results(args.results_dir)
    if df.empty:
        print("No results found — run experiments first.")
        return

    # Canonical config ordering
    df["_order"] = df["config"].map(
        {c: i for i, c in enumerate(_CONFIG_ORDER)}
    ).fillna(99)
    df = df.sort_values(["model", "_order"]).drop(columns=["_order"]).reset_index(drop=True)

    # 1. Full summary
    present = [c for c in _KEEP_COLS if c in df.columns]
    summary = df[present].copy()
    summary.to_csv(args.output, index=False)

    # 2. Ranked by cosine similarity
    ranked = summary.sort_values(
        ["model", "cosine_similarity"], ascending=[True, False]
    ).reset_index(drop=True)
    ranked.to_csv(output_dir / "summary_ranked.csv", index=False)

    # 3. Delta vs. FP16 baseline
    delta = compute_delta_vs_baseline(df)
    if not delta.empty:
        delta_cols = (
            ["config", "model"]
            + [c for c in present if c not in ("config", "model")]
            + [c for c in delta.columns if c.startswith("delta_")]
        )
        delta_present = [c for c in delta_cols if c in delta.columns]
        delta[delta_present].to_csv(output_dir / "delta_vs_baseline.csv", index=False)

    # 4. Memory footprint
    mem = compute_memory_footprint(df)
    if not mem.empty:
        mem.to_csv(output_dir / "memory_footprint.csv", index=False)

    # 5. Sensitivity tables
    summarize_sensitivity(args.results_dir, output_dir)

    # Console output
    print(f"\nSummary saved to {args.output}")
    print("=" * 80)
    for model_name, group in summary.groupby("model"):
        print(f"\nModel: {model_name}")
        print(group.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if not mem.empty:
        print("\n--- Memory Footprint ---")
        print(mem.to_string(index=False))
    print()


if __name__ == "__main__":
    main()
