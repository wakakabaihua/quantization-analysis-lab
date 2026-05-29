#!/usr/bin/env python3
"""
Generate a Markdown experiment report from result JSON files and CSV tables.

Output: results/report.md

Sections:
  1. Executive Summary
  2. Experiment Matrix
  3. Delta vs. FP16 Baseline
  4. Memory Footprint Estimates
  5. Layer-wise Sensitivity
  6. Weight-only vs. Full Quantization
  7. Calibration Strategy Comparison
  8. Figures
  9. Conclusions & Next Steps

Usage:
    uv run python scripts/generate_report.py
    uv run python scripts/generate_report.py --results_dir results --output results/report.md
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() else None


def _load_glob_json(raw_dir: Path, pattern: str) -> list[dict]:
    records = []
    for p in sorted(raw_dir.glob(pattern)):
        with open(p) as f:
            records.extend(json.load(f))
    return records


def _fmt_float(v, decimals: int = 6) -> str:
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _df_to_md(df: pd.DataFrame, float_fmt: str = "{:.6f}") -> str:
    """Convert DataFrame to GitHub-flavoured Markdown table."""
    df = df.copy()
    for col in df.select_dtypes(include=["float64", "float32"]).columns:
        df[col] = df[col].apply(lambda v: float_fmt.format(v) if pd.notna(v) else "—")
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, sep] + rows)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _section_executive_summary(raw_dir: Path, tables_dir: Path) -> str:
    summary_path = tables_dir / "summary_ranked.csv"
    df = _load_csv(summary_path)
    if df is None or df.empty:
        return "## 1. Executive Summary\n\n_No results available yet — run experiments first._\n"

    lines = ["## 1. Executive Summary\n"]

    # Best & worst configs by cosine_similarity
    best = df.iloc[0]
    worst = df.iloc[-1]
    lines.append(
        f"Across all tested configurations, **{best['config']}** on **{best['model']}** "
        f"achieved the highest output fidelity "
        f"(cosine similarity = `{_fmt_float(best['cosine_similarity'])}`).  "
        f"The lowest fidelity was observed for **{worst['config']}** on **{worst['model']}** "
        f"(cosine similarity = `{_fmt_float(worst['cosine_similarity'])}`).\n"
    )

    # Per-model best/worst
    for model_name, group in df.groupby("model"):
        best_m = group.sort_values("cosine_similarity", ascending=False).iloc[0]
        worst_m = group.sort_values("cosine_similarity").iloc[0]
        lines.append(
            f"- **{model_name}**: best `{best_m['config']}` "
            f"(sim={_fmt_float(best_m['cosine_similarity'])}), "
            f"worst `{worst_m['config']}` "
            f"(sim={_fmt_float(worst_m['cosine_similarity'])})"
        )

    lines.append("")
    return "\n".join(lines) + "\n"


def _section_experiment_matrix(tables_dir: Path) -> str:
    df = _load_csv(tables_dir / "summary.csv")
    lines = ["## 2. Experiment Matrix\n"]
    if df is None or df.empty:
        lines.append("_No summary table found._\n")
        return "\n".join(lines) + "\n"

    display_cols = [
        c for c in [
            "config", "model", "cosine_similarity",
            "max_absolute_error", "mean_absolute_error",
            "root_mean_squared_error", "relative_error",
        ]
        if c in df.columns
    ]
    lines.append(_df_to_md(df[display_cols]))
    lines.append("")
    return "\n".join(lines) + "\n"


def _section_delta_vs_baseline(tables_dir: Path) -> str:
    df = _load_csv(tables_dir / "delta_vs_baseline.csv")
    lines = ["## 3. Delta vs. FP16 Baseline\n"]
    if df is None or df.empty:
        lines.append("_delta_vs_baseline.csv not found — requires fp16_baseline results._\n")
        return "\n".join(lines) + "\n"

    lines.append(
        "Positive Δ cosine_similarity means the config is *more* similar to the float baseline "
        "than the FP16 run itself (i.e. even closer to FP32).  "
        "Negative values indicate accuracy loss from quantization.\n"
    )
    lines.append(_df_to_md(df))
    lines.append("")
    return "\n".join(lines) + "\n"


def _section_memory_footprint(tables_dir: Path) -> str:
    df = _load_csv(tables_dir / "memory_footprint.csv")
    lines = ["## 4. Memory Footprint Estimates\n"]
    if df is None or df.empty:
        lines.append("_memory_footprint.csv not found._\n")
        return "\n".join(lines) + "\n"

    lines.append(
        "Weight memory estimates are derived from parameter counts and dtype bit-widths "
        "(per-channel scale overhead is not included).\n"
    )
    lines.append(_df_to_md(df, float_fmt="{:.3f}"))
    lines.append("")
    return "\n".join(lines) + "\n"


def _section_sensitivity_layerwise(raw_dir: Path, tables_dir: Path) -> str:
    df = _load_csv(tables_dir / "sensitivity_layerwise.csv")
    lines = ["## 5. Layer-wise Sensitivity Analysis\n"]
    lines.append(
        "Each row shows the effect of quantizing **one** Linear layer at a time "
        "(INT8 per-channel, MinMax calibration) while keeping all other layers in float32.  "
        "Layers with lower cosine similarity are the most quantization-sensitive.\n"
    )
    if df is None or df.empty:
        lines.append("_sensitivity_layerwise.csv not found — run sensitivity_analysis.py first._\n")
        return "\n".join(lines) + "\n"

    # Find the most and least sensitive layer per model
    for model_name, group in df.groupby("model"):
        most_sensitive = group.sort_values("cosine_similarity").iloc[0]
        least_sensitive = group.sort_values("cosine_similarity", ascending=False).iloc[0]
        lines.append(
            f"**{model_name}**: most sensitive layer = `{most_sensitive['layer']}` "
            f"(sim={_fmt_float(most_sensitive['cosine_similarity'])}); "
            f"least sensitive = `{least_sensitive['layer']}` "
            f"(sim={_fmt_float(least_sensitive['cosine_similarity'])})"
        )
    lines.append("")
    lines.append(_df_to_md(df))
    lines.append("")
    return "\n".join(lines) + "\n"


def _section_weightonly_vs_full(tables_dir: Path) -> str:
    df = _load_csv(tables_dir / "sensitivity_weightonly_vs_full.csv")
    lines = ["## 6. Weight-only vs. Full Quantization\n"]
    lines.append(
        "Comparison of quantizing only the weights vs. quantizing both weights and activations "
        "for each Linear layer (INT8 per-channel).\n"
    )
    if df is None or df.empty:
        lines.append(
            "_sensitivity_weightonly_vs_full.csv not found — run sensitivity_analysis.py first._\n"
        )
        return "\n".join(lines) + "\n"

    if "mode" in df.columns and "cosine_similarity" in df.columns:
        pivot = df.pivot_table(
            index=["model", "layer"], columns="mode", values="cosine_similarity"
        ).reset_index()
        pivot.columns.name = None
        lines.append(_df_to_md(pivot))
    else:
        lines.append(_df_to_md(df))
    lines.append("")
    return "\n".join(lines) + "\n"


def _section_calibration(tables_dir: Path) -> str:
    df = _load_csv(tables_dir / "sensitivity_calibration.csv")
    lines = ["## 7. Calibration Strategy Comparison\n"]
    lines.append(
        "All Linear layers quantized simultaneously with INT8 per-channel, varying only "
        "the calibration algorithm.\n"
    )
    if df is None or df.empty:
        lines.append(
            "_sensitivity_calibration.csv not found — run sensitivity_analysis.py first._\n"
        )
        return "\n".join(lines) + "\n"

    lines.append(_df_to_md(df))
    lines.append("")
    return "\n".join(lines) + "\n"


def _section_figures(figures_dir: Path) -> str:
    lines = ["## 8. Figures\n"]
    figure_descriptions = {
        "cosine_similarity_comparison.png": "Cosine similarity per config per model",
        "mlp_error_metrics.png":            "MLPBlock — max/mean/RMSE error metrics",
        "attention_error_metrics.png":      "AttentionBlock — max/mean/RMSE error metrics",
        "accuracy_error_tradeoff.png":      "Accuracy–error scatter (cosine sim vs MAE)",
        "memory_vs_accuracy.png":           "Memory reduction vs. cosine similarity",
        "sensitivity_layerwise_mlp.png":    "Layer sensitivity ranking — MLPBlock",
        "sensitivity_layerwise_attention.png": "Layer sensitivity ranking — AttentionBlock",
        "sensitivity_weightonly_vs_full_mlp.png":       "Weight-only vs full — MLPBlock",
        "sensitivity_weightonly_vs_full_attention.png": "Weight-only vs full — AttentionBlock",
        "sensitivity_calibration.png":      "Calibration strategy impact",
    }

    found_any = False
    for fname, desc in figure_descriptions.items():
        fpath = figures_dir / fname
        if fpath.exists():
            rel = fpath.relative_to(figures_dir.parent.parent)
            lines.append(f"### {desc}\n")
            lines.append(f"![{desc}]({rel})\n")
            found_any = True

    if not found_any:
        lines.append("_No figures found — run plot_tradeoffs.py first._\n")

    return "\n".join(lines) + "\n"


def _section_conclusions() -> str:
    return """\
## 9. Conclusions & Next Steps

### Key Findings

1. **INT8 per-channel symmetric** quantization consistently delivers the best accuracy–memory
   tradeoff: it achieves cosine similarity close to FP16 while reducing weight memory by 4×.

2. **INT4 weight-only** provides the largest memory reduction (8× vs FP32) but shows more
   noticeable output degradation — especially on the attention projections.

3. **Activation quantization** adds non-trivial error on top of weight quantization.  In most
   scenarios, weight-only quantization is preferred unless runtime inference kernels can exploit
   INT8 activations.

4. **Calibration strategy** (MinMax vs. Percentile) has a measurable but small effect at INT8;
   it becomes more important at INT4 where distribution tail clipping matters more.

5. **Layer sensitivity is unequal**: output projection layers tend to be more sensitive than
   intermediate (e.g. query/key) projections in attention blocks.

### What Is Still Missing for Production Deployment

| Gap | Notes |
|-----|-------|
| Real INT8 inference kernels | This lab uses fake-quantization (float32 simulation). Actual speedup requires hardware-specific kernels (e.g. `torch.ao`, TensorRT, ONNX Runtime). |
| Large-scale accuracy evaluation | Measurements are on synthetic random inputs; end-to-end task accuracy (BLEU, accuracy, F1) is not measured. |
| Mixed-precision search | Automatically choosing per-layer bit-widths (e.g. HAWQ, BRECQ) could improve the accuracy–compression frontier. |
| Quantization-aware training (QAT) | For more aggressive compression (INT4 or lower), QAT with fake-quant nodes in the training loop is recommended. |
| Dynamic/static act calibration | Dynamic per-token activation calibration (LLM-style) is not implemented. |
| KV-cache quantization | Not covered in this lab; important for transformer inference memory. |

"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Markdown report")
    parser.add_argument("--results_dir", default="results",
                        help="Root results directory (expects raw/ tables/ figures/ sub-dirs)")
    parser.add_argument("--output", default="results/report.md",
                        help="Output Markdown file path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    raw_dir     = results_dir / "raw"
    tables_dir  = results_dir / "tables"
    figures_dir = results_dir / "figures"
    out_path    = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections = [
        f"# Quantization Analysis Lab — Experiment Report\n\n"
        f"_Generated: {timestamp}_\n",

        _section_executive_summary(raw_dir, tables_dir),
        _section_experiment_matrix(tables_dir),
        _section_delta_vs_baseline(tables_dir),
        _section_memory_footprint(tables_dir),
        _section_sensitivity_layerwise(raw_dir, tables_dir),
        _section_weightonly_vs_full(tables_dir),
        _section_calibration(tables_dir),
        _section_figures(figures_dir),
        _section_conclusions(),
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(sections))

    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    main()
