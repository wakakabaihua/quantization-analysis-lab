#!/usr/bin/env python3
"""
Generate tradeoff and sensitivity visualizations from experiment results.

Plot catalogue:
  1.  cosine_similarity_comparison.png   — bar chart per config, per model
  2.  error_metrics_{model}.png          — Max/Mean/RMSE bars per model
  3.  accuracy_error_tradeoff.png        — scatter: MAE vs cosine similarity
  4.  layerwise_sensitivity_heatmap.png  — heatmap: config × layer cosine sim
  5.  sensitivity_layerwise_{model}.png  — single-layer cosine sim ranking
  6.  sensitivity_weightonly_vs_full_{model}.png — weight-only vs full
  7.  sensitivity_calibration.png        — minmax vs percentile bar chart
  8.  memory_vs_accuracy.png             — memory reduction vs cosine sim
  9.  error_distribution_{config}_{model}.png — (requires raw tensors; skipped)

Usage:
    uv run python scripts/plot_tradeoffs.py
    uv run python scripts/plot_tradeoffs.py --results_dir results/raw --output_dir results/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Color palette (consistent across all plots)
# ---------------------------------------------------------------------------
_CONFIG_ORDER = [
    "fp16_baseline",
    "int8_per_tensor_sym",
    "int8_per_channel_sym",
    "int8_per_tensor_asym",
    "int8_per_channel_asym",
    "int4_weight_only",
]
_PALETTE = plt.cm.tab10.colors  # type: ignore[attr-defined]
_CONFIG_COLOR = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(_CONFIG_ORDER)}


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


def load_main(results_dir: str) -> pd.DataFrame:
    skip = ("sensitivity_",)
    records = []
    for path in sorted(Path(results_dir).glob("*.json")):
        if any(path.name.startswith(p) for p in skip):
            continue
        with open(path) as f:
            data = json.load(f)
        records.extend(data)
    return pd.DataFrame(records) if records else pd.DataFrame()


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by canonical config order."""
    order_map = {c: i for i, c in enumerate(_CONFIG_ORDER)}
    df = df.copy()
    df["_o"] = df["config"].map(order_map).fillna(99)
    return df.sort_values("_o").drop(columns=["_o"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _bar(ax, labels, values, colors=None, default_color="steelblue"):
    x = np.arange(len(labels))
    bars = ax.bar(
        x, values,
        color=colors if colors else [default_color] * len(labels),
        width=0.6, edgecolor="white", linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    return bars


# ---------------------------------------------------------------------------
# Plot 1 — Cosine similarity bar chart
# ---------------------------------------------------------------------------

def plot_similarity_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    if "cosine_similarity" not in df.columns:
        return
    model_names = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5), squeeze=False)

    for col_idx, model_name in enumerate(model_names):
        ax = axes[0][col_idx]
        group = _ordered(df[df["model"] == model_name])
        colors = [_CONFIG_COLOR.get(c, "grey") for c in group["config"]]
        _bar(ax, list(group["config"]), list(group["cosine_similarity"]), colors=colors)

        fp16_val = group.loc[group["config"] == "fp16_baseline", "cosine_similarity"]
        ref = float(fp16_val.iloc[0]) if not fp16_val.empty else 1.0
        ax.axhline(ref, color="crimson", linestyle="--", linewidth=1, label=f"FP16 ({ref:.4f})")

        y_min = max(0.0, float(group["cosine_similarity"].min()) - 0.005)
        ax.set_ylim(y_min, 1.002)
        ax.set_title(f"{model_name} — Output Cosine Similarity", fontsize=11, fontweight="bold")
        ax.set_ylabel("Cosine Similarity (↑ better)")
        ax.set_xlabel("Config")
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # Annotate values
        for bar, val in zip(ax.patches, group["cosine_similarity"]):
            ax.text(bar.get_x() + bar.get_width() / 2, float(val) + 0.0003,
                    f"{val:.5f}", ha="center", va="bottom", fontsize=6.5)

    plt.suptitle("Quantization Accuracy — Cosine Similarity", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, output_dir / "cosine_similarity_comparison.png")


# ---------------------------------------------------------------------------
# Plot 2 — Error metrics bars per model
# ---------------------------------------------------------------------------

def plot_error_metrics(df: pd.DataFrame, output_dir: Path) -> None:
    error_cols = [
        ("max_absolute_error",   "Max Abs Error",  "tomato"),
        ("mean_absolute_error",  "Mean Abs Error", "darkorange"),
        ("root_mean_squared_error", "RMSE",         "goldenrod"),
    ]
    available = [(col, lbl, clr) for col, lbl, clr in error_cols if col in df.columns]
    if not available:
        return

    for model_name, group in df.groupby("model"):
        group = _ordered(group)
        fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5), squeeze=False)

        for idx, (col, label, color) in enumerate(available):
            ax = axes[0][idx]
            _bar(ax, list(group["config"]), list(group[col]), default_color=color)
            ax.set_title(f"{label}", fontsize=10, fontweight="bold")
            ax.set_ylabel(label)
            ax.set_xlabel("Config")
            ax.grid(axis="y", linestyle="--", alpha=0.4)

        plt.suptitle(f"Error Metrics — {model_name}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        _save(fig, output_dir / f"{model_name}_error_metrics.png")


# ---------------------------------------------------------------------------
# Plot 3 — Accuracy–error scatter
# ---------------------------------------------------------------------------

def plot_tradeoff_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    if "cosine_similarity" not in df.columns or "mean_absolute_error" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    marker_cycle = ["o", "s", "^", "D", "v", "P"]

    for idx, (model_name, group) in enumerate(df.groupby("model")):
        marker = marker_cycle[idx % len(marker_cycle)]
        for _, row in group.iterrows():
            color = _CONFIG_COLOR.get(row["config"], "grey")
            ax.scatter(
                float(row["mean_absolute_error"]), float(row["cosine_similarity"]),
                color=color, marker=marker, s=100, zorder=3,
                label=f"{row['config']} ({model_name})" if idx == 0 else None,
            )
            ax.annotate(
                f"{row['config']}\n({model_name})",
                (float(row["mean_absolute_error"]), float(row["cosine_similarity"])),
                fontsize=6.5, xytext=(5, 4), textcoords="offset points", color=color,
            )

    ax.set_xlabel("Mean Absolute Error (↓ better)", fontsize=11)
    ax.set_ylabel("Cosine Similarity (↑ better)", fontsize=11)
    ax.set_title("Quantization Accuracy–Error Tradeoff", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    _save(fig, output_dir / "accuracy_error_tradeoff.png")


# ---------------------------------------------------------------------------
# Plot 4 — Layer-wise sensitivity heatmap (config × layer)
# ---------------------------------------------------------------------------

def plot_layerwise_heatmap(results_dir: str, output_dir: Path) -> None:
    lw = _load_glob(results_dir, "sensitivity_layerwise.json")
    if lw.empty or "cosine_similarity" not in lw.columns:
        return

    for model_name, group in lw.groupby("model"):
        pivot = group.pivot_table(
            index="layer", values="cosine_similarity", aggfunc="mean"
        )
        # We need a "config" axis too — this is single-config (INT8 minmax),
        # so draw as a horizontal bar chart instead of a heatmap.
        fig, ax = plt.subplots(figsize=(8, max(3, len(pivot) * 0.6 + 1)))

        layers = list(pivot.index)
        sims = list(pivot["cosine_similarity"])

        # Color by deviation from 1
        deviations = np.array([1.0 - s for s in sims])
        norm = plt.Normalize(vmin=0, vmax=max(deviations.max(), 1e-6))
        cmap = cm.RdYlGn_r
        colors = [cmap(norm(d)) for d in deviations]

        y = np.arange(len(layers))
        bars = ax.barh(y, sims, color=colors, edgecolor="white", height=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(layers, fontsize=9)
        ax.set_xlim(max(0, min(sims) - 0.002), 1.001)
        ax.axvline(1.0, color="crimson", linestyle="--", linewidth=1)
        ax.set_xlabel("Cosine Similarity (↑ better, ↓ = more sensitive)")
        ax.set_title(
            f"{model_name} — Layer-wise Sensitivity (INT8 per-channel, one layer at a time)",
            fontsize=10, fontweight="bold",
        )
        ax.grid(axis="x", linestyle="--", alpha=0.4)

        # Annotate
        for bar, val in zip(bars, sims):
            ax.text(float(val) - 0.0004, bar.get_y() + bar.get_height() / 2,
                    f"{val:.6f}", va="center", ha="right", fontsize=7.5, color="black")

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="1 − cosine_sim (degradation)", pad=0.01)

        plt.tight_layout()
        _save(fig, output_dir / f"sensitivity_layerwise_{model_name}.png")


# ---------------------------------------------------------------------------
# Plot 5 — Weight-only vs. full quantization per layer
# ---------------------------------------------------------------------------

def plot_weightonly_vs_full(results_dir: str, output_dir: Path) -> None:
    wf = _load_glob(results_dir, "sensitivity_weightonly_vs_full.json")
    if wf.empty:
        return

    for model_name, group in wf.groupby("model"):
        layers = sorted(group["layer"].unique())
        modes = sorted(group["mode"].unique())
        x = np.arange(len(layers))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(7, len(layers) * 1.5), 5))

        for i, mode in enumerate(modes):
            subset = group[group["mode"] == mode].set_index("layer")
            vals = [float(subset.loc[l, "cosine_similarity"]) if l in subset.index else 0.0
                    for l in layers]
            color = "steelblue" if "weight_only" in mode else "tomato"
            rects = ax.bar(x + i * width, vals, width, label=mode, color=color, alpha=0.85)

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(layers, rotation=30, ha="right", fontsize=9)
        y_min = max(0.0, group["cosine_similarity"].min() - 0.005)
        ax.set_ylim(y_min, 1.003)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylabel("Cosine Similarity (↑ better)")
        ax.set_title(
            f"{model_name} — Weight-only vs. Weight+Activation Quantization (INT8)",
            fontsize=10, fontweight="bold",
        )
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        _save(fig, output_dir / f"sensitivity_weightonly_vs_full_{model_name}.png")


# ---------------------------------------------------------------------------
# Plot 6 — Calibration strategy comparison
# ---------------------------------------------------------------------------

def plot_calibration_comparison(results_dir: str, output_dir: Path) -> None:
    cal = _load_glob(results_dir, "sensitivity_calibration.json")
    if cal.empty:
        return

    model_names = sorted(cal["model"].unique())
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5), squeeze=False)

    for col_idx, model_name in enumerate(model_names):
        ax = axes[0][col_idx]
        group = cal[cal["model"] == model_name].sort_values("calibration")
        labels = list(group["calibration"])
        vals = list(group["cosine_similarity"])
        colors = ["steelblue", "darkorange", "seagreen"]
        _bar(ax, labels, vals, colors=colors[:len(labels)])
        y_min = max(0.0, min(vals) - 0.005)
        ax.set_ylim(y_min, 1.003)
        ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1)
        ax.set_title(f"{model_name} — Calibration Strategy (INT8 per-channel)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Cosine Similarity (↑ better)")
        ax.set_xlabel("Calibration Method")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        for bar, val in zip(ax.patches, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, float(val) + 0.0003,
                    f"{val:.6f}", ha="center", va="bottom", fontsize=7)

    plt.suptitle("Calibration Strategy Impact on Accuracy", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, output_dir / "sensitivity_calibration.png")


# ---------------------------------------------------------------------------
# Plot 7 — Memory reduction vs. cosine similarity
# ---------------------------------------------------------------------------

_MEMORY_REDUCTION = {
    "fp16_baseline":        2.0,
    "int8_per_tensor_sym":  4.0,
    "int8_per_channel_sym": 4.0,
    "int8_per_tensor_asym": 4.0,
    "int8_per_channel_asym":4.0,
    "int4_weight_only":     8.0,
}


def plot_memory_vs_accuracy(df: pd.DataFrame, output_dir: Path) -> None:
    if "cosine_similarity" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    marker_cycle = ["o", "s", "^", "D"]

    for idx, (model_name, group) in enumerate(df.groupby("model")):
        marker = marker_cycle[idx % len(marker_cycle)]
        for _, row in group.iterrows():
            config = row["config"]
            mem_x = _MEMORY_REDUCTION.get(config, 1.0)
            color = _CONFIG_COLOR.get(config, "grey")
            ax.scatter(mem_x, float(row["cosine_similarity"]),
                       color=color, marker=marker, s=120, zorder=3)
            ax.annotate(
                f"{config}\n({model_name})",
                (mem_x, float(row["cosine_similarity"])),
                fontsize=6.5, xytext=(5, 4), textcoords="offset points", color=color,
            )

    ax.set_xlabel("Weight Memory Reduction vs FP32 (×)", fontsize=11)
    ax.set_ylabel("Cosine Similarity (↑ better)", fontsize=11)
    ax.set_title("Memory Reduction vs. Accuracy Tradeoff", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks([2, 4, 8])
    ax.set_xticklabels(["2× (FP16)", "4× (INT8)", "8× (INT4)"])
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    _save(fig, output_dir / "memory_vs_accuracy.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tradeoff and sensitivity plots")
    parser.add_argument("--results_dir", default="results/raw")
    parser.add_argument("--output_dir", default="results/figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_main(args.results_dir)
    if df.empty:
        print("No results found — run experiments first.")
        return

    print("Generating plots ...")
    plot_similarity_comparison(df, output_dir)
    plot_error_metrics(df, output_dir)
    plot_tradeoff_scatter(df, output_dir)
    plot_layerwise_heatmap(args.results_dir, output_dir)
    plot_weightonly_vs_full(args.results_dir, output_dir)
    plot_calibration_comparison(args.results_dir, output_dir)
    plot_memory_vs_accuracy(df, output_dir)
    print("All plots generated.")


if __name__ == "__main__":
    main()
