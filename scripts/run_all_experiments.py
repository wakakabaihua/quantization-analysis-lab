#!/usr/bin/env python3
"""
Batch experiment runner: executes all YAML configs in configs/ sequentially,
then automatically invokes sensitivity analysis, summarization, and plotting.

Progress, per-experiment timing, and a final wall-time summary are printed
to stdout. Errors in individual experiments are caught and reported without
stopping the remaining runs.

Usage:
    uv run python scripts/run_all_experiments.py
    uv run python scripts/run_all_experiments.py --configs_dir configs --output_dir results/raw
    uv run python scripts/run_all_experiments.py --skip_post   # skip analysis steps
    uv run python scripts/run_all_experiments.py --device cuda
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _run_script(script: str, extra_args: list[str]) -> tuple[bool, float]:
    """Run a scripts/*.py file via the same interpreter. Returns (ok, elapsed)."""
    t0 = time.perf_counter()
    cmd = [sys.executable, str(HERE / script)] + extra_args
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.perf_counter() - t0
    return result.returncode == 0, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run all PTQ experiments")
    parser.add_argument("--configs_dir", default="configs")
    parser.add_argument("--output_dir", default="results/raw")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--skip_post",
        action="store_true",
        help="Skip sensitivity analysis, summarization and plotting",
    )
    args = parser.parse_args()

    configs_dir = ROOT / args.configs_dir
    yaml_files = sorted(configs_dir.glob("*.yaml"))
    if not yaml_files:
        print(f"No YAML files found in {configs_dir}")
        sys.exit(1)

    print("=" * 64)
    print(f"  Quantization Analysis Lab — Batch Runner")
    print(f"  Configs : {configs_dir} ({len(yaml_files)} files)")
    print(f"  Output  : {args.output_dir}")
    print(f"  Device  : {args.device}")
    print("=" * 64)

    results: list[dict] = []
    wall_start = time.perf_counter()

    # -----------------------------------------------------------------------
    # Step 1: Run each config
    # -----------------------------------------------------------------------
    for idx, cfg_path in enumerate(yaml_files, 1):
        print(f"\n[{idx}/{len(yaml_files)}] {cfg_path.name}")
        ok, elapsed = _run_script(
            "run_experiments.py",
            ["--config", str(cfg_path),
             "--output_dir", args.output_dir,
             "--device", args.device],
        )
        status = "OK" if ok else "FAILED"
        results.append({"config": cfg_path.name, "status": status, "time": elapsed})
        print(f"  → {status}  ({_fmt(elapsed)})")

    # -----------------------------------------------------------------------
    # Step 2: Post-processing (unless --skip_post)
    # -----------------------------------------------------------------------
    post_steps: list[tuple[str, list[str]]] = []

    if not args.skip_post:
        post_steps = [
            ("sensitivity_analysis.py",
             ["--output_dir", args.output_dir, "--device", args.device]),
            ("summarize_results.py",
             ["--results_dir", args.output_dir,
              "--output", "results/tables/summary.csv"]),
            ("plot_tradeoffs.py",
             ["--results_dir", args.output_dir,
              "--output_dir", "results/figures"]),
            ("generate_report.py",
             ["--results_dir", "results",
              "--output", "results/report.md"]),
        ]

    for script, extra in post_steps:
        print(f"\n[post] {script} ...")
        ok, elapsed = _run_script(script, extra)
        status = "OK" if ok else "FAILED"
        results.append({"config": script, "status": status, "time": elapsed})
        print(f"  → {status}  ({_fmt(elapsed)})")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    total = time.perf_counter() - wall_start
    print("\n" + "=" * 64)
    print("  Run summary")
    print("=" * 64)
    col_w = max(len(r["config"]) for r in results) + 2
    for r in results:
        mark = "✓" if r["status"] == "OK" else "✗"
        print(f"  {mark}  {r['config']:<{col_w}}  {_fmt(r['time']):>6}  {r['status']}")
    print("-" * 64)
    print(f"  Total wall time: {_fmt(total)}")
    print("=" * 64)

    failed = [r for r in results if r["status"] == "FAILED"]
    if failed:
        print(f"\n  {len(failed)} step(s) failed — check output above.")
        sys.exit(1)
    print("\n  All steps completed successfully.")
    print(f"  Report: results/report.md")


if __name__ == "__main__":
    main()
