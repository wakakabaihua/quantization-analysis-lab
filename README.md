# Quantization Analysis Lab

A post-training quantization (PTQ) analysis framework for understanding
low-precision inference tradeoffs in transformer-style architectures.

This project intentionally implements quantization from first principles
rather than wrapping an existing toolkit. The goal is to make the tradeoffs
between precision, accuracy, and memory explicitly visible and measurable.

## What This Project Answers

1. How do different quantization strategies affect accuracy and runtime characteristics?
2. How should quantization decisions be made at the layer, tensor, and calibration level?
3. Which layers are most sensitive to quantization and why?

## Project Structure

```text
quantization-analysis-lab/
  configs/          # Experiment configuration files (YAML)
  scripts/          # Experiment runner and analysis scripts
  src/
    models/         # Baseline reference modules (MLP, Attention)
    quant/          # Fake quant, observers, calibrators, PTQ pipeline
    utils/          # Metrics and logging
  results/          # Experiment outputs (raw data, figures, tables)
  tests/            # Unit tests for core quantization primitives
  docs/             # Design notes and experiment analysis
```

## Setup

```bash
pip install -r requirements.txt
```

## Running Experiments

Run a single config:

```bash
python scripts/run_experiments.py --config configs/ptq_int8_per_tensor.yaml
```

Run all configs:

```bash
for cfg in configs/*.yaml; do
    python scripts/run_experiments.py --config "$cfg"
done
```

Evaluate accuracy across all results:

```bash
python scripts/evaluate_accuracy.py --results_dir results/raw
```

Generate summary table:

```bash
python scripts/summarize_results.py \
    --results_dir results/raw \
    --output results/tables/summary.csv
```

Generate tradeoff plots:

```bash
python scripts/plot_tradeoffs.py \
    --results_dir results/raw \
    --output_dir results/figures
```

## Experiment Matrix

| Config | Calibration | Granularity | Symmetric | Mode |
|--------|-------------|-------------|-----------|------|
| FP16 baseline | — | — | — | Reference |
| INT8 | Min-max | Per-tensor | Yes | Weights + Activations |
| INT8 | Min-max | Per-channel | Yes | Weights + Activations |
| INT8 | Percentile | Per-tensor | No | Weights + Activations |
| INT8 | Percentile | Per-channel | No | Weights + Activations |
| INT4 weight-only | Percentile | Per-channel | Yes | Weights only |

## Key Design Decisions

- **Fake quantization**: All quantization simulates the effect in float32.
  No integer kernels are used, so actual runtime speedup is not measured.
  This is a deliberate tradeoff to keep analysis code portable and explicit.
- **Observer/Calibrator separation**: Statistics collection and scale
  computation are decoupled so different calibration strategies can be
  applied to the same collected data.
- **Per-layer sensitivity**: Every experiment logs layer-wise error
  contributions so bottleneck layers can be identified.

## Key Results

See `results/tables/summary.csv` and `docs/conclusion.md` after running
all experiments.

## Documentation

- [docs/design.md](docs/design.md) — Quantization design decisions
- [docs/experiment_matrix.md](docs/experiment_matrix.md) — Full experiment plan
- [docs/conclusion.md](docs/conclusion.md) — Findings and tradeoff summary
