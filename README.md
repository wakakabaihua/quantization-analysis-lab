# Quantization Analysis Lab

A post-training quantization (PTQ) analysis framework for understanding
low-precision inference tradeoffs in transformer-style architectures.

This project intentionally implements quantization from first principles
rather than wrapping an existing toolkit. The goal is to make the tradeoffs
between precision, accuracy, and memory explicitly visible and measurable.

**Status**: All six implementation phases complete. 197 tests passing.

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

## Backend Ecosystem

Six quantization backends implemented:

| Backend | Technique | Real Kernels | Test Count |
|---------|-----------|--------------|------------|
| `FakeQuantBackend` | Observer → Calibrator → FakeQuantize pipeline | No (float32 simulation) | 41 |
| `TorchAOBackend` | `torch.ao.quantization.quantize_dynamic` | **Yes** (real INT8 kernels) | 56 |
| `BitsAndBytesBackend` | LLM.int8() / NF4 (QLoRA) | **Yes** (int8/int4 CUDA kernels) | 80 |
| `GPTQBackend` | Hessian-guided weight quantization (INT4 packed) | **Yes** (INT4 packed storage, Triton dequant on CUDA) | 136 |
| `AWQBackend` | Activation-aware weight scaling (INT4 packed) | **Yes** (INT4 packed storage, Triton dequant on CUDA) | 136 |
| `MixedPrecisionBackend` | Per-layer bit-width assignment | No (fake-quant forward) | 170 |

**Phase 6 (INT4 Kernels)**: 27 new tests cover `pack_int4`, `unpack_int4`, `quantize_to_uint4`, `compute_groupwise_qparams`, Triton `int4_dequant_gemm`, and end-to-end numerical equivalence with fake-quant. Total: **197 tests**.

All backends share the `QuantBackend` ABC (`calibrate` / `convert`) and are registered in the backend registry.

## Experiment Matrix

Core configurations (see `configs/` for all YAML files):

| Config | Calibration | Granularity | Symmetric | Mode |
|--------|-------------|-------------|-----------|------|
| FP16 baseline | — | — | — | Reference |
| INT8 per-tensor symmetric | Min-max | Per-tensor | Yes | Weights + Activations |
| INT8 per-channel symmetric | Min-max | Per-channel | Yes | Weights + Activations |
| INT8 per-tensor asymmetric | Percentile | Per-tensor | No | Weights + Activations |
| INT8 per-channel asymmetric | Percentile | Per-channel | No | Weights + Activations |
| INT4 weight-only | Percentile | Per-channel | Yes | Weights only |
| Mixed-precision INT8/INT4 | Sensitivity-guided | Per-layer | Yes | Weights only |

## Key Design Decisions

- **Hybrid fake-quant + real kernels**: 
  - `TorchAOBackend` and `BitsAndBytesBackend` use real INT8/INT4 kernels for actual speedup.
  - **Phase 6**: `GPTQBackend` and `AWQBackend` now store weights in **packed INT4 format** (8 nibbles per int32, AutoGPTQ-compatible) and dispatch to **Triton dequant kernels** on CUDA or PyTorch fallback on CPU. INT8 and other bit-widths fall back to float32 for compatibility.
  - `FakeQuantBackend` and `MixedPrecisionBackend` continue using fake-quantization (float32 simulation) for maximum portability and algorithm transparency.
- **Observer/Calibrator separation**: Statistics collection and scale
  computation are decoupled so different calibration strategies can be
  applied to the same collected data.
- **Per-layer sensitivity**: Every experiment logs layer-wise error
  contributions so bottleneck layers can be identified.
- **Sensitivity-guided mixed-precision**: `MixedPrecisionBackend.from_sensitivity()` automatically assigns per-layer bit-widths based on quantization sensitivity scores.

## Key Results

See `results/tables/summary.csv` and `docs/conclusion.md` after running
all experiments.

## Documentation

- [docs/design.md](docs/design.md) — Quantization design decisions
- [docs/experiment_matrix.md](docs/experiment_matrix.md) — Full experiment plan
- [docs/conclusion.md](docs/conclusion.md) — Findings and tradeoff summary
