# Design Notes

## Overview

This project implements a post-training quantization (PTQ) analysis framework
from first principles. Rather than wrapping an existing toolkit (GPTQ, AWQ,
bitsandbytes), each component is explicit so that tradeoffs between precision,
accuracy, and memory are directly observable and measurable.

The target audience for the resulting artifacts is a reviewer evaluating
low-precision inference understanding for compiler or deployment roles.

---

## Component Architecture

```
calibration data
      │
      ▼
  BaseObserver  ──── MinMaxObserver
  (forward hook)     HistogramObserver
      │
      │ stats dict
      ▼
  BaseCalibrator ─── MinMaxCalibrator
                     PercentileCalibrator
                     KLCalibrator
      │
      │ (scale, zero_point)
      ▼
  FakeQuantize ──── wraps weights + activations per Linear layer
      │
      ▼
  QuantizedLinear  (replaces nn.Linear in PTQPipeline.quantize())
```

---

## Design Decisions

### Fake Quantization

All quantization is implemented as fake quantization: quantize to integers
then immediately dequantize back to float32. This simulates the effect of
quantization without requiring integer compute kernels.

**Implication**: Runtime speedup is not measured in the first version. Actual
speedup requires INT8 GEMM support at the kernel level (cuBLAS, CUTLASS,
TensorRT, TVM, etc.). The framework is designed to be extended with a real
backend once the accuracy tradeoffs are understood.

### Observer / Calibrator Separation

Statistics collection (observer) and scale computation (calibrator) are
separate classes with a shared dict interface. This means:

- The same observer data can be post-processed by different calibrators
  without rerunning forward passes.
- MinMax stats are sufficient for `MinMaxCalibrator` but not for
  `PercentileCalibrator` or `KLCalibrator`, which require a histogram.
- The pipeline selects the right observer type based on the calibration method.

### Granularity

| Granularity   | Scale shape         | When useful |
|---------------|---------------------|-------------|
| Per-tensor    | scalar              | Simplest; baseline for comparison |
| Per-channel   | [C_out] for weights | When weight channels have different dynamic ranges |

Per-channel activation quantization is *not* implemented. It requires
channel-aware dispatch in the runtime kernel and is rarely used in practice.

### Symmetric vs. Asymmetric

| Mode        | Zero-point | Integer range       | Good for |
|-------------|------------|---------------------|----------|
| Symmetric   | 0          | [-127, 127] (int8)  | Weights; zero-centered distributions |
| Asymmetric  | ≠ 0        | [0, 255] (uint8)    | Post-ReLU/GELU activations |

### Calibration Strategies

| Strategy    | Observer needed  | Sensitivity to outliers | Cost |
|-------------|-----------------|------------------------|------|
| MinMax      | MinMaxObserver  | High (one outlier inflates range) | O(1)/batch |
| Percentile  | HistogramObserver | Low (clips tail) | O(N) samples |
| KL          | HistogramObserver | Principled minimization | O(bins²) |

---

## Limitations

1. **No integer kernels** — no real runtime speedup is measured.
2. **Activation quantization is per-tensor only** — per-channel activations
   require channel-aware runtime that is out of scope here.
3. **INT4 implementation** — uses the same fake-quantize path as INT8;
   actual INT4 packing and decompression are not implemented.
4. **KL calibrator is approximate** — the histogram-search approach is
   conceptually correct but not identical to TensorRT's implementation.
5. **Synthetic calibration data** — real accuracy numbers require domain-
   specific calibration data (e.g., natural language tokens, image batches).

---

## Extension Points

1. Add QAT: replace `FakeQuantize.enabled = True` during training.
2. Add ONNX export: call `torch.onnx.export` on the quantized model.
3. Add TVM / ONNX Runtime backend: measure actual latency vs. float.
4. Add mixed-precision: select per-layer bit-width from sensitivity scores.
5. Extend to a small end-to-end transformer: wire `MLPBlock` + `AttentionBlock`
   into a `TransformerLayer`, add embedding + LM head, evaluate on a toy task.
