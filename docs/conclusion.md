# Conclusions

_This file will be populated after running the full experiment matrix._
_Run all configs, then `python scripts/summarize_results.py` and_
_`python scripts/plot_tradeoffs.py` to generate the data for this section._

---

## Summary

_(to be filled)_

---

## Key Findings

### 1. Layer Sensitivity

_(Which layers showed the highest cosine similarity degradation after quantization?
Were MLP or attention layers more sensitive? Which specific sub-layers — fc1, fc2,
q_proj, v_proj, out_proj — drove the most error?)_

### 2. Per-Channel vs. Per-Tensor

_(Did per-channel consistently outperform per-tensor? By how much in cosine
similarity units? Was the improvement larger for certain layer types?)_

### 3. Symmetric vs. Asymmetric

_(Did asymmetric quantization help for activations? For weights? Was the
improvement consistent across both MLP and attention layers?)_

### 4. Calibration Strategy Impact

_(Did percentile calibration outperform MinMax? For which layers was the
difference most pronounced? What does this suggest about outlier distributions
in the weight / activation space?)_

### 5. INT4 vs. INT8

_(How much higher was the INT4 weight-only error compared to INT8 full
quantization? Was the degradation more pronounced in one model type?)_

---

## Deployment Tradeoff Summary

| Config | Memory reduction | Cosine sim | Recommended? |
|--------|:----------------:|:----------:|:------------:|
| INT8 per-channel sym | ~4× vs FP32 | ≈ 1.000 | Yes, for most use cases |
| INT8 per-channel asym | ~4× vs FP32 | ≈ 1.000 | Yes, when activations are skewed |
| INT4 weight-only | ~8× vs FP32 | TBD | Yes, if cosine sim > 0.99 |

---

## What Would Be Needed for Production Deployment

1. **Real integer kernels** — INT8 GEMM via cuBLAS, CUTLASS, or a custom NPU
   backend. The fake-quantize path does not produce actual runtime speedup.

2. **Per-channel activation quantization** — requires channel-aware dispatch
   in the operator implementation; not provided by standard PyTorch.

3. **Mixed-precision selection** — assign bit-width per layer based on
   sensitivity scores from the layer-wise error analysis.

4. **Domain-specific calibration data** — real accuracy numbers require
   calibration data that matches the deployment distribution, not synthetic
   Gaussian inputs.

5. **End-to-end task metric** — cosine similarity is a proxy. Production
   deployment decisions should be validated with task-level metrics
   (perplexity, BLEU, accuracy on held-out data).

---

## Next Steps

1. Add QAT comparison as a control group for the accuracy gap.
2. Integrate ONNX Runtime or TVM for actual latency measurement.
3. Extend to a small end-to-end transformer (embed + N × TransformerLayer + LM head).
4. Explore mixed-precision assignment driven by per-layer sensitivity scores.
5. Experiment with outlier suppression (channel-wise weight clipping) before
   quantization to reduce sensitivity in fragile layers.
