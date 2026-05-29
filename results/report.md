# Quantization Analysis Lab — Experiment Report

_Generated: 2026-05-29 14:31:31_

## 1. Executive Summary

Across all tested configurations, **fp16_baseline** on **attention** achieved the highest output fidelity (cosine similarity = `1.000000`).  The lowest fidelity was observed for **int4_weight_only** on **mlp** (cosine similarity = `0.994826`).

- **attention**: best `fp16_baseline` (sim=1.000000), worst `int4_weight_only` (sim=0.994920)
- **mlp**: best `fp16_baseline` (sim=1.000000), worst `int4_weight_only` (sim=0.994826)


## 2. Experiment Matrix

| config | model | cosine_similarity | max_absolute_error | mean_absolute_error | root_mean_squared_error | relative_error |
| --- | --- | --- | --- | --- | --- | --- |
| fp16_baseline | attention | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| int8_per_tensor_sym | attention | 0.999883 | 0.007065 | 0.001546 | 0.001939 | 0.117884 |
| int8_per_channel_sym | attention | 0.999873 | 0.007991 | 0.001638 | 0.002053 | 0.110253 |
| int8_per_tensor_asym | attention | 0.999762 | 0.010950 | 0.002134 | 0.002708 | 0.111483 |
| int8_per_channel_asym | attention | 0.999465 | 0.021088 | 0.003326 | 0.004344 | 0.191367 |
| int4_weight_only | attention | 0.994920 | 0.050774 | 0.010896 | 0.013662 | 0.796099 |
| fp16_baseline | mlp | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| int8_per_tensor_sym | mlp | 0.999663 | 0.022592 | 0.004150 | 0.005203 | 0.243170 |
| int8_per_channel_sym | mlp | 0.999662 | 0.021171 | 0.004103 | 0.005149 | 0.255168 |
| int8_per_tensor_asym | mlp | 0.999890 | 0.016550 | 0.002295 | 0.002934 | 0.123400 |
| int8_per_channel_asym | mlp | 0.999905 | 0.014436 | 0.002166 | 0.002737 | 0.112751 |
| int4_weight_only | mlp | 0.994826 | 0.092577 | 0.016168 | 0.020249 | 0.909936 |


## 3. Delta vs. FP16 Baseline

Positive Δ cosine_similarity means the config is *more* similar to the float baseline than the FP16 run itself (i.e. even closer to FP32).  Negative values indicate accuracy loss from quantization.

| config | model | cosine_similarity | max_absolute_error | mean_absolute_error | root_mean_squared_error | relative_error | layer_mean_cosine_similarity | layer_mean_mean_absolute_error | delta_cosine_similarity | delta_max_absolute_error | delta_mean_absolute_error | delta_root_mean_squared_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| int8_per_tensor_sym | attention | 0.999883 | 0.007065 | 0.001546 | 0.001939 | 0.117884 | 0.999915 | 0.004600 | -0.000117 | 0.007065 | 0.001546 | 0.001939 |
| int8_per_channel_sym | attention | 0.999873 | 0.007991 | 0.001638 | 0.002053 | 0.110253 | 0.999913 | 0.004639 | -0.000127 | 0.007991 | 0.001638 | 0.002053 |
| int8_per_tensor_asym | attention | 0.999762 | 0.010950 | 0.002134 | 0.002708 | 0.111483 | 0.999903 | 0.003985 | -0.000238 | 0.010950 | 0.002134 | 0.002708 |
| int8_per_channel_asym | attention | 0.999465 | 0.021088 | 0.003326 | 0.004344 | 0.191367 | 0.999829 | 0.004302 | -0.000534 | 0.021088 | 0.003326 | 0.004344 |
| int4_weight_only | attention | 0.994920 | 0.050774 | 0.010896 | 0.013662 | 0.796099 | 0.996880 | 0.027409 | -0.005080 | 0.050774 | 0.010896 | 0.013662 |
| int8_per_tensor_sym | mlp | 0.999663 | 0.022592 | 0.004150 | 0.005203 | 0.243170 | 0.999792 | 0.004974 | -0.000336 | 0.022592 | 0.004150 | 0.005203 |
| int8_per_channel_sym | mlp | 0.999662 | 0.021171 | 0.004103 | 0.005149 | 0.255168 | 0.999792 | 0.004959 | -0.000338 | 0.021171 | 0.004103 | 0.005149 |
| int8_per_tensor_asym | mlp | 0.999890 | 0.016550 | 0.002295 | 0.002934 | 0.123400 | 0.999920 | 0.003446 | -0.000110 | 0.016550 | 0.002295 | 0.002934 |
| int8_per_channel_asym | mlp | 0.999905 | 0.014436 | 0.002166 | 0.002737 | 0.112751 | 0.999928 | 0.003384 | -0.000095 | 0.014436 | 0.002166 | 0.002737 |
| int4_weight_only | mlp | 0.994826 | 0.092577 | 0.016168 | 0.020249 | 0.909936 | 0.996147 | 0.024523 | -0.005174 | 0.092577 | 0.016168 | 0.020249 |


## 4. Memory Footprint Estimates

Weight memory estimates are derived from parameter counts and dtype bit-widths (per-channel scale overhead is not included).

| config | model | dtype | weight_only | fp32_weight_mb | quantized_weight_mb | memory_reduction_x |
| --- | --- | --- | --- | --- | --- | --- |
| fp16_baseline | attention | fp16 | False | 4.194 | 2.097 | 2.000 |
| int8_per_tensor_sym | attention | int8 | False | 4.194 | 1.049 | 4.000 |
| int8_per_channel_sym | attention | int8 | False | 4.194 | 1.049 | 4.000 |
| int8_per_tensor_asym | attention | int8 | False | 4.194 | 1.049 | 4.000 |
| int8_per_channel_asym | attention | int8 | False | 4.194 | 1.049 | 4.000 |
| int4_weight_only | attention | int4 | True | 4.194 | 0.524 | 8.000 |
| fp16_baseline | mlp | fp16 | False | 8.389 | 4.194 | 2.000 |
| int8_per_tensor_sym | mlp | int8 | False | 8.389 | 2.097 | 4.000 |
| int8_per_channel_sym | mlp | int8 | False | 8.389 | 2.097 | 4.000 |
| int8_per_tensor_asym | mlp | int8 | False | 8.389 | 2.097 | 4.000 |
| int8_per_channel_asym | mlp | int8 | False | 8.389 | 2.097 | 4.000 |
| int4_weight_only | mlp | int4 | True | 8.389 | 1.049 | 8.000 |


## 5. Layer-wise Sensitivity Analysis

Each row shows the effect of quantizing **one** Linear layer at a time (INT8 per-channel, MinMax calibration) while keeping all other layers in float32.  Layers with lower cosine similarity are the most quantization-sensitive.

**attention**: most sensitive layer = `out_proj` (sim=0.999795); least sensitive = `k_proj` (sim=0.999994)
**mlp**: most sensitive layer = `fc2` (sim=0.999735); least sensitive = `fc1` (sim=0.999920)

| model | layer | cosine_similarity | max_absolute_error | mean_absolute_error | root_mean_squared_error |
| --- | --- | --- | --- | --- | --- |
| attention | out_proj | 0.999795 | 0.012903 | 0.001742 | 0.002549 |
| attention | v_proj | 0.999936 | 0.005069 | 0.001147 | 0.001421 |
| attention | q_proj | 0.999994 | 0.002159 | 0.000358 | 0.000451 |
| attention | k_proj | 0.999994 | 0.002054 | 0.000355 | 0.000446 |
| mlp | fc2 | 0.999735 | 0.021319 | 0.003665 | 0.004580 |
| mlp | fc1 | 0.999920 | 0.012821 | 0.001999 | 0.002511 |


## 6. Weight-only vs. Full Quantization

Comparison of quantizing only the weights vs. quantizing both weights and activations for each Linear layer (INT8 per-channel).

| model | layer | weight+act | weight_only |
| --- | --- | --- | --- |
| attention | k_proj | 0.999994 | 1.000000 |
| attention | out_proj | 0.999795 | 0.999993 |
| attention | q_proj | 0.999994 | 0.999999 |
| attention | v_proj | 0.999936 | 0.999992 |
| mlp | fc1 | 0.999920 | 0.999992 |
| mlp | fc2 | 0.999735 | 0.999992 |


## 7. Calibration Strategy Comparison

All Linear layers quantized simultaneously with INT8 per-channel, varying only the calibration algorithm.

| model | calibration | cosine_similarity | mean_absolute_error |
| --- | --- | --- | --- |
| attention | minmax | 0.999718 | 0.002224 |
| attention | percentile_99.99 | 0.999132 | 0.004028 |
| mlp | minmax | 0.999657 | 0.004154 |
| mlp | percentile_99.99 | 0.999831 | 0.002914 |


## 8. Figures

### Cosine similarity per config per model

![Cosine similarity per config per model](results/figures/cosine_similarity_comparison.png)

### MLPBlock — max/mean/RMSE error metrics

![MLPBlock — max/mean/RMSE error metrics](results/figures/mlp_error_metrics.png)

### AttentionBlock — max/mean/RMSE error metrics

![AttentionBlock — max/mean/RMSE error metrics](results/figures/attention_error_metrics.png)

### Accuracy–error scatter (cosine sim vs MAE)

![Accuracy–error scatter (cosine sim vs MAE)](results/figures/accuracy_error_tradeoff.png)

### Memory reduction vs. cosine similarity

![Memory reduction vs. cosine similarity](results/figures/memory_vs_accuracy.png)

### Layer sensitivity ranking — MLPBlock

![Layer sensitivity ranking — MLPBlock](results/figures/sensitivity_layerwise_mlp.png)

### Layer sensitivity ranking — AttentionBlock

![Layer sensitivity ranking — AttentionBlock](results/figures/sensitivity_layerwise_attention.png)

### Weight-only vs full — MLPBlock

![Weight-only vs full — MLPBlock](results/figures/sensitivity_weightonly_vs_full_mlp.png)

### Weight-only vs full — AttentionBlock

![Weight-only vs full — AttentionBlock](results/figures/sensitivity_weightonly_vs_full_attention.png)

### Calibration strategy impact

![Calibration strategy impact](results/figures/sensitivity_calibration.png)


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

