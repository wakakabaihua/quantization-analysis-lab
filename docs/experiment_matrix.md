# Experiment Matrix

## Configuration Set

| Config file | Name | Dtype | Granularity | Symmetric | Weight Only | Calibration |
|-------------|------|-------|-------------|-----------|-------------|-------------|
| ptq_fp16.yaml | fp16_baseline | FP16 | — | — | — | — |
| ptq_int8_per_tensor.yaml | int8_per_tensor_sym | INT8 | Per-tensor | Yes | No | MinMax |
| ptq_int8_per_channel.yaml | int8_per_channel_sym | INT8 | Per-channel | Yes | No | MinMax |
| ptq_int8_per_tensor_asym.yaml | int8_per_tensor_asym | INT8 | Per-tensor | No | No | Percentile 99.99% |
| ptq_int8_per_channel_asym.yaml | int8_per_channel_asym | INT8 | Per-channel | No | No | Percentile 99.99% |
| ptq_int4_exploration.yaml | int4_weight_only | INT4 | Per-channel | Yes | Yes | Percentile 99.9% |

## Models

| Model | Structure | Linear layers | Experimental purpose |
|-------|-----------|---------------|----------------------|
| MLPBlock | Linear(512→2048) + GELU + Linear(2048→512) | fc1, fc2 | Isolate FFN quantization |
| AttentionBlock | q_proj, k_proj, v_proj + SDPA + out_proj | 4 Linear | Isolate attention projection quantization |

## Experiment Cross-Product

For each config × each model:
- Calibrate with 128 synthetic samples (fixed seed 42)
- Evaluate on 1 test batch (fixed seed 0)
- Log: cosine similarity, max/mean/RMSE absolute error, relative error
- Log: layer-wise error summary (mean, max, min across layers)

## Results Table

To populate: run `for cfg in configs/*.yaml; do python scripts/run_experiments.py --config $cfg; done`
then `python scripts/summarize_results.py`.

| Config | Model | Cosine Sim | Max Abs Err | Mean Abs Err | RMSE |
|--------|-------|:----------:|:-----------:|:------------:|:----:|
| fp16_baseline | mlp | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| fp16_baseline | attention | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| int8_per_tensor_sym | mlp | | | | |
| int8_per_tensor_sym | attention | | | | |
| int8_per_channel_sym | mlp | | | | |
| int8_per_channel_sym | attention | | | | |
| int8_per_tensor_asym | mlp | | | | |
| int8_per_tensor_asym | attention | | | | |
| int8_per_channel_asym | mlp | | | | |
| int8_per_channel_asym | attention | | | | |
| int4_weight_only | mlp | | | | |
| int4_weight_only | attention | | | | |

## Hypotheses to Test

1. **Per-channel > Per-tensor**: Cosine similarity should be higher for
   per-channel in both symmetric and asymmetric modes because per-channel
   compensates for inter-channel weight variance.

2. **Asymmetric > Symmetric for activations**: Post-GELU activations are
   non-zero-centered; asymmetric quantization should allocate the unsigned
   range more efficiently.

3. **INT4 error >> INT8 error**: Each additional bit halves the quantization
   step size, so INT4 error should be ~16× larger than INT8 error at the
   same granularity.

4. **Attention more sensitive than MLP**: Attention weights are smaller and
   the softmax-normalized attention scores are sensitive to small numerical
   perturbations.

5. **MinMax worse than Percentile when outliers present**: A single extreme
   weight value can inflate the MinMax range; percentile clipping should
   produce a more concentrated scale.
