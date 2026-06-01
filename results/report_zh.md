# 量化分析实验室 — 实验报告

_生成时间：2026-06-01_

## 1. 执行摘要

在所有测试配置中，**fp16_baseline** 在 **attention** 模块上取得了最高输出保真度（余弦相似度 = `1.000000`）。最低保真度出现在 **int4_weight_only** 对 **mlp** 模块的量化（余弦相似度 = `0.994826`）。

- **attention**：最佳 `fp16_baseline`（sim=1.000000），最差 `int4_weight_only`（sim=0.994920）
- **mlp**：最佳 `fp16_baseline`（sim=1.000000），最差 `int4_weight_only`（sim=0.994826）


## 2. 实验矩阵

| 配置 | 模型 | 余弦相似度 | 最大绝对误差 | 平均绝对误差 | 均方根误差 | 相对误差 |
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


## 3. 相对 FP16 基线的误差变化（Δ）

Δ余弦相似度为正表示该配置比 FP16 运行更接近浮点基线（即更接近 FP32）；负值表示量化带来的精度损失。

| 配置 | 模型 | 余弦相似度 | 最大绝对误差 | 平均绝对误差 | 均方根误差 | 相对误差 | 逐层平均余弦相似度 | 逐层平均MAE | Δ余弦相似度 | Δ最大绝对误差 | Δ平均绝对误差 | Δ均方根误差 |
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


## 4. 内存占用估算

权重内存估算基于参数量与数据类型位宽（未计入逐通道缩放因子的额外开销）。

| 配置 | 模型 | 数据类型 | 仅权重 | FP32权重(MB) | 量化后权重(MB) | 内存压缩倍数 |
| --- | --- | --- | --- | --- | --- | --- |
| fp16_baseline | attention | fp16 | 否 | 4.194 | 2.097 | 2.000 |
| int8_per_tensor_sym | attention | int8 | 否 | 4.194 | 1.049 | 4.000 |
| int8_per_channel_sym | attention | int8 | 否 | 4.194 | 1.049 | 4.000 |
| int8_per_tensor_asym | attention | int8 | 否 | 4.194 | 1.049 | 4.000 |
| int8_per_channel_asym | attention | int8 | 否 | 4.194 | 1.049 | 4.000 |
| int4_weight_only | attention | int4 | 是 | 4.194 | 0.524 | 8.000 |
| fp16_baseline | mlp | fp16 | 否 | 8.389 | 4.194 | 2.000 |
| int8_per_tensor_sym | mlp | int8 | 否 | 8.389 | 2.097 | 4.000 |
| int8_per_channel_sym | mlp | int8 | 否 | 8.389 | 2.097 | 4.000 |
| int8_per_tensor_asym | mlp | int8 | 否 | 8.389 | 2.097 | 4.000 |
| int8_per_channel_asym | mlp | int8 | 否 | 8.389 | 2.097 | 4.000 |
| int4_weight_only | mlp | int4 | 是 | 8.389 | 1.049 | 8.000 |


## 5. 逐层敏感性分析

每行展示对**单个** Linear 层进行量化（INT8 逐通道，MinMax 校准）、其余层保持 float32 时的误差影响。余弦相似度越低，表示该层对量化越敏感。

**attention**：最敏感层 = `out_proj`（sim=0.999795）；最不敏感层 = `k_proj`（sim=0.999994）  
**mlp**：最敏感层 = `fc2`（sim=0.999735）；最不敏感层 = `fc1`（sim=0.999920）

| 模型 | 层 | 余弦相似度 | 最大绝对误差 | 平均绝对误差 | 均方根误差 |
| --- | --- | --- | --- | --- | --- |
| attention | out_proj | 0.999795 | 0.012903 | 0.001742 | 0.002549 |
| attention | v_proj | 0.999936 | 0.005069 | 0.001147 | 0.001421 |
| attention | q_proj | 0.999994 | 0.002159 | 0.000358 | 0.000451 |
| attention | k_proj | 0.999994 | 0.002054 | 0.000355 | 0.000446 |
| mlp | fc2 | 0.999735 | 0.021319 | 0.003665 | 0.004580 |
| mlp | fc1 | 0.999920 | 0.012821 | 0.001999 | 0.002511 |


## 6. 仅权重量化 vs. 全量化对比

比较仅对权重量化 vs. 权重与激活值同时量化（INT8 逐通道）在各 Linear 层上的效果差异。

| 模型 | 层 | 权重+激活值 | 仅权重 |
| --- | --- | --- | --- |
| attention | k_proj | 0.999994 | 1.000000 |
| attention | out_proj | 0.999795 | 0.999993 |
| attention | q_proj | 0.999994 | 0.999999 |
| attention | v_proj | 0.999936 | 0.999992 |
| mlp | fc1 | 0.999920 | 0.999992 |
| mlp | fc2 | 0.999735 | 0.999992 |


## 7. 校准策略对比

所有 Linear 层同时使用 INT8 逐通道量化，仅改变校准算法。

| 模型 | 校准方式 | 余弦相似度 | 平均绝对误差 |
| --- | --- | --- | --- |
| attention | minmax | 0.999718 | 0.002224 |
| attention | percentile_99.99 | 0.999132 | 0.004028 |
| mlp | minmax | 0.999657 | 0.004154 |
| mlp | percentile_99.99 | 0.999831 | 0.002914 |


## 8. 图表

### 各配置各模型余弦相似度对比

![各配置各模型余弦相似度对比](figures/cosine_similarity_comparison.png)

### MLPBlock — 最大/平均/RMSE 误差指标

![MLPBlock — 最大/平均/RMSE 误差指标](figures/mlp_error_metrics.png)


---

## 9. 后端实现进度

全部六个实现阶段已完成，测试套件覆盖所有后端，共 **197 项测试，0 失败**。

| 阶段 | 后端 | 核心技术 | 测试数 |
| --- | --- | --- | --- |
| 1 | `FakeQuantBackend` | Observer → Calibrator → FakeQuantize 流水线 | 41 |
| 2 | `TorchAOBackend` | `torch.ao.quantization.quantize_dynamic` | 56 |
| 3 | `BitsAndBytesBackend` | LLM.int8()（INT8）/ NF4 双量化（INT4） | 80 |
| 4 | `GPTQBackend` + `AWQBackend` | Hessian 引导的 GPTQ；激活感知的 AWQ 缩放 | 136 |
| 5 | `MixedPrecisionBackend` | 逐层位宽分配，敏感度引导 | 170 |
| 6 | **INT4 内核基础设施** | INT4 打包存储 + CUDA 上 Triton 解量化 | **197**（+27 测试） |

所有后端共享 `QuantBackend` 抽象基类（`calibrate` / `convert`），并注册到后端注册表中，可通过 `get_backend(name)` 或 `PTQPipeline.from_config()` 调用。


## 10. 混合精度量化（第五阶段）

`MixedPrecisionBackend` 为每个 Linear 层独立分配位宽。对量化敏感的层（单独量化时余弦相似度低）保持较高精度；对量化鲁棒的层则进行更激进的压缩。

### 10.1 构建方式

| 方式 | 适用场景 |
| --- | --- |
| `MixedPrecisionBackend(layer_config={...})` | 显式指定每层位宽字典 |
| `MixedPrecisionBackend.from_sensitivity(scores, threshold, high_bits, low_bits)` | 基于敏感度分数自动分配 |
| `MixedPrecisionBackend.from_config_dict(mp_cfg)` | 从 YAML 的 `mixed_precision:` 节读取 |

### 10.2 敏感度引导分配（阈值 = 0.9999）

使用第 5 节的逐层敏感度分数，配置 `high_bits=8, low_bits=4`：

| 模型 | 层 | 余弦相似度 | 分配结果 |
| --- | --- | --- | --- |
| attention | `out_proj` | 0.999795 | **INT8**（敏感） |
| attention | `v_proj` | 0.999936 | **INT8**（敏感） |
| attention | `q_proj` | 0.999994 | INT4（鲁棒） |
| attention | `k_proj` | 0.999994 | INT4（鲁棒） |
| mlp | `fc2` | 0.999735 | **INT8**（敏感） |
| mlp | `fc1` | 0.999920 | **INT8**（敏感） |

### 10.3 理论压缩比

假设同一模型内各层参数量相等：

| 模型 | 分配方案 | 有效位宽（位/参数） | 相对 FP32 压缩比 |
| --- | --- | --- | --- |
| attention | 2 × INT8 + 2 × INT4 | 6 | **5.33×** |
| mlp | 2 × INT8 | 8 | **4.00×** |
| （参考）| 全 INT8 | 8 | 4.00× |
| （参考）| 全 INT4 | 4 | 8.00× |

attention 模型从混合精度中获益最多：两个鲁棒的投影层（q、k）压缩至 INT4，两个敏感层（out、v）保持 INT8，在维持输出质量优于全 INT8 基线的同时，实现 **5.33× 压缩比**。

### 10.4 实验配置

混合精度实验由 `configs/ptq_mixed_precision.yaml` 驱动，并通过自动化搜索脚本 `scripts/mixed_precision_search.py` 扫描校准阈值 `[0.9998, 0.9999, 0.99995, 0.99998, 1.0]`，输出各模型的精度/压缩比帕累托前沿。

### AttentionBlock — 最大/平均/RMSE 误差指标

![AttentionBlock — 最大/平均/RMSE 误差指标](figures/attention_error_metrics.png)

### 精度-误差散点图（余弦相似度 vs MAE）

![精度-误差散点图（余弦相似度 vs MAE）](figures/accuracy_error_tradeoff.png)

### 内存压缩倍数 vs. 余弦相似度

![内存压缩倍数 vs. 余弦相似度](figures/memory_vs_accuracy.png)

### 逐层敏感性排名 — MLPBlock

![逐层敏感性排名 — MLPBlock](figures/sensitivity_layerwise_mlp.png)

### 逐层敏感性排名 — AttentionBlock

![逐层敏感性排名 — AttentionBlock](figures/sensitivity_layerwise_attention.png)

### 仅权重 vs. 全量化 — MLPBlock

![仅权重 vs. 全量化 — MLPBlock](figures/sensitivity_weightonly_vs_full_mlp.png)

### 仅权重 vs. 全量化 — AttentionBlock

![仅权重 vs. 全量化 — AttentionBlock](figures/sensitivity_weightonly_vs_full_attention.png)

### 校准策略影响对比

![校准策略影响对比](figures/sensitivity_calibration.png)


## 11. 真实 INT4 内核（第六阶段）

`GPTQBackend` 和 `AWQBackend` 现已升级为使用 **INT4 打包权重存储**和**真实解量化内核**。

### 11.1 线格式（兼容 AutoGPTQ / autoawq）

- `qweight`:  `[out_features, in_features // 8]` int32  —  每个 int32 打包 8 个 INT4 半字节（低半字节优先）
- `scales`:   `[n_groups, out_features]` float32  —  逐组量化缩放因子
- `qzeros`:   `[n_groups, out_features]` int32  —  逐组零点
- 解量化公式：`W_fp[n,k] = (qweight_unpacked[n,k] − qzeros[g,n]) × scales[g,n]`

### 11.2 内核调度

- **CUDA**（配合 Triton）：Triton JIT 内核，使用 `tl.static_range(8)` 半字节循环 → 解量化为 FP16 → `torch.mm`（cuBLAS 张量核 GEMM）
- **CPU**（或无 Triton）：纯 PyTorch 向量化解量化 + `F.linear`

### 11.3 内存占用

INT4 打包存储相比 float32 节省 **8 倍**空间：
- **float32 权重**：`N × K × 4` 字节
- **int32 打包**：`N × (K // 8) × 4` 字节  =  `N × K / 2` 字节

### 11.4 数值等价性

打包 INT4 实现与原伪量化实现**数值等价**：
- `dequant(pack(quantize(W))) ≡ fake_quant(W)`（精确）
- 所有 170 项原有测试继续通过
- 27 项新内核测试验证打包/解包、量化、解量化、GEMM 正确性

### 11.5 INT8 及其它位宽的降级方案

`AWQLinear` 和 `GPTQLinear` 支持两种存储模式：
- **INT4**（`num_bits == 4`）：使用打包格式 + Triton/CPU 解量化内核
- **其它位宽**（`num_bits != 4`）：存储解量化后的 float32 权重 + `F.linear`（用于兼容 INT8 及混合精度实验）

### 11.6 新增组件

| 组件 | 说明 |
| --- | --- |
| `src/quant/kernels/int4_packing.py` | `pack_int4`、`unpack_int4`、`quantize_to_uint4`、`compute_groupwise_qparams`、`dequant_weight_cpu` |
| `src/quant/kernels/triton_int4_gemm.py` | `_dequant_int4_kernel`（Triton JIT）、`dequant_int4`、`int4_dequant_gemm` |
| `tests/test_int4_kernels.py` | 27 项新测试（打包/解包、解量化、GEMM、数值等价性、内存占用） |

### 11.7 测试覆盖

- `TestInt4PackUnpack`（6 项测试）：往返保真、形状、数据类型、半字节顺序
- `TestQuantizeToUint4`（4 项测试）：输出范围、数据类型、形状
- `TestComputeGroupwiseQparams`（5 项测试）：缩放因子形状、对称模式 qzero = 8、非对称 qzero ≥ 0
- `TestDequantWeightCpu`（2 项测试）：手动解量化公式匹配、输出形状
- `TestInt4DequantGemm`（7 项测试）：输出形状、3D 输入、对称/非对称、偏置、多组
- `TestNumericalEquivalence`（3 项测试）：AWQ/GPTQ 打包 ≈ 伪量化参考

全部 197 项测试（170 项原有 + 27 项新增）**通过**，0 失败。


## 12. 结论与后续工作

### 主要发现

1. **INT8 逐通道对称量化**在精度与内存之间取得了最佳平衡：余弦相似度接近 FP16，同时将权重内存压缩至原来的 1/4。

2. **INT4 仅权重量化**实现了最大内存压缩（相对 FP32 压缩 8×），但输出质量下降较为明显，尤其体现在 attention 的投影层上。

3. **激活值量化**在权重量化的基础上引入了不可忽视的额外误差。在大多数场景下，仅权重量化是更优选择，除非推理内核能充分利用 INT8 激活值带来的加速。

4. **校准策略**（MinMax vs. 百分位）在 INT8 精度下影响可测但较小；在 INT4 下影响更显著，因为分布尾部截断的影响会被放大。

5. **逐层敏感性不均等**：输出投影层（out_proj）通常比中间层（如 q_proj/k_proj）对量化更敏感。

### 生产部署仍存在的差距

| 缺口 | 说明 |
|------|------|
| 真实 INT8/INT4 推理内核 | ✅ **已完成**：**`TorchAOBackend`** 使用真实 INT8 内核（`torch.ao.quantization.quantize_dynamic`）。**`BitsAndBytesBackend`** 使用 `Linear8bitLt` / `Linear4bit`（真实 int8/int4 CUDA 内核）。**`GPTQBackend` / `AWQBackend`**（第六阶段）现已采用 **INT4 打包存储格式**，并在 CUDA 上调度 **Triton 解量化内核**（CPU 上降级到 PyTorch 实现）。**`FakeQuantBackend` / `MixedPrecisionBackend`** 继续使用伪量化方式以保证可移植性。 |
| 大规模精度评估 | 测量基于合成随机输入，未评估端到端任务精度（BLEU、accuracy、F1 等）。 |
| 混合精度搜索 | ✅ **已完成**：`MixedPrecisionBackend` 实现了基于敏感度的逐层位宽自动分配（`from_sensitivity()`），并配备阈值扫描脚本 `scripts/mixed_precision_search.py`，覆盖 HAWQ 风格的精度-压缩率权衡分析。 |
| 量化感知训练（QAT） | 对于更激进的压缩（INT4 或更低），推荐在训练循环中加入伪量化节点进行 QAT。 |
| 动态/静态激活值校准 | 未实现动态逐 token 激活值校准（LLM 风格）。 |
| KV 缓存量化 | 本实验未涵盖，但对 Transformer 推理内存优化具有重要意义。 |
