

# 量化分析实验室

一个训练后量化（PTQ）分析框架，用于研究 Transformer 架构在低精度推理下的精度与效率权衡。

本项目有意从头实现量化逻辑，而非封装现有工具库。目标是让精度、准确率与内存之间的权衡关系清晰可见、可量化分析。

**状态**：全部六个实现阶段已完成，197 项测试全部通过。

## 本项目回答的问题

1. 不同量化策略如何影响模型精度和运行时特性？
2. 应如何在层级、张量级和校准级别做出量化决策？
3. 哪些层对量化最敏感，原因是什么？

## 项目结构

```text
quantization-analysis-lab/
  configs/          # 实验配置文件（YAML）
  scripts/          # 实验运行与分析脚本
  src/
    models/         # 基准参考模块（MLP、Attention）
    quant/          # 伪量化、观测器、校准器、PTQ 流水线
    utils/          # 指标计算与日志工具
  results/          # 实验输出（原始数据、图表、汇总表）
  tests/            # 核心量化原语的单元测试
  docs/             # 设计说明与实验分析文档
```

## 安装

```bash
pip install -r requirements.txt
```

## 运行实验

运行单个配置：

```bash
python scripts/run_experiments.py --config configs/ptq_int8_per_tensor.yaml
```

运行所有配置：

```bash
for cfg in configs/*.yaml; do
    python scripts/run_experiments.py --config "$cfg"
done
```

评估所有结果的精度：

```bash
python scripts/evaluate_accuracy.py --results_dir results/raw
```

生成汇总表：

```bash
python scripts/summarize_results.py \
    --results_dir results/raw \
    --output results/tables/summary.csv
```

生成权衡对比图：

```bash
python scripts/plot_tradeoffs.py \
    --results_dir results/raw \
    --output_dir results/figures
```

## 后端生态系统

已实现六个量化后端：

| 后端 | 技术 | 真实内核 | 测试数 |
|------|------|----------|--------|
| `FakeQuantBackend` | Observer → Calibrator → FakeQuantize 流水线 | 否（float32 模拟） | 41 |
| `TorchAOBackend` | `torch.ao.quantization.quantize_dynamic` | **是**（真实 INT8 内核） | 56 |
| `BitsAndBytesBackend` | LLM.int8() / NF4 (QLoRA) | **是**（int8/int4 CUDA 内核） | 80 |
| `GPTQBackend` | Hessian 引导的权重量化（INT4 打包） | **是**（INT4 打包存储，CUDA 上 Triton 解量化） | 136 |
| `AWQBackend` | 激活感知的权重缩放（INT4 打包） | **是**（INT4 打包存储，CUDA 上 Triton 解量化） | 136 |
| `MixedPrecisionBackend` | 逐层位宽分配 | 否（伪量化前向） | 170 |

**第六阶段（INT4 内核）**：新增 27 项测试覆盖 `pack_int4`、`unpack_int4`、`quantize_to_uint4`、`compute_groupwise_qparams`、Triton `int4_dequant_gemm`，以及与伪量化的端到端数值等价性验证。总计：**197 项测试**。

所有后端共享 `QuantBackend` 抽象基类（`calibrate` / `convert`）并注册到后端注册表中。

## 实验矩阵

核心配置（完整 YAML 文件见 `configs/`）：

| 配置 | 校准方式 | 量化粒度 | 对称性 | 模式 |
|------|----------|----------|--------|------|
| FP16 基线 | — | — | — | 参考基准 |
| INT8 逐张量对称 | Min-max | 逐张量 | 是 | 权重 + 激活值 |
| INT8 逐通道对称 | Min-max | 逐通道 | 是 | 权重 + 激活值 |
| INT8 逐张量非对称 | 百分位 | 逐张量 | 否 | 权重 + 激活值 |
| INT8 逐通道非对称 | 百分位 | 逐通道 | 否 | 权重 + 激活值 |
| INT4 仅权重 | 百分位 | 逐通道 | 是 | 仅权重 |
| 混合精度 INT8/INT4 | 敏感度引导 | 逐层 | 是 | 仅权重 |

## 关键设计决策

- **混合伪量化 + 真实内核**：
  - `TorchAOBackend` 和 `BitsAndBytesBackend` 使用真实 INT8/INT4 内核，实现真正的加速。
  - **第六阶段**：`GPTQBackend` 和 `AWQBackend` 现已改用 **INT4 打包存储格式**（8 个半字节打包为一个 int32，兼容 AutoGPTQ），并在 CUDA 上调度 **Triton 解量化内核**，CPU 上降级到 PyTorch 实现。INT8 及其它位宽为保持兼容性降级到 float32。
  - `FakeQuantBackend` 和 `MixedPrecisionBackend` 继续使用伪量化（float32 模拟），以最大化可移植性和算法透明性。
- **观测器与校准器分离**：统计信息收集与缩放因子计算解耦，使得不同的校准策略可以应用于相同的采集数据。
- **逐层敏感性分析**：每次实验均记录逐层误差贡献，便于识别瓶颈层。
- **敏感度引导的混合精度**：`MixedPrecisionBackend.from_sensitivity()` 根据量化敏感度分数自动分配逐层位宽。

## 主要结果

运行所有实验后，查看 `results/tables/summary.csv` 和 `docs/conclusion.md`。

## 文档

- [docs/design.md](docs/design.md) — 量化设计决策说明
- [docs/experiment_matrix.md](docs/experiment_matrix.md) — 完整实验计划
- [docs/conclusion.md](docs/conclusion.md) — 实验结论与权衡分析总结
