<!--
 * @Author: wakakabaihua ford.du@foxmail.com
 * @Date: 2026-05-29 14:31:20
 * @LastEditors: wakakabaihua ford.du@foxmail.com
 * @LastEditTime: 2026-05-29 14:36:08
 * @FilePath: /quantization-analysis-lab/README_zh.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# 量化分析实验室

一个训练后量化（PTQ）分析框架，用于研究 Transformer 架构在低精度推理下的精度与效率权衡。

本项目有意从头实现量化逻辑，而非封装现有工具库。目标是让精度、准确率与内存之间的权衡关系清晰可见、可量化分析。

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

## 实验矩阵

| 配置 | 校准方式 | 量化粒度 | 对称性 | 模式 |
|------|----------|----------|--------|------|
| FP16 基线 | — | — | — | 参考基准 |
| INT8 | Min-max | 逐张量 | 是 | 权重 + 激活值 |
| INT8 | Min-max | 逐通道 | 是 | 权重 + 激活值 |
| INT8 | 百分位 | 逐张量 | 否 | 权重 + 激活值 |
| INT8 | 百分位 | 逐通道 | 否 | 权重 + 激活值 |
| INT4 仅权重 | 百分位 | 逐通道 | 是 | 仅权重 |

## 关键设计决策

- **伪量化**：所有量化均在 float32 中模拟量化效果，不使用整数计算内核，因此不测量实际运行时加速。这是有意为之的权衡，以保持分析代码的可移植性和可读性。
- **观测器与校准器分离**：统计信息收集与缩放因子计算解耦，使得不同的校准策略可以应用于相同的采集数据。
- **逐层敏感性分析**：每次实验均记录逐层误差贡献，便于识别瓶颈层。

## 主要结果

运行所有实验后，查看 `results/tables/summary.csv` 和 `docs/conclusion.md`。

## 文档

- [docs/design.md](docs/design.md) — 量化设计决策说明
- [docs/experiment_matrix.md](docs/experiment_matrix.md) — 完整实验计划
- [docs/conclusion.md](docs/conclusion.md) — 实验结论与权衡分析总结
