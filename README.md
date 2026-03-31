# LMCA-TIC

`LMCA-TIC` 是一个面向时序知识图谱归纳补全的科研工程仓库，覆盖以下代码链路：

- 本地 ICEWS 数据挂载与预处理
- 背景知识增强（BIE）提示构造
- `Qwen3-8B + LoRA/QLoRA` 文本语义注入
- `TCN` 关系时序建模
- 基于 `PyG` 风格接口的时间图编码
- 门控融合与双线性打分
- 基于 `KGIST` 摘要的困难负例扩展
- 训练、评测、绘图与实验汇总

## 最小工作流

```bash
python3 -m pip install -e ".[dev]"
lmca-tic preprocess --config configs/datasets/icews14.yaml
lmca-tic train --config configs/experiments/full_icews14.yaml
lmca-tic evaluate --config configs/experiments/full_icews14.yaml
lmca-tic run-suite --config configs/experiments/suite_icews14.yaml
```

## 目录

```text
src/lmca_tic
  config/       配置与 dataclass schema
  data/         数据契约、预处理与数据集
  kgist/        由 kgist.ipynb 重构得到的规则摘要与负误差模块
  models/       LLM、TCN、TGN、融合与打分
  training/     负例采样与训练器
  evaluation/   Filtered 指标与评测
  experiments/  实验编排与汇总
tests/          单元测试与 smoke 测试
configs/        数据集、模型和实验配置
```
