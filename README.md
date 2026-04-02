# LMCA-TIC

`LMCA-TIC` 是一个面向时序知识图谱归纳补全的科研工程仓库，覆盖以下代码链路：

- 本地 ICEWS 数据挂载与预处理
- 基于图谱内部统计与规则归纳的背景知识增强（离线 BIE）
- `Qwen3-8B + LoRA/QLoRA` 文本语义注入
- `TCN` 关系时序建模
- 基于 `PyG` 风格接口的时间图编码
- 门控融合与双线性打分
- 基于 `KGIST` 摘要的困难负例扩展
- 训练、评测、绘图与实验汇总

## A10 单卡推荐入口

当前主线不再推荐在 `A10 24GB` 上直接尝试 `Qwen3-8B + full ICEWS14`。若目标是先完成离线闭环、录屏展示与可复现实验流程，优先使用：

- `configs/experiments/icews14_record_qwen25_05b_a10.yaml`
- `configs/experiments/icews14_demo_plus_qwen25_05b_a10.yaml`

完整的离线命令链路、数据切分、BIE 构建、训练、评测、绘图与录屏展示说明见：

- `docs/modelscope_a10_offline_workflow.md`

## 最小工作流

```bash
python3 -m pip install -e ".[dev]"
PYTHONPATH=src python3 -m lmca_tic.cli build-bie --config configs/experiments/icews14_record_qwen25_05b_a10.yaml
PYTHONPATH=src python3 -m lmca_tic.cli preprocess --config configs/experiments/icews14_record_qwen25_05b_a10.yaml
PYTHONPATH=src python3 -m lmca_tic.cli train --config configs/experiments/icews14_record_qwen25_05b_a10.yaml
PYTHONPATH=src python3 -m lmca_tic.cli evaluate --config configs/experiments/icews14_record_qwen25_05b_a10.yaml
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
docs/           离线部署与实验工作流说明
```
