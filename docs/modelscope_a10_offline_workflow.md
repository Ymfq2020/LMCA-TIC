# ModelScope A10 离线闭环工作流

本文档只针对当前主线目标：

- 单卡 `A10 24GB`
- `ModelScope Notebook`
- 不能访问 Hugging Face 在线服务
- 优先完成可复现闭环、录屏展示与结果产物

本文档不覆盖 `Qwen3-8B + full ICEWS14` 正式主实验。该路线在当前 `A10` 条件下不具备工程可行性。

## 0. 前提

假设当前工作目录为：

```bash
cd /mnt/workspace/LMCA-TIC
```

假设本地离线模型目录已经准备好：

```text
/mnt/workspace/LMCA-TIC/models/Qwen2.5-0.5B-Instruct
```

假设正式原始数据已经准备好：

```text
/mnt/workspace/LMCA-TIC/data/local/icews14/raw/train.txt
/mnt/workspace/LMCA-TIC/data/local/icews14/raw/valid.txt
/mnt/workspace/LMCA-TIC/data/local/icews14/raw/test.txt
```

先强制离线模式：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

## 1. 切分两套可复现子数据

录屏版采用 `300/30/30`，展示版采用 `2000/100/100`。这里使用确定性的前缀切分，目标是保证流程可复现，不是追求最佳指标。

```bash
mkdir -p data/local/icews14_record_small/raw data/local/icews14_record_small/bie
mkdir -p data/local/icews14_demo_plus/raw data/local/icews14_demo_plus/bie
```

```bash
sed -n '1,300p' data/local/icews14/raw/train.txt > data/local/icews14_record_small/raw/train.txt
sed -n '1,30p' data/local/icews14/raw/valid.txt > data/local/icews14_record_small/raw/valid.txt
sed -n '1,30p' data/local/icews14/raw/test.txt > data/local/icews14_record_small/raw/test.txt
```

```bash
sed -n '1,2000p' data/local/icews14/raw/train.txt > data/local/icews14_demo_plus/raw/train.txt
sed -n '1,100p' data/local/icews14/raw/valid.txt > data/local/icews14_demo_plus/raw/valid.txt
sed -n '1,100p' data/local/icews14/raw/test.txt > data/local/icews14_demo_plus/raw/test.txt
```

核对行数：

```bash
wc -l \
  data/local/icews14_record_small/raw/train.txt \
  data/local/icews14_record_small/raw/valid.txt \
  data/local/icews14_record_small/raw/test.txt \
  data/local/icews14_demo_plus/raw/train.txt \
  data/local/icews14_demo_plus/raw/valid.txt \
  data/local/icews14_demo_plus/raw/test.txt
```

## 2. 构建离线 BIE

录屏版：

```bash
PYTHONPATH=src python3 -m lmca_tic.cli build-bie \
  --config configs/experiments/icews14_record_qwen25_05b_a10.yaml
```

展示版：

```bash
PYTHONPATH=src python3 -m lmca_tic.cli build-bie \
  --config configs/experiments/icews14_demo_plus_qwen25_05b_a10.yaml
```

构建完成后，应看到：

```text
data/local/icews14_record_small/bie/entity_metadata.jsonl
data/local/icews14_demo_plus/bie/entity_metadata.jsonl
```

## 3. 预处理

录屏版：

```bash
PYTHONPATH=src python3 -m lmca_tic.cli preprocess \
  --config configs/experiments/icews14_record_qwen25_05b_a10.yaml
```

展示版：

```bash
PYTHONPATH=src python3 -m lmca_tic.cli preprocess \
  --config configs/experiments/icews14_demo_plus_qwen25_05b_a10.yaml
```

构建完成后，检查：

```bash
ls -lah data/processed/icews14_record_small
ls -lah data/processed/icews14_demo_plus
```

关键文件包括：

- `train.jsonl`
- `valid.jsonl`
- `test.jsonl`
- `entities.json`
- `relations.json`
- `filtered_targets.json`
- `manifest.json`

## 4. 训练

只允许同一时刻运行一个训练进程，不要并发启动多个 `train`。

录屏版：

```bash
PYTHONPATH=src python3 -m lmca_tic.cli train \
  --config configs/experiments/icews14_record_qwen25_05b_a10.yaml
```

展示版：

```bash
PYTHONPATH=src python3 -m lmca_tic.cli train \
  --config configs/experiments/icews14_demo_plus_qwen25_05b_a10.yaml
```

说明：当前实现中，`train` 命令会在进入训练前再次执行一次 `preprocess`。因此若你已经单独跑过预处理，这里的重复构建属于当前设计行为，不是故障。

训练过程中应直接在当前窗口看到：

- `preprocess` 进度
- `epoch` 级训练进度条
- `loss / lr / opt_steps / step_s / bps / eta`
- GPU 诊断日志

不要使用 `Ctrl+Z` 挂起进程。若需要终止，直接 `Ctrl+C`。

## 5. 评测

训练完成后，如需单独重跑评测：

录屏版：

```bash
PYTHONPATH=src python3 -m lmca_tic.cli evaluate \
  --config configs/experiments/icews14_record_qwen25_05b_a10.yaml \
  --checkpoint best.pt \
  --split test
```

展示版：

```bash
PYTHONPATH=src python3 -m lmca_tic.cli evaluate \
  --config configs/experiments/icews14_demo_plus_qwen25_05b_a10.yaml \
  --checkpoint best.pt \
  --split test
```

## 6. 绘图

录屏版：

```bash
python3 scripts/plot_training_curves.py \
  --history outputs/icews14_record_qwen25_05b_a10/train_history.jsonl \
  --output outputs/icews14_record_qwen25_05b_a10/training_curves.png
```

展示版：

```bash
python3 scripts/plot_training_curves.py \
  --history outputs/icews14_demo_plus_qwen25_05b_a10/train_history.jsonl \
  --output outputs/icews14_demo_plus_qwen25_05b_a10/training_curves.png
```

## 7. 录屏建议展示内容

建议按如下顺序录制：

1. 展示仓库结构与两套配置文件。
2. 展示 `data/local/icews14_record_small/raw` 与 `data/local/icews14_record_small/bie`。
3. 运行 `train`，停留在实时进度条和 GPU 日志界面。
4. 展示 `outputs/icews14_record_qwen25_05b_a10/test_metrics.json`。
5. 展示 `outputs/icews14_record_qwen25_05b_a10/training_curves.png`。
6. 展示 `outputs/icews14_record_qwen25_05b_a10/graph_summary.json`。

## 8. 结果解释边界

若录屏版指标较低，解释口径应固定为：

- 当前运行的是缩小规模的离线闭环演示实验。
- 目标是证明从数据、BIE、训练、评测到结果产出的完整可复现链路。
- 指标不作为论文正式主实验结果。
- 若要接近论文主实验，应切换更强机器或继续优化训练器。
