# Chapter 3 完整实验工作流（ModelScope）

适用于：

- ModelScope Notebook，单卡 A10 (24GB) 或多卡 A100 服务器
- 所有 LLM 权重已下载到本地 `models/<backbone-dir>/`
- ICEWS14 / ICEWS05-15 原始数据已挂载到 `data/local/<dataset>/raw/`
- 离线模式：`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1`

如果论文里所述硬件资源不够（A10 单卡跑不动 Qwen3-8B + 完整 ICEWS14），可以：

- 把 `model.use_4bit: true` + `micro_batch_size: 32 / gradient_accumulation_steps: 32` 维持等效 1024 的全局批次；
- 用 `SEEDS="42"` 单种子先跑一次跑通，再考虑全量 5 seeds；
- 骨干替换默认就是单 seed，论文里也是 "single representative run"。

## 0. 一次性准备

```bash
cd /mnt/workspace/LMCA-TIC
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=src
python3 -m pip install -e ".[dev,graph,qlora]"
```

`configs/experiments/backbone_*` 默认假设权重在 HF 仓库标识下；若是离线副本，请把 `model.llm_name` 改成本地目录，例如：

```yaml
model:
  llm_name: models/Qwen3-8B
```

## 1. BIE + 预处理

```bash
python3 -m lmca_tic.cli build-bie --config configs/experiments/full_icews14.yaml
python3 -m lmca_tic.cli build-bie --config configs/experiments/full_icews05_15.yaml
python3 -m lmca_tic.cli preprocess --config configs/experiments/full_icews14.yaml
python3 -m lmca_tic.cli preprocess --config configs/experiments/full_icews05_15.yaml
```

预处理产物在 `data/processed/<dataset>/{train,valid,test}.jsonl`，并写入 `entities.json` / `relations.json` / `filtered_targets.json` / `manifest.json` / `relation_frequency.json`。

## 2. 主实验（表 3-3）

```bash
python3 -m lmca_tic.cli run-suite \
  --suite main \
  --seeds 42 123 456 789 1024 \
  --output-root outputs/experiments/main
```

输出：
- `outputs/experiments/main/full_icews14/{seed_*/outputs}` 每个 seed 的 train_history、test_metrics、test_predictions
- `outputs/experiments/main/per_seed_metrics.csv` 每行 (experiment, dataset, seed, MRR, …)
- `outputs/experiments/main/summary.csv` 每实验的 mean / std

`references/table3_2_reference.csv` 仅作为公开榜单的对照表，论文表 3-3 的 8 个迁移基线数字需要作者按章节里"迁移到统一时间分割并重新调优"的口径单独跑各开源基线再填进 `references/table3_3_reference.csv`，本仓库不提供基线训练代码。

## 3. 归纳子集 + N-Shot（表 3-4 / 3-5）

主实验跑完后，直接基于 `test_predictions.jsonl` 重算子集：

```bash
python3 scripts/build_subset_metrics.py \
  --processed-dir data/processed/icews14 \
  --predictions outputs/experiments/main/full_icews14/seed_42/outputs/test_predictions.jsonl \
  --output-prefix outputs/experiments/subsets/lmca_tic_icews14
```

产出 `..._inductive.json` / `..._inductive_shot1.json` / `..._inductive_shot3.json` / `..._inductive_shot5.json` / `..._inductive_shot10.json`。

## 4. 数据规模、历史链、窗口（表 3-6 / 3-7 / 3-8）

```bash
python3 -m lmca_tic.cli train-ratio \
  --config configs/experiments/full_icews14.yaml \
  --ratios 0.2 0.4 0.6 0.8 1.0 \
  --seeds 42 123 456 789 1024 \
  --output-root outputs/experiments/train_ratio

python3 -m lmca_tic.cli history-chain \
  --config configs/experiments/full_icews14.yaml \
  --lengths 1 3 5 7 10 \
  --output-root outputs/experiments/history_chain

python3 -m lmca_tic.cli window-sensitivity \
  --config configs/experiments/full_icews14.yaml \
  --windows 3 7 14 21 30 \
  --seeds 42 123 456 789 1024 \
  --output-root outputs/experiments/window_sensitivity
```

`history-chain` 是评测期裁剪邻居数量来近似论文中的"连续历史链长度"；它必须在 `outputs/full_icews14/checkpoints/best.pt` 已经存在时才能运行。

## 5. 宏观 / 微观消融（表 3-9 / 3-12）

```bash
python3 -m lmca_tic.cli run-suite --suite ablation --seeds 42 123 456 789 1024 \
  --output-root outputs/experiments/ablation
python3 -m lmca_tic.cli run-suite --suite micro --seeds 42 123 456 789 1024 \
  --output-root outputs/experiments/micro
```

宏观消融 4 项：`ablation_wo_llm`、`ablation_wo_tgn`、`ablation_wo_temporal`、`ablation_wo_gate`。`ablation_wo_tcn` 只是 `wo_temporal` 的旧别名，保留兼容。

## 6. 噪声鲁棒（表 3-11）

```bash
python3 -m lmca_tic.cli noise-sensitivity \
  --config configs/experiments/full_icews14.yaml \
  --rates 0.0 0.1 0.2 0.3 \
  --output-root outputs/experiments/noise_sensitivity
```

LMCA-TIC 与 `wo_gate` 的对照可由 `outputs/experiments/main/full_icews14/...` 与 `outputs/experiments/ablation/ablation_wo_gate_icews14/...` 在相同噪声下重跑得到，把它们的 `test_metrics.json` 合并到一份对比表。

## 7. ρ 扫描（表 3-14 / 图 3-7）

```bash
python3 -m lmca_tic.cli rho-sensitivity \
  --config configs/experiments/full_icews14.yaml \
  --rhos 0.0 0.1 0.2 0.3 0.5 0.7 1.0 \
  --seeds 42 \
  --output-root outputs/experiments/rho_sensitivity

python3 scripts/plot_rho_sensitivity.py \
  --summary outputs/experiments/rho_sensitivity/summary.csv \
  --output outputs/experiments/figures/figure_3_7_rho.png
```

## 8. 三策略 × 主/归纳/1-Shot（表 3-15）

```bash
python3 -m lmca_tic.cli run-suite --suite negative --seeds 42 \
  --output-root outputs/experiments/negative

for variant in negative_random_icews14 negative_contrastive_icews14 negative_ontology_icews14; do
  python3 scripts/build_subset_metrics.py \
    --processed-dir data/processed/icews14 \
    --predictions outputs/experiments/negative/${variant}/seed_42/outputs/test_predictions.jsonl \
    --output-prefix outputs/experiments/subsets/${variant}
done
```

## 9. 骨干替换（表 3-10）

```bash
python3 -m lmca_tic.cli backbone-sensitivity --suite backbone_icews14 \
  --seeds 42 --output-root outputs/experiments/backbone_icews14
python3 -m lmca_tic.cli backbone-sensitivity --suite backbone_icews05_15 \
  --seeds 42 --output-root outputs/experiments/backbone_icews05_15
```

5 个骨干（`Qwen3-8B / LLaMA-3-8B / Mistral-7B / Qwen2.5-1.5B / DeBERTa-v3-large`）逐一训练并报告主/归纳/1-Shot 指标。

## 10. 图表与汇总

```bash
python3 scripts/plot_inductive_dumbbell.py \
  --metrics LMCA-TIC:1:outputs/experiments/subsets/lmca_tic_icews14_inductive_shot1.json \
            LMCA-TIC:10:outputs/experiments/subsets/lmca_tic_icews14_inductive_shot10.json \
            POSTRA:1:outputs/baselines/postra/inductive_shot1.json \
            POSTRA:10:outputs/baselines/postra/inductive_shot10.json \
  --output outputs/experiments/figures/figure_3_3.png

python3 scripts/plot_train_ratio_curves.py \
  --summary outputs/experiments/train_ratio/summary.csv \
  --labels LMCA-TIC \
  --output outputs/experiments/figures/figure_3_4.png

python3 scripts/plot_history_heatmap.py \
  --summary outputs/experiments/history_chain/summary.csv \
  --labels LMCA-TIC \
  --output outputs/experiments/figures/figure_3_5.png

python3 scripts/plot_compute_tradeoff.py \
  --variant V1:outputs/experiments/micro/micro_v1_gs_icews14/seed_42/outputs/test_metrics.json:outputs/experiments/micro/micro_v1_gs_icews14/seed_42/outputs/train_history.jsonl \
            V2:outputs/experiments/micro/micro_v2_ni_icews14/seed_42/outputs/test_metrics.json:outputs/experiments/micro/micro_v2_ni_icews14/seed_42/outputs/train_history.jsonl \
            V3:outputs/experiments/micro/micro_v3_sl_icews14/seed_42/outputs/test_metrics.json:outputs/experiments/micro/micro_v3_sl_icews14/seed_42/outputs/train_history.jsonl \
            V4:outputs/experiments/micro/micro_v4_gs_ni_sl_icews14/seed_42/outputs/test_metrics.json:outputs/experiments/micro/micro_v4_gs_ni_sl_icews14/seed_42/outputs/train_history.jsonl \
            Ours:outputs/experiments/micro/micro_ours_ni_sl_icews14/seed_42/outputs/test_metrics.json:outputs/experiments/micro/micro_ours_ni_sl_icews14/seed_42/outputs/train_history.jsonl \
  --output outputs/experiments/figures/figure_3_6.png

python3 scripts/aggregate_chapter3_tables.py \
  --main-per-seed outputs/experiments/main/per_seed_metrics.csv \
  --main-reference references/table3_3_reference.csv \
  --ablation-summary outputs/experiments/ablation/summary.csv \
  --negative-summary outputs/experiments/negative/summary.csv \
  --micro-summary outputs/experiments/micro/summary.csv \
  --rho-summary outputs/experiments/rho_sensitivity/summary.csv \
  --backbone-summary outputs/experiments/backbone_icews14/summary.csv \
  --window-summary outputs/experiments/window_sensitivity/summary.csv \
  --train-ratio-summary outputs/experiments/train_ratio/summary.csv \
  --history-chain-summary outputs/experiments/history_chain/summary.csv \
  --noise-summary outputs/experiments/noise_sensitivity/summary.csv \
  --output-dir outputs/experiments/chapter3_tables
```

## 11. 一键脚本

`scripts/run_chapter3_full_workflow.sh` 串起以上所有步骤。可以用 `SEEDS="42"` 单 seed 走通，再放到正式 5-seed 跑。
