# ModelScope Experiment Commands

This repository is intended to run on ModelScope Notebook when the full ICEWS
datasets and local model weights are already available on the server.

Assume the repository root is the current working directory:

```bash
cd /mnt/workspace/LMCA-TIC
export PYTHONPATH=src
```

If the model weights are stored in a local ModelScope directory, set
`model.llm_name` in the corresponding YAML config to that local path, for
example `models/Qwen3-8B` or `/mnt/workspace/models/Qwen3-8B`.

## Standard Pipeline

```bash
python3 -m lmca_tic.cli build-bie --config configs/experiments/full_icews14.yaml
python3 -m lmca_tic.cli preprocess --config configs/experiments/full_icews14.yaml
python3 -m lmca_tic.cli train --config configs/experiments/full_icews14.yaml
python3 -m lmca_tic.cli evaluate --config configs/experiments/full_icews14.yaml --checkpoint best.pt --split test
```

## Multi-Seed Experiment Suites

Run the main two-dataset suite:

```bash
python3 -m lmca_tic.cli run-suite --suite main --seeds 42 123 456 789 1024 --output-root outputs/experiments/main
```

Run macro ablations, negative sampling variants, or TGN micro variants:

```bash
python3 -m lmca_tic.cli run-suite --suite ablation --seeds 42 123 456 789 1024 --output-root outputs/experiments/ablation
python3 -m lmca_tic.cli run-suite --suite negative --seeds 42 123 456 789 1024 --output-root outputs/experiments/negative
python3 -m lmca_tic.cli run-suite --suite micro --seeds 42 123 456 789 1024 --output-root outputs/experiments/micro
```

Each suite writes `per_seed_metrics.csv`, `summary.csv`, and `summary.json`.

## Sensitivity Experiments

Local history window width:

```bash
python3 -m lmca_tic.cli window-sensitivity --config configs/experiments/full_icews14.yaml --windows 3 7 14 21 30 --seeds 42 123 456 789 1024
```

Training data ratio. This creates derived raw splits under
`data/derived/train_ratio` and does not modify the original dataset:

```bash
python3 -m lmca_tic.cli train-ratio --config configs/experiments/full_icews14.yaml --ratios 0.2 0.4 0.6 0.8 1.0 --seeds 42 123 456 789 1024
```

Continuous local-history chain length. Run this after a checkpoint already
exists for the config:

```bash
python3 -m lmca_tic.cli history-chain --config configs/experiments/full_icews14.yaml --lengths 1 3 5 7 10 --checkpoint best.pt
```

Structural noise robustness. This creates derived processed copies and does not
modify the original processed dataset:

```bash
python3 -m lmca_tic.cli noise-sensitivity --config configs/experiments/full_icews14.yaml --rates 0.0 0.1 0.2 0.3 --checkpoint best.pt
```

## Subset Metrics

Inductive subset metrics from an existing prediction file:

```bash
python3 -m lmca_tic.cli eval-subset \
  --processed-dir data/processed/icews14 \
  --predictions outputs/full_icews14/test_predictions.jsonl \
  --subset inductive \
  --output outputs/full_icews14/test_inductive_metrics.json
```

Shot-limited inductive subset:

```bash
python3 -m lmca_tic.cli eval-subset \
  --processed-dir data/processed/icews14 \
  --predictions outputs/full_icews14/test_predictions.jsonl \
  --subset inductive \
  --history-shot 1 \
  --output outputs/full_icews14/test_inductive_1shot_metrics.json
```

## Merge External Baselines

External baselines can be kept in a CSV reference file and merged with generated
LMCA-TIC summaries:

```bash
python3 -m lmca_tic.cli aggregate-results \
  --reference references/table3_2_reference.csv \
  --summary outputs/experiments/main/summary.csv \
  --output outputs/experiments/main/merged_main_table.csv
```
