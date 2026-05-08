#!/usr/bin/env bash
# Chapter 3 full experiment workflow for ModelScope A10/A100 servers.
#
# Assumptions:
#   - PWD is the repo root (e.g. /mnt/workspace/LMCA-TIC)
#   - data/local/icews14/raw and data/local/icews05_15/raw are populated
#   - All LLM weights live under models/<backbone-dir>/ (offline)
#   - HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 are exported beforehand
#
# Override SEEDS via env to shrink runtime, e.g. SEEDS="42" ./this.sh.

set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
SEEDS_DEFAULT="42 123 456 789 1024"
SEEDS="${SEEDS:-$SEEDS_DEFAULT}"
SINGLE_SEED="${SINGLE_SEED:-42}"

export PYTHONPATH="${PYTHONPATH:-src}"
cd "$ROOT_DIR"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

log "step 1: build offline BIE for both datasets"
python3 -m lmca_tic.cli build-bie --config configs/experiments/full_icews14.yaml
python3 -m lmca_tic.cli build-bie --config configs/experiments/full_icews05_15.yaml

log "step 2: preprocess"
python3 -m lmca_tic.cli preprocess --config configs/experiments/full_icews14.yaml
python3 -m lmca_tic.cli preprocess --config configs/experiments/full_icews05_15.yaml

log "step 3.1: main suite (table 3-3, 5 seeds × 2 datasets)"
python3 -m lmca_tic.cli run-suite \
  --suite main \
  --seeds $SEEDS \
  --output-root outputs/experiments/main

log "step 3.2: macro ablation (table 3-9)"
python3 -m lmca_tic.cli run-suite \
  --suite ablation \
  --seeds $SEEDS \
  --output-root outputs/experiments/ablation

log "step 3.3: negative-sampling variants (table 3-13)"
python3 -m lmca_tic.cli run-suite \
  --suite negative \
  --seeds $SEEDS \
  --output-root outputs/experiments/negative

log "step 3.4: micro ablation (table 3-12)"
python3 -m lmca_tic.cli run-suite \
  --suite micro \
  --seeds $SEEDS \
  --output-root outputs/experiments/micro

log "step 3.5: window sensitivity (table 3-8)"
python3 -m lmca_tic.cli window-sensitivity \
  --config configs/experiments/full_icews14.yaml \
  --windows 3 7 14 21 30 \
  --seeds $SEEDS \
  --output-root outputs/experiments/window_sensitivity

log "step 3.6: training-ratio sensitivity (table 3-6)"
python3 -m lmca_tic.cli train-ratio \
  --config configs/experiments/full_icews14.yaml \
  --ratios 0.2 0.4 0.6 0.8 1.0 \
  --seeds $SEEDS \
  --output-root outputs/experiments/train_ratio

log "step 3.7: history-chain sensitivity (table 3-7) -- requires existing checkpoint"
python3 -m lmca_tic.cli history-chain \
  --config configs/experiments/full_icews14.yaml \
  --lengths 1 3 5 7 10 \
  --output-root outputs/experiments/history_chain

log "step 3.8: structural noise sensitivity (table 3-11)"
python3 -m lmca_tic.cli noise-sensitivity \
  --config configs/experiments/full_icews14.yaml \
  --rates 0.0 0.1 0.2 0.3 \
  --output-root outputs/experiments/noise_sensitivity

log "step 3.9: rho sweep (table 3-14)"
python3 -m lmca_tic.cli rho-sensitivity \
  --config configs/experiments/full_icews14.yaml \
  --rhos 0.0 0.1 0.2 0.3 0.5 0.7 1.0 \
  --seeds $SINGLE_SEED \
  --output-root outputs/experiments/rho_sensitivity

log "step 3.10: backbone replacement (table 3-10) -- single seed per backbone"
python3 -m lmca_tic.cli backbone-sensitivity \
  --suite backbone_icews14 \
  --seeds $SINGLE_SEED \
  --output-root outputs/experiments/backbone_icews14
python3 -m lmca_tic.cli backbone-sensitivity \
  --suite backbone_icews05_15 \
  --seeds $SINGLE_SEED \
  --output-root outputs/experiments/backbone_icews05_15

log "step 4: derive inductive / N-shot subset metrics for table 3-4 / 3-5 / 3-15"
for predictions in \
  outputs/experiments/main/full_icews14/seed_${SINGLE_SEED}/outputs/test_predictions.jsonl \
  outputs/experiments/negative/negative_random_icews14/seed_${SINGLE_SEED}/outputs/test_predictions.jsonl \
  outputs/experiments/negative/negative_contrastive_icews14/seed_${SINGLE_SEED}/outputs/test_predictions.jsonl \
  outputs/experiments/negative/negative_ontology_icews14/seed_${SINGLE_SEED}/outputs/test_predictions.jsonl ; do
  if [[ -f "$predictions" ]]; then
    name=$(echo "$predictions" | sed 's|/|_|g' | sed 's|.jsonl$||')
    python3 scripts/build_subset_metrics.py \
      --processed-dir data/processed/icews14 \
      --predictions "$predictions" \
      --output-prefix "outputs/experiments/subsets/${name}"
  fi
done

log "step 5: aggregate result tables"
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

log "step 6: render figures (requires the per-baseline subset/summary files to exist)"
mkdir -p outputs/experiments/figures
python3 scripts/plot_train_ratio_curves.py \
  --summary outputs/experiments/train_ratio/summary.csv \
  --labels LMCA-TIC \
  --output outputs/experiments/figures/figure_3_4_train_ratio.png || true

python3 scripts/plot_rho_sensitivity.py \
  --summary outputs/experiments/rho_sensitivity/summary.csv \
  --output outputs/experiments/figures/figure_3_7_rho.png || true

log "chapter 3 workflow finished"
