#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE="${1:-/mnt/workspace/LMCA-TIC}"

FULL_RAW_DIR="$WORKSPACE/data/local/icews14/raw"
RECORD_RAW_DIR="$WORKSPACE/data/local/icews14_record_small/raw"
RECORD_BIE_DIR="$WORKSPACE/data/local/icews14_record_small/bie"
PROCESSED_DIR="$WORKSPACE/data/processed/icews14_record_small"
OUTPUT_DIR="$WORKSPACE/outputs/icews14_record_qwen25_05b_a10"
MODEL_DIR="$WORKSPACE/models/Qwen2.5-0.5B-Instruct"
CONFIG_PATH="$WORKSPACE/configs/experiments/icews14_record_qwen25_05b_a10.yaml"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    printf 'ERROR: required path not found: %s\n' "$path" >&2
    exit 1
  fi
}

run() {
  log "$*"
  bash -lc "$*"
}

log "workspace=$WORKSPACE"
require_path "$WORKSPACE"
require_path "$FULL_RAW_DIR/train.txt"
require_path "$FULL_RAW_DIR/valid.txt"
require_path "$FULL_RAW_DIR/test.txt"
require_path "$MODEL_DIR"
require_path "$CONFIG_PATH"

cd "$WORKSPACE"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

log "offline mode enabled"
log "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
log "TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
log "HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"

log "step 1/7: split ICEWS14 record subset (300/30/30)"
mkdir -p "$RECORD_RAW_DIR" "$RECORD_BIE_DIR"
sed -n '1,300p' "$FULL_RAW_DIR/train.txt" > "$RECORD_RAW_DIR/train.txt"
sed -n '1,30p' "$FULL_RAW_DIR/valid.txt" > "$RECORD_RAW_DIR/valid.txt"
sed -n '1,30p' "$FULL_RAW_DIR/test.txt" > "$RECORD_RAW_DIR/test.txt"
wc -l \
  "$RECORD_RAW_DIR/train.txt" \
  "$RECORD_RAW_DIR/valid.txt" \
  "$RECORD_RAW_DIR/test.txt"

log "step 2/7: build offline BIE"
run "PYTHONPATH=src python3 -m lmca_tic.cli build-bie --config configs/experiments/icews14_record_qwen25_05b_a10.yaml"

log "step 3/7: preprocess dataset"
run "PYTHONPATH=src python3 -m lmca_tic.cli preprocess --config configs/experiments/icews14_record_qwen25_05b_a10.yaml"

log "step 4/7: train"
run "PYTHONPATH=src python3 -m lmca_tic.cli train --config configs/experiments/icews14_record_qwen25_05b_a10.yaml"

log "step 5/7: evaluate test split"
run "PYTHONPATH=src python3 -m lmca_tic.cli evaluate --config configs/experiments/icews14_record_qwen25_05b_a10.yaml --checkpoint best.pt --split test"

log "step 6/7: plot training curves"
run "PYTHONPATH=src python3 scripts/plot_training_curves.py --history outputs/icews14_record_qwen25_05b_a10/train_history.jsonl --output outputs/icews14_record_qwen25_05b_a10/training_curves.png"

log "step 7/7: show key artifacts"
require_path "$PROCESSED_DIR/manifest.json"
require_path "$OUTPUT_DIR/test_metrics.json"
require_path "$OUTPUT_DIR/graph_summary.json"
require_path "$OUTPUT_DIR/training_curves.png"

ls -lah "$PROCESSED_DIR"
ls -lah "$OUTPUT_DIR"

log "test metrics"
python3 - <<'PY'
import json
from pathlib import Path

path = Path("outputs/icews14_record_qwen25_05b_a10/test_metrics.json")
print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, ensure_ascii=False))
PY

log "workflow completed"
