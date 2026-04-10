#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${1:-/mnt/workspace/LMCA-TIC}"

source "$SCRIPT_DIR/common.sh" "$WORKSPACE"
prepare_env

log "plot training curves"
PYTHONPATH=src python3 scripts/plot_training_curves.py \
  --history outputs/icews14_record_qwen25_05b_a10/train_history.jsonl \
  --output outputs/icews14_record_qwen25_05b_a10/training_curves.png

require_path "$OUTPUT_DIR/training_curves.png"
ls -lah "$OUTPUT_DIR/training_curves.png"
