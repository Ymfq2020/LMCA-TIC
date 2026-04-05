#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${1:-/mnt/workspace/LMCA-TIC}"

source "$SCRIPT_DIR/common.sh" "$WORKSPACE"
prepare_env

log "evaluate best checkpoint on test split"
PYTHONPATH=src python3 -m lmca_tic.cli evaluate \
  --config configs/experiments/icews14_record_qwen25_05b_a10.yaml \
  --checkpoint best.pt \
  --split test

require_path "$OUTPUT_DIR/test_metrics.json"
cat "$OUTPUT_DIR/test_metrics.json"
