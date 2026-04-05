#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${1:-/mnt/workspace/LMCA-TIC}"

source "$SCRIPT_DIR/common.sh" "$WORKSPACE"
prepare_env

log "show key artifacts for recording"
require_path "$OUTPUT_DIR/test_metrics.json"
require_path "$OUTPUT_DIR/graph_summary.json"
require_path "$OUTPUT_DIR/training_curves.png"

ls -lah "$PROCESSED_DIR"
ls -lah "$OUTPUT_DIR"

printf '\n[test_metrics.json]\n'
cat "$OUTPUT_DIR/test_metrics.json"

printf '\n[graph_summary.json head]\n'
python3 - <<'PY'
from pathlib import Path

path = Path("outputs/icews14_record_qwen25_05b_a10/graph_summary.json")
for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
    print(line)
    if idx >= 40:
        break
PY
