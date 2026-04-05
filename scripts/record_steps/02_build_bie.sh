#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${1:-/mnt/workspace/LMCA-TIC}"

source "$SCRIPT_DIR/common.sh" "$WORKSPACE"
prepare_env

log "build offline BIE"
PYTHONPATH=src python3 -m lmca_tic.cli build-bie --config configs/experiments/icews14_record_qwen25_05b_a10.yaml

require_path "$RECORD_BIE_DIR/entity_metadata.jsonl"
ls -lah "$RECORD_BIE_DIR"
