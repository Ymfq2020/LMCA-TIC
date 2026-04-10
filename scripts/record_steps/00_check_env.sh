#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${1:-/mnt/workspace/LMCA-TIC}"

source "$SCRIPT_DIR/common.sh" "$WORKSPACE"
prepare_env

log "workspace=$WORKSPACE"
log "config=$CONFIG_PATH"
log "model=$MODEL_DIR"
log "offline: HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"

ls -lah "$FULL_RAW_DIR"
ls -lah "$MODEL_DIR" | sed -n '1,20p'
