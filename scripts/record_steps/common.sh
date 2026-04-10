#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE="${1:-/mnt/workspace/LMCA-TIC}"

FULL_RAW_DIR="$WORKSPACE/data/local/icews14/raw"
RECORD_RAW_DIR="$WORKSPACE/data/local/icews14_record_small/raw"
RECORD_BIE_DIR="$WORKSPACE/data/local/icews14_record_small/bie"
PROCESSED_DIR="$WORKSPACE/data/processed/icews14_record_small"
OUTPUT_DIR="$WORKSPACE/outputs/icews14_record_qwen25_05b_a10"
LOG_DIR="$WORKSPACE/logs/icews14_record_qwen25_05b_a10"
CHECKPOINT_DIR="$WORKSPACE/checkpoints/icews14_record_qwen25_05b_a10"
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

prepare_env() {
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
}
