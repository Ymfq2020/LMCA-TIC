#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${1:-/mnt/workspace/LMCA-TIC}"

source "$SCRIPT_DIR/common.sh" "$WORKSPACE"
prepare_env

log "split ICEWS14 record subset (300/30/30)"
mkdir -p "$RECORD_RAW_DIR" "$RECORD_BIE_DIR"
sed -n '1,300p' "$FULL_RAW_DIR/train.txt" > "$RECORD_RAW_DIR/train.txt"
sed -n '1,30p' "$FULL_RAW_DIR/valid.txt" > "$RECORD_RAW_DIR/valid.txt"
sed -n '1,30p' "$FULL_RAW_DIR/test.txt" > "$RECORD_RAW_DIR/test.txt"

wc -l \
  "$RECORD_RAW_DIR/train.txt" \
  "$RECORD_RAW_DIR/valid.txt" \
  "$RECORD_RAW_DIR/test.txt"

ls -lah "$RECORD_RAW_DIR"
