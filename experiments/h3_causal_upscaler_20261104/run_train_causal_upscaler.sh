#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
device="${1:?Pass one verified-idle physical NPU ID}"
initialization="${2:-aligned_past}"
steps="${3:-20}"
python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$device" -- \
  python "$EXP_CODE_DIR/train_causal_upscaler.py" \
  --upstream "$H3_UPSTREAM_DIR" \
  --initialization "$initialization" --steps "$steps" \
  --output "$EXP_ROOT/upscaler_causal_finetune/${initialization}_steps${steps}" \
  --device npu:0
