#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
device="${1:?Pass one verified-idle physical NPU ID}"
python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$device" -- \
  python "$EXP_CODE_DIR/prepare_upscaler_pairs.py" \
  --existing-data "$EXP_ROOT/data" --output "$EXP_ROOT/upscaler_trial/pairs" \
  --base-model "$H3_BASE_MODEL" --device npu:0
