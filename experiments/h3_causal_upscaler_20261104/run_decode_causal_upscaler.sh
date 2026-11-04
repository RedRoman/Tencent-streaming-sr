#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
device="${1:?Pass one verified-idle physical NPU ID}"
python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$device" -- \
  python "$EXP_CODE_DIR/evaluate_causal_upscaler_pixels.py" \
  --pairs "$EXP_ROOT/upscaler_trial/pairs" --trial "$EXP_ROOT/upscaler_causal_trial" \
  --output "$EXP_ROOT/upscaler_causal_trial/decoded" \
  --base-model "$H3_BASE_MODEL" \
  --device npu:0
