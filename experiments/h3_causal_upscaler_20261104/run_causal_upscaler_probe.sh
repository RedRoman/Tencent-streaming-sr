#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
device="${1:?Pass one verified-idle physical NPU ID}"
python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$device" -- \
  python "$EXP_CODE_DIR/probe_causal_upscaler.py" \
  --root "$EXP_ROOT" --upstream "$H3_UPSTREAM_DIR" \
  --output "$EXP_ROOT/upscaler_causal_trial" --device npu:0
