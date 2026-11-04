#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
device="${1:?Pass one verified-idle physical NPU ID}"
python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$device" -- \
  python "$EXP_CODE_DIR/train_upscaler_trial.py" \
  --upstream "$H3_UPSTREAM_DIR" --output "$EXP_ROOT/upscaler_trial/finetune20" \
  --device npu:0 --steps 20 --checkpoint-every 10
