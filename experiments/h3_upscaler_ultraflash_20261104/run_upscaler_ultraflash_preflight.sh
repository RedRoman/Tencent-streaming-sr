#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
physical="${1:?Pass one verified-idle physical NPU ID}"
export H3_ULTRAFLASH_PREFLIGHT_PAIRS="${H3_ULTRAFLASH_PREFLIGHT_PAIRS:-$EXP_ROOT/upscaler_ultraflash/preflight_pairs}"
python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$physical" -- bash -c '
set -euo pipefail
python -B "$EXP_CODE_DIR/test_ultraflash_ewarp.py" --device cpu
python -B "$EXP_CODE_DIR/test_ultraflash_ewarp.py" --device npu:0
python "$EXP_CODE_DIR/train_upscaler_ultraflash.py" \
  --pairs "$H3_ULTRAFLASH_PREFLIGHT_PAIRS" \
  --initial "$H3_UPSCALER_INITIAL_CHECKPOINT" \
  --upstream "$H3_UPSTREAM_DIR" --upscaler "$H3_UPSCALER_WEIGHTS" \
  --output "$EXP_ROOT/upscaler_ultraflash/preflight_run" \
  --preflight --steps 4 --warmup 1 --save-every 4 --validate-every 4 --device npu:0
'
