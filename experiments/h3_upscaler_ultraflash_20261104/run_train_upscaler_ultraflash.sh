#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
physical="${1:?Pass one verified-idle physical NPU ID}"
family="${2:-ewarp}"
steps="${3:-2000}"
case "$family" in ewarp|temporal) ;; *) echo "Unknown loss family: $family" >&2; exit 2;; esac
extra=()
if [[ $# -gt 3 ]]; then extra=("${@:4}"); fi
python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$physical" -- \
  python "$EXP_CODE_DIR/train_upscaler_ultraflash.py" \
  --pairs "$H3_ULTRAFLASH_PAIRS" --initial "$H3_UPSCALER_INITIAL_CHECKPOINT" \
  --upstream "$H3_UPSTREAM_DIR" --upscaler "$H3_UPSCALER_WEIGHTS" \
  --family "$family" --steps "$steps" --lr 1e-5 --min-lr 1e-6 --warmup 100 \
  --validate-every 200 --save-every 200 \
  --output "$EXP_ROOT/upscaler_ultraflash/${family}_steps${steps}" --device npu:0 \
  "${extra[@]}"
