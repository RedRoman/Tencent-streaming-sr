#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
physical="${1:?Pass one verified-idle physical NPU ID}"
steps="${2:-10000}"
extra=()
if [[ $# -gt 2 ]]; then extra=("${@:3}"); fi
python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$physical" --   python "$EXP_CODE_DIR/train_h3_cmn.py" --pairs "$H3_ULTRAFLASH_PAIRS"   --output "$EXP_ROOT/upscaler_cmn/ewarp_steps${steps}" --steps "$steps"   --width 64 --blocks 2 --weight 0.1   --lr 0.0002 --min-lr 2e-05 --warmup 200 --validate-every 500   --seed 20261105 --device npu:0 "${extra[@]}"
