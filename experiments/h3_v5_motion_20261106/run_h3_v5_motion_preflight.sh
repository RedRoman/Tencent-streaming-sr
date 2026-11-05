#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
physical="${1:?Supply verified idle physical NPU}"
destination="$EXP_ROOT/upscaler_v5_motion_20261106"
test ! -e "$destination/preflight.log"
test ! -e "$destination/preflight"
python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$physical" -- \
  python "$EXP_CODE_DIR/train_h3_v5_motion.py" --device npu:0 --arm motion \
  --steps 8 --warmup 2 --validate-every 4 --save-every 4 --preflight \
  --output "$destination/preflight" > "$destination/preflight.log" 2>&1
