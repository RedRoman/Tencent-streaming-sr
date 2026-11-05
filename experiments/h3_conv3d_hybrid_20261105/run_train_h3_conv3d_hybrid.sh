#!/usr/bin/env bash
# New run only: the original 345M causal upscaler with the hybrid pixel recipe.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
physical="${1:?Supply a verified idle physical NPU ID}"
mode="${2:?Supply preflight or train}"
shift 2
destination="$H3_CONV3D_HYBRID_HOME"
mkdir -p "$destination"
if [[ "$mode" == preflight ]]; then
  test ! -e "$destination/preflight_v1.log"
  python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$physical" -- \
    python "$EXP_CODE_DIR/train_h3_conv3d_hybrid.py" --device npu:0 \
    --steps 8 --warmup 2 --validate-every 4 --save-every 4 --preflight \
    --output "$destination/preflight_v1" "$@" \
    > "$destination/preflight_v1.log" 2>&1
elif [[ "$mode" == train ]]; then
  test ! -e "$destination/train.log"
  python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$physical" -- \
    python "$EXP_CODE_DIR/train_h3_conv3d_hybrid.py" --device npu:0 \
    --steps 2000 --output "$destination/kd_pixel_steps2000" "$@" \
    > "$destination/train.log" 2>&1
else
  echo "Unknown mode: $mode" >&2
  exit 2
fi
