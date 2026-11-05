#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
physical="${1:?Pass one verified-idle physical NPU ID}"
shift
mkdir -p "$H3_HYBRID_HOME"
python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$physical" -- \
  python "$EXP_CODE_DIR/prepare_h3_hybrid_teacher.py" --device npu:0 \
  --pairs "$H3_ULTRAFLASH_PAIRS" --upstream "$H3_UPSTREAM_DIR" \
  --teacher "$H3_UPSCALER_WEIGHTS" --output "$H3_HYBRID_TEACHER_CACHE" "$@" \
  > "$H3_HYBRID_HOME/teacher_prepare.log" 2>&1
