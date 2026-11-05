#!/usr/bin/env bash
# One allocation/lock covers preparation, inference and all14 full-video decodes.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
physical="${1:?Supply a verified idle physical NPU ID}"
shift
python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$physical" -- \
  python "$EXP_CODE_DIR/evaluate_h3_conv3d_test16.py" --stage all --device npu:0 "$@"
