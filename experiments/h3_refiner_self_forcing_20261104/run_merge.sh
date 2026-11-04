#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
python "$EXP_CODE_DIR/merge_refiner.py" \
    --base "$H3_BASE_MODEL" \
    --assets "$EXP_ROOT/assets" --output "$EXP_ROOT/merged_transformer"
