#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
physical="${1:?Supply a verified idle physical NPU ID}"
test ! -e "$H3_CONV3D_HYBRID_HOME/causal_prior_hf32_off.log"
python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$physical" -- \
  python "$EXP_CODE_DIR/audit_h3_conv3d_precision.py" --hf32-conv false \
  --device npu:0 --upstream "$H3_UPSTREAM_DIR" \
  --weights "$H3_UPSCALER_WEIGHTS" \
  --initial "$H3_CONV3D_INITIAL_CHECKPOINT" --checkpoint-kind initial \
  --output "$H3_CONV3D_HYBRID_HOME/causal_prior_hf32_off.json" \
  > "$H3_CONV3D_HYBRID_HOME/causal_prior_hf32_off.log" 2>&1
