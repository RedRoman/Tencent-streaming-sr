#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
physical="${1:?Supply a verified idle physical NPU ID}"
destination="$H3_CONV3D_HYBRID_HOME"
mkdir -p "$destination"
test ! -e "$destination/causal_prior_long.log"
python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$physical" -- \
  python "$EXP_CODE_DIR/audit_h3_conv3d_causality.py" --device npu:0 \
  --upstream "$H3_UPSTREAM_DIR" \
  --weights "$H3_UPSCALER_WEIGHTS" \
  --initial "$H3_CONV3D_INITIAL_CHECKPOINT" --checkpoint-kind initial \
  --output "$destination/causal_prior_long.json" > "$destination/causal_prior_long.log" 2>&1 || {
  # Preserve the original failure. Only a numerical stream/full discrepancy
  # may proceed to the separate full-precision diagnostic; causality must pass.
  python - "$destination/causal_prior_long.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
assert s['mode'] == 'actual_345m_checkpoint' and s['check_count'] == 12
for name, check in s['checks'].items():
    if name.startswith(('stream_', 'wrapper_forward_chunk')):
        assert check.get('finite'), name
    else:
        assert check['passed'], name
PY
}
bash "$EXP_CODE_DIR/run_h3_conv3d_precision_diagnostic.sh" "$physical"
bash "$EXP_CODE_DIR/run_train_h3_conv3d_hybrid.sh" "$physical" preflight
