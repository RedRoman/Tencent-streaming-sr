#!/usr/bin/env bash
# Start once, after the independent eight-update preflight. No implicit restart.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
physical="${1:?Supply verified idle physical NPU}"
if [[ "${2:-}" != --inside-lock ]]; then
  exec python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$physical" -- bash "$0" "$physical" --inside-lock
fi
destination="$EXP_ROOT/upscaler_v5_motion_20261106"
python "$EXP_CODE_DIR/h3_v5_pipeline_contract.py" prepare
stage=starting
status_on_exit() {
  code=$?
  python "$EXP_CODE_DIR/h3_v5_pipeline_contract.py" status "$stage" "$code"
}
trap status_on_exit EXIT
for arm in control motion; do
  stage="train_$arm"
  python "$EXP_CODE_DIR/h3_v5_pipeline_contract.py" progress "$stage"
  python "$EXP_CODE_DIR/h3_v5_pipeline_contract.py" verify
  test ! -e "$destination/${arm}.log"
  python "$EXP_CODE_DIR/train_h3_v5_motion.py" --device npu:0 --arm "$arm" \
    --steps 2000 --output "$destination/${arm}_steps2000" > "$destination/${arm}.log" 2>&1
  stage="audit_${arm}_hf32_off"
  python "$EXP_CODE_DIR/h3_v5_pipeline_contract.py" progress "$stage"
  python "$H3_CONV3D_CODE_DIR/audit_h3_conv3d_precision.py" --hf32-conv false --device npu:0 \
    --upstream "$H3_UPSTREAM_DIR" \
    --weights "$H3_UPSCALER_WEIGHTS" \
    --initial "$destination/${arm}_steps2000/best.pt" --checkpoint-kind hybrid \
    --output "$destination/causal_${arm}_hf32_off.json" > "$destination/causal_${arm}_hf32_off.log" 2>&1
 done
stage=paired_training_contract
python "$EXP_CODE_DIR/h3_v5_pipeline_contract.py" progress "$stage"
python "$EXP_CODE_DIR/h3_v5_pipeline_contract.py" paired
stage=full120_dev_gate_then_test
python "$EXP_CODE_DIR/h3_v5_pipeline_contract.py" progress "$stage"
python "$EXP_CODE_DIR/evaluate_h3_v5_motion.py" --stage all --device npu:0 > "$destination/evaluate.log" 2>&1
stage=verify_final_source
python "$EXP_CODE_DIR/h3_v5_pipeline_contract.py" verify
stage=completed
