#!/usr/bin/env bash
# Teacher preparation and the eight-update preflight are explicit prior steps.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
physical="${1:?Pass one verified-idle physical NPU ID}"
[[ -f "$H3_HYBRID_TEACHER_CACHE/READY.json" ]] || { echo "Prepare teacher cache first" >&2; exit 2; }
bash "$EXP_CODE_DIR/run_train_h3_hybrid.sh" "$physical" train
bash "$EXP_CODE_DIR/run_evaluate_h3_hybrid.sh" "$physical" latents \
  > "$H3_HYBRID_HOME/evaluate_latents.log" 2>&1
bash "$EXP_CODE_DIR/run_evaluate_h3_hybrid.sh" "$physical" decode 0 1 \
  > "$H3_HYBRID_HOME/decode.log" 2>&1
bash "$EXP_CODE_DIR/run_evaluate_h3_hybrid.sh" cpu finalize 1 \
  > "$H3_HYBRID_HOME/finalize.log" 2>&1
python "$EXP_CODE_DIR/plot_h3_hybrid.py" --run "$H3_HYBRID_RUN_DIR" \
  --output "$H3_HYBRID_HOME/training_curves.png"
