#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
physical="${1:?Pass one verified-idle physical NPU ID}"
mode="${2:-train}"
if [[ $# -ge 2 ]]; then shift 2; else shift; fi
mkdir -p "$H3_HYBRID_HOME"
case "$mode" in
  preflight)
    schedule=(--steps 8 --warmup 2 --validate-every 4 --save-every 4 --preflight)
    output="$H3_HYBRID_PREFLIGHT_DIR"; log="$H3_HYBRID_HOME/preflight_v2.log" ;;
  train)
    schedule=(--steps 2000 --warmup 100 --validate-every 200 --save-every 100)
    output="$H3_HYBRID_RUN_DIR"; log="$H3_HYBRID_HOME/train.log" ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$physical" -- \
  python "$EXP_CODE_DIR/train_h3_hybrid.py" --device npu:0 \
  --pairs "$H3_ULTRAFLASH_PAIRS" --teacher-cache "$H3_HYBRID_TEACHER_CACHE" \
  --initial "$H3_CMN_INITIAL_CHECKPOINT" --base-model "$H3_BASE_MODEL" \
  --output "$output" --lr 3e-5 --min-lr 3e-6 --kd-weight .25 --warp-weight .1 \
  --l1-weight 1 --lpips-weight .2 --full-every 4 --seed 20261105 \
  "${schedule[@]}" "$@" > "$log" 2>&1
