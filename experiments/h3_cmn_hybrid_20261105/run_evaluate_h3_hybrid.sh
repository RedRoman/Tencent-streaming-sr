#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
physical="${1:?Pass one verified-idle physical NPU ID, or cpu for finalize}"
mode="${2:-latents}"
extra=()
case "$mode" in
  latents) ;;
  decode) extra=(--decode-only --shard-index "${3:-0}" --num-shards "${4:-1}") ;;
  finalize) extra=(--finalize --num-shards "${3:-1}") ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
mkdir -p "$H3_HYBRID_HOME"
compute_device=npu:0
if [[ "$mode" == finalize ]]; then compute_device=cpu; fi
cmd=(python "$EXP_CODE_DIR/evaluate_h3_hybrid.py"
  --run "$H3_HYBRID_RUN_DIR" --pairs "$H3_ULTRAFLASH_PAIRS"
  --cmn-evaluation "$H3_CMN_EVALUATION" --output "$H3_HYBRID_EVALUATION"
  --base-model "$H3_BASE_MODEL" --device "$compute_device" "${extra[@]}")
if [[ "$mode" == finalize ]]; then
  "${cmd[@]}"
else
  python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$physical" -- "${cmd[@]}"
fi
