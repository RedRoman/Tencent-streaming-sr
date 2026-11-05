#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
physical="${1:?Pass a verified-idle physical NPU ID, or cpu for finalize}"
mode="${2:-latents}"
extra=()
case "$mode" in
  latents) ;;
  decode) extra=(--decode-only --shard-index "${3:-0}" --num-shards "${4:-1}") ;;
  finalize) extra=(--finalize --num-shards "${3:-1}") ;;
  *) echo "Unknown mode: $mode" >&2; exit 2;;
esac
compute_device=npu:0
if [[ "$mode" == finalize ]]; then compute_device=cpu; fi
cmd=(python "$EXP_CODE_DIR/evaluate_h3_cmn.py"   --run "${H3_CMN_RUN_DIR:-$EXP_ROOT/upscaler_cmn/ewarp_steps10000}"   --pairs "$H3_ULTRAFLASH_PAIRS"   --old-evaluation "${H3_OLD_UPSCALER_EVALUATION:-$EXP_ROOT/upscaler_ultraflash/evaluation}"   --output "$EXP_ROOT/upscaler_cmn/evaluation" --base-model "$H3_BASE_MODEL"   --device "$compute_device" "${extra[@]}")
if [[ "$mode" == finalize ]]; then
  "${cmd[@]}"
else
  python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$physical" -- "${cmd[@]}"
fi
