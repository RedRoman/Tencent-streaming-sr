#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
physical="${1:?Pass one verified-idle physical NPU ID, or cpu for finalize}"
mode="${2:-latents}"
steps="${3:-2000}"
extra=()
case "$mode" in
  latents) ;;
  decode) extra=(--decode-only --shard-index "${4:-0}" --num-shards "${5:-1}") ;;
  finalize) extra=(--finalize --num-shards "${4:-1}") ;;
  *) echo "Unknown evaluation mode: $mode" >&2; exit 2;;
esac
compute_device=npu:0
if [[ "$mode" == finalize ]]; then compute_device=cpu; fi
evaluate_cmd=(python "$EXP_CODE_DIR/evaluate_upscaler_ultraflash.py" \
  --pairs "$H3_ULTRAFLASH_PAIRS" --initial "$H3_UPSCALER_INITIAL_CHECKPOINT" \
  --upstream "$H3_UPSTREAM_DIR" --upscaler "$H3_UPSCALER_WEIGHTS" \
  --base-model "$H3_BASE_MODEL" \
  --ewarp-run "$EXP_ROOT/upscaler_ultraflash/ewarp_steps${steps}" \
  --control-run "$EXP_ROOT/upscaler_ultraflash/temporal_steps${steps}" \
  --output "$EXP_ROOT/upscaler_ultraflash/evaluation_steps${steps}" \
  --device "$compute_device" "${extra[@]}")
if [[ "$mode" == finalize ]]; then
  # Fingerprint and file checks only; this mode does not load models or use NPU.
  "${evaluate_cmd[@]}"
else
  python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$physical" -- "${evaluate_cmd[@]}"
fi
