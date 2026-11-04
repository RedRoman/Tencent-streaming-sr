#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
device="${1:?Pass one verified-idle physical NPU ID}"
probe="$EXP_ROOT/refiner_causality_probe"
python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$device" -- \
  python "$H3_REFINER_CODE_DIR/decode_results.py" --fields '' --frames 43 --fps 24 \
  --latent "upscaled=$probe/upscaled.pt" \
  --latent "block5_sigma03=$probe/causal_rollout_sigma03.pt" \
  --latent "frame1_sigma03=$probe/strict_frame_rollout_sigma03.pt" \
  --latent "reference=$probe/reference.pt" \
  --device npu:0 --output "$probe/decoded"
