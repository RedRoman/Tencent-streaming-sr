#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
device="${1:?Pass one verified-idle physical NPU ID}"
probe="$EXP_ROOT/refiner_fixed_shape"
python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$device" -- \
  python "$H3_REFINER_CODE_DIR/decode_results.py" --fields '' --frames 43 --fps 24 \
  --latent "upscaled=$probe/upscaled.pt" \
  --latent "fixed_block5_sigma03=$probe/fixed_block5_sigma03.pt" \
  --latent "fixed_frame1_sigma03=$probe/fixed_block1_sigma03.pt" \
  --latent "reference=$probe/reference.pt" \
  --device npu:0 --output "$probe/decoded"
