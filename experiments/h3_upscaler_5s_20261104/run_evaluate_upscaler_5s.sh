#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
device="${1:?Pass one verified-idle physical NPU ID}"
python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$device" -- \
  python "$EXP_CODE_DIR/evaluate_upscaler_5s.py" \
  --pairs "$EXP_ROOT/upscaler_5s/pairs" \
  --prior-trial "$EXP_ROOT/upscaler_causal_finetune/aligned_past_steps50" \
  --upstream "$H3_UPSTREAM_DIR" \
  --upscaler "$EXP_ROOT/assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors" \
  --base-model "$H3_BASE_MODEL" \
  --output "$EXP_ROOT/upscaler_5s/evaluation" --device npu:0
