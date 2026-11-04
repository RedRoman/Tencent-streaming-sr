#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
device="${1:?Pass one verified-idle physical NPU ID}"
python "$EXP_CODE_DIR/locked_run.py" --devices "$device" -- \
    python "$EXP_CODE_DIR/decode_results.py" --fields '' --frames 17 --fps 24 \
    --latent "upscaled=$EXP_ROOT/causality_ablation/upscaled.pt" \
    --latent "dense13=$EXP_ROOT/causality_ablation/dense13.pt" \
    --latent "static_mask13=$EXP_ROOT/causality_ablation/static_mask13.pt" \
    --latent "dense5=$EXP_ROOT/causality_ablation/dense5.pt" \
    --latent "static_mask5=$EXP_ROOT/causality_ablation/static_mask5.pt" \
    --latent "static_mask5_sigma03=$EXP_ROOT/causality_ablation/static_mask5_sigma03.pt" \
    --device npu:0 --output "$EXP_ROOT/causality_ablation/decoded"
