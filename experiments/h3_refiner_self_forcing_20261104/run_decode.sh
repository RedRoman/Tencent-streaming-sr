#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
device="${1:?Pass one verified-idle physical NPU ID}"
python "$EXP_CODE_DIR/locked_run.py" --devices "$device" -- \
    python "$EXP_CODE_DIR/decode_results.py" \
    --sample "$EXP_ROOT/data/samples/VIDGEN5S_0380588.pt" --fields '' \
    --latent "official_3step=$EXP_ROOT/outputs/dense_three.pt" \
    --latent "dense_1step=$EXP_ROOT/outputs/dense_one_untrained.pt" \
    --latent "causal_before=$EXP_ROOT/outputs/causal_one_untrained.pt" \
    --latent "causal_after=$EXP_ROOT/outputs/causal_one_trained.pt" \
    --device npu:0 --output "$EXP_ROOT/decoded_results/VIDGEN5S_0380588"
