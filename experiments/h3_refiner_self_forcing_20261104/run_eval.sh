#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
devices="${1:?Pass verified-idle physical NPU IDs}"
count=$(awk -F, '{print NF}' <<< "$devices")
python "$EXP_CODE_DIR/locked_run.py" --devices "$devices" -- \
    python -m torch.distributed.run --nnodes=1 --nproc-per-node="$count" \
    --master-port=29639 "$EXP_CODE_DIR/train.py" --root "$EXP_ROOT" \
    --adapter-rank 8 --block-frames 5 --eval-only --checkpoint "$EXP_ROOT/outputs/adapters.pt"
