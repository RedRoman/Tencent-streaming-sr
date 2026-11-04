#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
devices="${1:?Pass verified-idle physical NPU IDs, e.g. 1,2,3,4,5,6,7}"
count=$(awk -F, '{print NF}' <<< "$devices")
test -f "$EXP_ROOT/merged_transformer/MERGED.json"
test -f "$EXP_ROOT/data/holdout.jsonl"
python "$EXP_CODE_DIR/locked_run.py" --devices "$devices" -- \
    python -m torch.distributed.run --nnodes=1 --nproc-per-node="$count" \
    --master-port=29637 "$EXP_CODE_DIR/train.py" --root "$EXP_ROOT" \
    --steps 6 --fake-warmup 4 --adapter-rank 8 --block-frames 5
