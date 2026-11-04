#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
python "$EXP_CODE_DIR/locked_run.py" --devices 0,1,2,3,4,5,6,7 -- \
    python -m torch.distributed.run --nnodes=1 --nproc-per-node=8 --master-port=29643 \
    "$EXP_CODE_DIR/test_full_runtime.py" > "$EXP_ROOT/full_preflight.log" 2>&1
