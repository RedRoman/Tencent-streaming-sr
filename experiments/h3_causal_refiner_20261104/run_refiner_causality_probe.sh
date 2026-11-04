#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
devices="${1:?Pass verified-idle physical NPU IDs}"
count=$(awk -F, '{print NF}' <<< "$devices")
python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$devices" -- \
  python -m torch.distributed.run --nnodes=1 --nproc-per-node="$count" \
  --master-port=29673 "$EXP_CODE_DIR/probe_refiner_causality.py" --root "$EXP_ROOT" --strict-frame
