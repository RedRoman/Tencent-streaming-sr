#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
[[ -f "$H3_ULTRAFLASH_PAIRS/READY.json" ]] || { echo "Finalize all pairs before flow preparation" >&2; exit 1; }
# CPU only. Launch each shard index at most once with the same shard count.
python "$EXP_CODE_DIR/ultraflash_ewarp.py" --pairs "$H3_ULTRAFLASH_PAIRS" \
  --shard-index "${1:-0}" --num-shards "${2:-1}"
