#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
mode="${1:?Pass plan, worker, or finalize}"
script="$EXP_CODE_DIR/prepare_upscaler_ultraflash_pairs.py"
case "$mode" in
  plan)
    : "${H3_SOURCE_DATASET:?Set H3_SOURCE_DATASET to the source dataset directory}"
    num_shards="${2:-3}"
    python "$script" --plan-only --dataset "$H3_SOURCE_DATASET" \
      --output "$H3_ULTRAFLASH_PAIRS" --base-model "$H3_BASE_MODEL" \
      --split-seed 20261104 --posterior-seed 42 --num-shards "$num_shards"
    ;;
  worker)
    device="${2:?Pass one verified-idle physical NPU ID}"
    shard_index="${3:?Pass the shard index}"
    num_shards="${4:?Pass total number of shards}"
    python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$device" -- \
      python "$script" --plan "$H3_ULTRAFLASH_PAIRS/plan.json" \
      --output "$H3_ULTRAFLASH_PAIRS" --base-model "$H3_BASE_MODEL" \
      --device npu:0 --shard-index "$shard_index" --num-shards "$num_shards"
    ;;
  finalize)
    num_shards="${2:?Pass total number of completed shards}"
    python "$script" --finalize --plan "$H3_ULTRAFLASH_PAIRS/plan.json" \
      --output "$H3_ULTRAFLASH_PAIRS" --base-model "$H3_BASE_MODEL" \
      --num-shards "$num_shards"
    ;;
  *)
    echo "Unknown mode: $mode" >&2
    exit 2
    ;;
esac
