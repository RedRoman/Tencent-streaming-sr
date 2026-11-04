#!/bin/bash
# Capacity sweep: single variable is model size. Everything else matches the
# current deliverable (results/best_2114/base_long): window 1, lpips 1.0,
# clips-per-step 8, 40000 steps, rank by val_lpips.
set -u
source ./solh3_env.sh
OUT=results/cap_2230
mkdir -p "$OUT" logs
i=0
for cfg in "192 24 0" "224 24 1" "256 24 3"; do
  set -- $cfg
  w=$1; b=$2; dev=$3
  name="w${w}x${b}"
  ASCEND_RT_VISIBLE_DEVICES=$dev nohup "$SOLH3_PY" -u pixel_train.py \
    --window 1 --width "$w" --blocks "$b" --lpips 1.0 \
    --clips-per-step 8 --steps 40000 --eval-every 2000 \
    --out "$OUT/$name" > "logs/cap_$name.log" 2>&1 &
  echo "launched $name on npu$dev pid $!"
  i=$((i+1))
  sleep 8
done
