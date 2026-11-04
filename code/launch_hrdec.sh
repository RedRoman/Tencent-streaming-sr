#!/bin/bash
# Distil the 2.42B HR VAE decoder. One variable: student capacity.
# Speed is already far inside budget for all three (82-265 ms vs 708), so this
# sweep is purely about how much quality each size can reach.
set -u
source ./solh3_env.sh
OUT=results/hrdec_1200
mkdir -p "$OUT" logs
launch () {  # name dev width blocks
  ASCEND_RT_VISIBLE_DEVICES=$2 nohup "$SOLH3_PY" -u hrdec_train.py \
    --width "$3" --blocks "$4" --crop 16 --batch 6 --lpips 0.5 \
    --steps 20000 --eval-every 1000 \
    --out "$OUT/$1" > "logs/hrdec_$1.log" 2>&1 &
  echo "launched $1 (w$3 $4) on npu$2 pid $!"
  sleep 6
}
launch s384  0 384  2,2,2,1
launch s512  1 512  4,4,2,2
launch s768  2 768  4,4,2,2
