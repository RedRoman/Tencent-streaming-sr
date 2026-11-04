#!/bin/bash
set -u
source ./solh3_env.sh
OUT=results/latsr_2200
mkdir -p "$OUT" logs
run () {  # name dev width lrb hrb
  ASCEND_RT_VISIBLE_DEVICES=$2 nohup "$SOLH3_PY" -u latsr_train.py \
    --width "$3" --lr-blocks "$4" --hr-blocks "$5" \
    --crop 16 --batch 8 --kf-chunks 2 --lpips 1.0 \
    --steps 20000 --eval-every 1000 \
    --out "$OUT/$1" > "logs/latsr_$1.log" 2>&1 &
  echo "launched $1 (w$3 $4+$5) on npu$2 pid $!"
  sleep 6
}
run w256 0 256 8 6
run w384 1 384 10 8
