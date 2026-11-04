#!/bin/bash
# Flow-matching manifold alignment on top of the frozen stage-1 regressor.
# The logged `gain` (stage1_lpips - val_lpips) is the whole point: the head is
# zero-initialised so step 0 reproduces stage 1 exactly, and any positive gain
# is the generative stage doing something regression could not.
set -u
source ./solh3_env.sh
OUT=results/latflow_0140
mkdir -p "$OUT" logs
run () {  # name dev width blocks stage1
  ASCEND_RT_VISIBLE_DEVICES=$2 nohup "$SOLH3_PY" -u latflow_train.py \
    --stage1 "$5" --width "$3" --blocks "$4" \
    --crop 16 --batch 6 --kf-chunks 2 --cond-noise 0.3 --nfe 2 \
    --steps 20000 --eval-every 1000 \
    --out "$OUT/$1" > "logs/latflow_$1.log" 2>&1 &
  echo "launched $1 (w$3 x$4) on npu$2 pid $!"
  sleep 6
}
run f192 0 192 10 results/latsr_2200/w256/best.pt
run f320 2 320 14 results/latsr_2200/w256/best.pt
