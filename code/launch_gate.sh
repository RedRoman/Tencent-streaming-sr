#!/bin/bash
# Does weighing the keyframe PER REGION beat weighing it per frame?
#
# From the latent route's error map: the keyframe remains the better source over
# ~30% of the picture (whatever has not moved) and goes badly wrong on whatever
# has. The shipping refiner only knows HOW OLD the keyframe is -- one scalar for
# the whole frame -- so it must pick a single compromise. The gate feeds it
# |LR_t - LR_keyframe| per position, which costs nothing: both frames are
# already decoded and on the same grid.
#
# Single variable against the shipped recipe (results/latref_2200/lat30):
# window 1, w160x20, lpips 1.0, batch 48, 40000 steps, ref-max 30, ref-latent.
set -u
source ./solh3_env.sh
OUT=results/gate_1710
mkdir -p "$OUT" logs
run () {  # name dev local_gate
  ASCEND_RT_VISIBLE_DEVICES=$2 nohup "$SOLH3_PY" -u pixel_train.py \
    --window 1 --width 160 --blocks 20 --lpips 1.0 \
    --ref-max 30 --ref-latent 1 --local-gate "$3" \
    --clips-per-step 48 --steps 40000 --eval-every 2000 \
    --out "$OUT/$1" > "logs/gate_$1.log" 2>&1 &
  echo "launched $1 (local-gate $3) on npu$2 pid $!"
  sleep 6
}
run gate 0 1
run ctrl 2 0     # same code path, gate off: isolates the gate from any other drift
