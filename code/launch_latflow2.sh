#!/bin/bash
# Round 2: the flow stage made generative. Two changes from round 1, both
# aimed at the same diagnosis -- round 1 was still a deterministic map and so
# could only re-derive stage 1's conditional mean, worse.
#   src-sigma   the ODE's source is perturbed, so the stage transports a
#               neighbourhood instead of memorising one correction
#   z1 channel  stage 1's answer is an explicit condition, not just the start,
#               so this stage corrects rather than recomputes
# src-sigma is swept because the stage-1 residual has power 0.233 (std ~0.48),
# so 0.25 and 0.5 bracket "a fraction of" and "comparable to" that residual.
set -u
source ./solh3_env.sh
OUT=results/latflow2_1030
mkdir -p "$OUT" logs
run () {  # name dev sigma
  ASCEND_RT_VISIBLE_DEVICES=$2 nohup "$SOLH3_PY" -u latflow_train.py \
    --stage1 results/latsr_2200/w256/best.pt \
    --width 256 --blocks 12 --crop 16 --batch 6 --kf-chunks 2 \
    --cond-noise 0.3 --src-sigma "$3" --nfe 2 \
    --steps 20000 --eval-every 1000 \
    --out "$OUT/$1" > "logs/latflow2_$1.log" 2>&1 &
  echo "launched $1 (src-sigma $3) on npu$2 pid $!"
  sleep 6
}
run s025 0 0.25
run s050 2 0.50
