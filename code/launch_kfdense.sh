#!/bin/bash
# Does denser keyframing close the STRUCTURE gap that the flow stage cannot?
#
# The deficit is 5.6 dB of structure (28.50 vs 34.06) and only 2.3 dB of detail.
# Noise injection and generative objectives address detail, not placement, and
# the flow stage sees no information stage 1 did not. A denser keyframe is the
# one lever that injects genuinely new on-manifold HR signal, which is what
# structure needs. The pixel route already showed the direction: kf=30 -> kf=8
# took LPIPS 0.1714 -> 0.1576.
#
# One variable: how far back the keyframe may sit.
#   k1  <= 1 chunk  (~17 frames)
#   k0  the chunk's own keyframe (0 back) -- an upper bound on what density buys
set -u
source ./solh3_env.sh
OUT=results/latsr_kf_1100
mkdir -p "$OUT" logs
run () {  # name dev kf_chunks
  ASCEND_RT_VISIBLE_DEVICES=$2 nohup "$SOLH3_PY" -u latsr_train.py \
    --width 256 --lr-blocks 8 --hr-blocks 6 \
    --crop 16 --batch 8 --kf-chunks "$3" --lpips 1.0 \
    --steps 20000 --eval-every 1000 \
    --out "$OUT/$1" > "logs/latsr_kf_$1.log" 2>&1 &
  echo "launched $1 (kf-chunks $3) on npu$2 pid $!"
  sleep 6
}
run k1 1 1
run k0 3 0
