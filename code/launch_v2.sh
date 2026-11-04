#!/usr/bin/env bash
# Sol-H3 sweep v2: learned LR re-gridding + LR->HR bridge.
#
# v1 (archived as sweep_*_trilinear_prior) fed the refiner a trilinearly
# upsampled LR latent. That prior is wrong -- H3's 16x-compressed latents put LR
# and HR in different coordinate systems (corr 0.67, 1.9 dB) -- so every v1 arm
# was capped by its input, not its capacity. v2 learns the re-gridding.
set -uo pipefail
cd "$HOME/modelarts/user-job-dir/sol_h3"
source ./solh3_env.sh

STEPS=${STEPS:-4000}
RUNROOT="results/sweep_v2_$(date +%H%M)"
mkdir -p "$RUNROOT" logs

# card : chunk : extra flags : name
ARMS=(
  "0:1:--learned-upsample --objective bridge:c1_lu_bridge"
  "1:5:--learned-upsample --objective bridge:c5_lu_bridge"
  "2:37:--learned-upsample --objective bridge:c37_lu_bridge"
  "3:1:--objective bridge:c1_bridge_nolu"
  "4:1:--learned-upsample --objective noise:c1_lu_noise"
  "5:5:--learned-upsample --objective bridge --cond none:c5_lu_nocond"
  "6:1:--learned-upsample --objective bridge --lr-source degraded:c1_lu_degraded"
  "7:37:--learned-upsample --objective bridge --lr-source degraded:c37_lu_degraded"
)

for spec in "${ARMS[@]}"; do
  CARD="${spec%%:*}"; rest="${spec#*:}"
  CHUNK="${rest%%:*}"; rest="${rest#*:}"
  FLAGS="${rest%:*}"; NAME="${rest##*:}"
  LOG="logs/${RUNROOT##*/}_${NAME}.log"
  ASCEND_RT_VISIBLE_DEVICES=$CARD nohup "$SOLH3_PY" -u solh3_train.py \
      --chunk "$CHUNK" $FLAGS --steps "$STEPS" --eval-every 250 \
      --out "$RUNROOT/$NAME" > "$LOG" 2>&1 &
  echo "card $CARD  $NAME  (chunk=$CHUNK $FLAGS)  pid $!"
  sleep 3
done

echo "$RUNROOT" > results/CURRENT_SWEEP
echo; echo "sweep root: $RUNROOT"
