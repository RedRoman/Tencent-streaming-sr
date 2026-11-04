#!/usr/bin/env bash
# Launch the Sol-H3 temporal-window sweep: one arm per NPU, all arms identical
# except for --chunk (and the one --cond none control).
set -uo pipefail
cd /home/ma-user/modelarts/user-job-dir/sol_h3
source ./solh3_env.sh

STEPS=4000
STAMP=$(date +%Y%m%d_%H%M)
RUNROOT="results/sweep_${STAMP}"
mkdir -p "$RUNROOT" logs

# card:chunk:cond
ARMS=(
  "0:1:concat"
  "1:5:concat"
  "2:13:concat"
  "3:37:concat"
  "4:5:none"
)

for spec in "${ARMS[@]}"; do
  IFS=: read -r CARD CHUNK COND <<<"$spec"
  NAME="c${CHUNK}_${COND}"
  OUT="$RUNROOT/$NAME"
  LOG="logs/sweep_${STAMP}_${NAME}.log"
  echo "launching $NAME on NPU $CARD -> $OUT"
  ASCEND_RT_VISIBLE_DEVICES=$CARD nohup "$SOLH3_PY" -u solh3_train.py \
      --chunk "$CHUNK" --cond "$COND" --steps "$STEPS" \
      --eval-every 250 --out "$OUT" \
      > "$LOG" 2>&1 &
  echo "  pid $! log $LOG"
  sleep 3
done

echo "$RUNROOT" > results/CURRENT_SWEEP
echo
echo "sweep root: $RUNROOT"
jobs -l
