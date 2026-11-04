#!/usr/bin/env bash
# Produce the full Sol-H3 deliverable in one pass: deterministic re-scoring of
# every arm, pixel metrics + latency on the arms that matter, and side-by-side
# MP4s. Safe to run while training is still going -- it only reads checkpoints.
set -uo pipefail
cd "$HOME/modelarts/user-job-dir/sol_h3"
source ./solh3_env.sh

R=$(cat results/CURRENT_SWEEP)
CARD=${CARD:-7}
NFE=${NFE:-2}
CLIPS=${CLIPS:-6}
OUT=results/report_$(date +%H%M)
mkdir -p "$OUT"

echo "== 1/3 deterministic re-score of every arm"
ASCEND_RT_VISIBLE_DEVICES=$CARD $SOLH3_PY -u rescore_arms.py "$R" best.pt 2>&1 \
  | grep -vE "UserWarning|warnings.warn|Warning:|loading support|ded due|re-store" \
  | tee "$OUT/rescore.txt"

echo
echo "== 2/3 pixel metrics + latency (NFE=$NFE, $CLIPS clips)"
for arm in "$@"; do
  [ -f "$R/$arm/best.pt" ] || { echo "  skip $arm (no checkpoint)"; continue; }
  echo "-- $arm"
  ASCEND_RT_VISIBLE_DEVICES=$CARD $SOLH3_PY -u solh3_eval.py \
      --ckpt "$R/$arm/best.pt" --steps $NFE --clips $CLIPS --frames 37 \
      --save-video 1 --out "$OUT/$arm" 2>&1 \
    | grep -vE "UserWarning|warnings.warn|Warning:|loading support|ded due|re-store|SyntaxWarning|match = " \
    | tee "$OUT/$arm.log" | grep -E '^\{|SUMMARY'
done

echo
echo "== 3/3 comparison videos"
for arm in "$@"; do
  [ -d "$OUT/$arm" ] && $SOLH3_PY -u solh3_mkvideo.py --dir "$OUT/$arm" 2>&1 | grep -E 'wrote|Error'
done

echo
echo "report written to $OUT"
