#!/bin/bash
# Resume the latent-keyframe arms onto the local mmap'd HR latents.
set -u
source ./solh3_env.sh
OUT=results/latref_2200
for spec in "lat30 4 30" "lat8 5 8"; do
  set -- $spec
  name=$1; dev=$2; rm=$3
  R=""
  [ -f "$OUT/$name/latest.pt" ] && R="--resume $OUT/$name/latest.pt"
  ASCEND_RT_VISIBLE_DEVICES=$dev nohup "$SOLH3_PY" -u pixel_train.py \
    --window 1 --width 160 --blocks 20 --lpips 1.0 \
    --ref-max "$rm" --ref-latent 1 --clips-per-step 48 \
    --steps 40000 --eval-every 2000 $R \
    --out "$OUT/$name" > "logs/latref_${name}b.log" 2>&1 &
  echo "launched $name on npu$dev pid $!"
  sleep 6
done
