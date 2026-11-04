#!/bin/bash
# Does a *temporal* discriminator keep the GAN's high-frequency gain without the
# flicker? Three arms, one variable:
#   ctrl  : window 3 sampling, no GAN            (controls the sampling change)
#   spat  : window 3 + spatial PatchGAN          (the known negative result)
#   temp  : window 3 + temporal PatchGAN         (the new mechanism)
# The generator is memoryless in all three (--temporal-taps 1), so whatever wins
# is deployable at W=1.
set -u
source ./solh3_env.sh
OUT=results/tgan_2310
mkdir -p "$OUT" logs
launch () { # name dev extra...
  local name=$1 dev=$2; shift 2
  ASCEND_RT_VISIBLE_DEVICES=$dev nohup "$SOLH3_PY" -u pixel_train.py \
    --window 3 --temporal-taps 1 --width 160 --blocks 20 --lpips 1.0 \
    --clips-per-step 8 --steps 20000 --eval-every 1000 \
    "$@" --out "$OUT/$name" > "logs/tgan_$name.log" 2>&1 &
  echo "launched $name on npu$dev pid $!"
}
launch ctrl 4
launch spat 5 --gan 0.05
launch temp 6 --gan 0.05 --gan-temporal 3
