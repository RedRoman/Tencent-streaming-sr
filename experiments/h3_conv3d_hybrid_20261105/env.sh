#!/usr/bin/env bash
set -e
conv3d_hybrid_code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export H3_HYBRID_CODE_DIR="${H3_HYBRID_CODE_DIR:-$conv3d_hybrid_code_dir/../h3_cmn_hybrid_20261105}"
source "$H3_HYBRID_CODE_DIR/env.sh"
export EXP_CODE_DIR="$conv3d_hybrid_code_dir"
export PYTHONPATH="$EXP_CODE_DIR:$PYTHONPATH"
export H3_CONV3D_HYBRID_HOME="${H3_CONV3D_HYBRID_HOME:-$EXP_ROOT/upscaler_conv3d_hybrid}"
export H3_CONV3D_INITIAL_CHECKPOINT="${H3_CONV3D_INITIAL_CHECKPOINT:-$EXP_ROOT/upscaler_ultraflash/ewarp_steps2000/best.pt}"
unset conv3d_hybrid_code_dir
