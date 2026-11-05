#!/usr/bin/env bash
set -e
# Inherit the published CMN -> UltraFlash -> causal/upscaler -> H3 helper chain.
hybrid_code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export H3_CMN_CODE_DIR="${H3_CMN_CODE_DIR:-$hybrid_code_dir/../h3_cmn_upscaler_20261105}"
source "$H3_CMN_CODE_DIR/env.sh"
export EXP_CODE_DIR="$hybrid_code_dir"
export PYTHONPATH="$EXP_CODE_DIR:$PYTHONPATH"
export H3_HYBRID_HOME="${H3_HYBRID_HOME:-$EXP_ROOT/upscaler_hybrid}"
export H3_HYBRID_TEACHER_CACHE="${H3_HYBRID_TEACHER_CACHE:-$H3_HYBRID_HOME/teacher_cache}"
export H3_HYBRID_RUN_DIR="${H3_HYBRID_RUN_DIR:-$H3_HYBRID_HOME/cmn_kd_pixel_steps2000}"
export H3_HYBRID_PREFLIGHT_DIR="${H3_HYBRID_PREFLIGHT_DIR:-$H3_HYBRID_HOME/preflight_v2}"
export H3_HYBRID_EVALUATION="${H3_HYBRID_EVALUATION:-$H3_HYBRID_HOME/evaluation}"
export H3_CMN_INITIAL_CHECKPOINT="${H3_CMN_INITIAL_CHECKPOINT:-$EXP_ROOT/upscaler_cmn/ewarp_steps10000/best_step_009500.pt}"
export H3_CMN_EVALUATION="${H3_CMN_EVALUATION:-$EXP_ROOT/upscaler_cmn/evaluation}"
unset hybrid_code_dir
