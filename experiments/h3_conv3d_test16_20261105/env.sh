#!/usr/bin/env bash
set -e
h3_test16_code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export H3_CONV3D_CODE_DIR="${H3_CONV3D_CODE_DIR:-$h3_test16_code_dir/../h3_conv3d_hybrid_20261105}"
source "$H3_CONV3D_CODE_DIR/env.sh"
export EXP_CODE_DIR="$h3_test16_code_dir"
export PYTHONPATH="$EXP_CODE_DIR:$PYTHONPATH"
export H3_CONV3D_TEST16_EVALUATION="${H3_CONV3D_TEST16_EVALUATION:-$H3_CONV3D_HYBRID_HOME/evaluation_test16_extension}"
unset h3_test16_code_dir
