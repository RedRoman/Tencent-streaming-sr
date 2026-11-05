#!/usr/bin/env bash
set -e
v5_motion_code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export H3_TEST16_CODE_DIR="${H3_TEST16_CODE_DIR:-$v5_motion_code_dir/../h3_conv3d_test16_20261105}"
source "$H3_TEST16_CODE_DIR/env.sh"
export EXP_CODE_DIR="$v5_motion_code_dir"
export PYTHONPATH="$EXP_CODE_DIR:$PYTHONPATH"
unset v5_motion_code_dir
