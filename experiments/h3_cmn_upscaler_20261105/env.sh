#!/usr/bin/env bash
set -e
cmn_code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export H3_ULTRAFLASH_CODE_DIR="${H3_ULTRAFLASH_CODE_DIR:-$cmn_code_dir/../h3_upscaler_ultraflash_20261104}"
source "$H3_ULTRAFLASH_CODE_DIR/env.sh"
export EXP_CODE_DIR="$cmn_code_dir"
export PYTHONPATH="$EXP_CODE_DIR:$PYTHONPATH"
unset cmn_code_dir
