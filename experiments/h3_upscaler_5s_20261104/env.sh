#!/usr/bin/env bash
set -e
# Inherit causal-upscaler, earlier training, and shared VAE/device-lock helpers.
five_second_code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export H3_CAUSAL_UPSCALER_CODE_DIR="${H3_CAUSAL_UPSCALER_CODE_DIR:-$five_second_code_dir/../h3_causal_upscaler_20261104}"
source "$H3_CAUSAL_UPSCALER_CODE_DIR/env.sh"
export EXP_CODE_DIR="$five_second_code_dir"
export PYTHONPATH="$EXP_CODE_DIR:$PYTHONPATH"
unset five_second_code_dir
