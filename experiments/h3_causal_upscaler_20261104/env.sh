#!/usr/bin/env bash
set -e
# The previous upscaler experiment supplies training helpers and loads the
# shared refiner/VAE environment. This directory takes Python import priority.
causal_upscaler_code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export H3_UPSCALER_CODE_DIR="${H3_UPSCALER_CODE_DIR:-$causal_upscaler_code_dir/../h3_upscaler_trial_20261104}"
source "$H3_UPSCALER_CODE_DIR/env.sh"
export EXP_CODE_DIR="$causal_upscaler_code_dir"
export PYTHONPATH="$EXP_CODE_DIR:$PYTHONPATH"
unset causal_upscaler_code_dir
