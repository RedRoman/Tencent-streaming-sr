#!/usr/bin/env bash
set -e
# Reuse the published refiner experiment's runtime and tensor/VAE helpers.
upscaler_code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export H3_REFINER_CODE_DIR="${H3_REFINER_CODE_DIR:-$upscaler_code_dir/../h3_refiner_self_forcing_20261104}"
source "$H3_REFINER_CODE_DIR/env.sh"
export EXP_CODE_DIR="$upscaler_code_dir"
export PYTHONPATH="$EXP_CODE_DIR:$H3_REFINER_CODE_DIR:$PYTHONPATH"
unset upscaler_code_dir
