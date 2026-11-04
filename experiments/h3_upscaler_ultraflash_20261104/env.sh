#!/usr/bin/env bash
set -e
# Reuse the causal adapter, original upscaler/VAE helpers, and strict 5s reader.
ultraflash_code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export H3_CAUSAL_UPSCALER_CODE_DIR="${H3_CAUSAL_UPSCALER_CODE_DIR:-$ultraflash_code_dir/../h3_causal_upscaler_20261104}"
export H3_FIVE_SECOND_CODE_DIR="${H3_FIVE_SECOND_CODE_DIR:-$ultraflash_code_dir/../h3_upscaler_5s_20261104}"
source "$H3_CAUSAL_UPSCALER_CODE_DIR/env.sh"
export EXP_CODE_DIR="$ultraflash_code_dir"
export PYTHONPATH="$EXP_CODE_DIR:$H3_FIVE_SECOND_CODE_DIR:$PYTHONPATH"
export H3_ULTRAFLASH_PAIRS="${H3_ULTRAFLASH_PAIRS:-$EXP_ROOT/upscaler_ultraflash/pairs}"
export H3_UPSCALER_WEIGHTS="${H3_UPSCALER_WEIGHTS:-$EXP_ROOT/assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors}"
export H3_UPSCALER_INITIAL_CHECKPOINT="${H3_UPSCALER_INITIAL_CHECKPOINT:-$EXP_ROOT/upscaler_causal_finetune/aligned_past_steps50/checkpoint/model.pt}"
unset ultraflash_code_dir
