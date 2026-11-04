#!/usr/bin/env bash
set -e
# Configure these locations for the deployment before running an entry point.
export EXP_CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${EXP_ROOT:?Set EXP_ROOT to the external experiment working directory}"
: "${H3_BASE_MODEL:?Set H3_BASE_MODEL to the H3 base directory}"
: "${H3_UPSTREAM_DIR:?Set H3_UPSTREAM_DIR to the directory containing open_h3}"
: "${H3_DIFFUSERS_SRC:?Set H3_DIFFUSERS_SRC to the compatible patched diffusers src directory}"
export EXP_ROOT H3_BASE_MODEL H3_UPSTREAM_DIR H3_DIFFUSERS_SRC
if [[ -n "${H3_CANN_ENV:-}" ]]; then
    source "$H3_CANN_ENV"
fi
if [[ -n "${H3_PYTHON_BIN:-}" ]]; then
    export PATH="$H3_PYTHON_BIN:$PATH"
fi
if [[ -n "${H3_FFMPEG_BIN:-}" ]]; then
    export PATH="$H3_FFMPEG_BIN:$PATH"
fi
export PYTHONPATH="$EXP_CODE_DIR:$H3_UPSTREAM_DIR:$H3_DIFFUSERS_SRC${PYTHONPATH:+:$PYTHONPATH}"
export HCCL_CONNECT_TIMEOUT=600 HCCL_EXEC_TIMEOUT=1800
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export DIFFUSERS_ATTN_BACKEND=native
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
