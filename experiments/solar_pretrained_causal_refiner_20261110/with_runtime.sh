#!/usr/bin/env bash
set -euo pipefail
source /opt/shared-runtime/envs/minimax-h3-npu/Ascend/cann-9.0.1/set_env.sh
export PYTHONDONTWRITEBYTECODE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OMP_NUM_THREADS=8
export PYTHONPATH=/opt/refiner-cache/h3-pretrained-causal-refiner-20261109:/opt/refiner-cache/h3-pretrained-causal-refiner-20261109/vendor/SolarRenderer-0ca4f5e45b062cc9e9aee506de46aabdc22e4c3d/src:/opt/runtime-temp/h3_training_assets_20260921/runtime/diffusers-patched/src:${PYTHONPATH:-}
exec /opt/runtime-temp/h3_training_assets_20260921/runtime/.venv-npu/bin/python -u "$@"
