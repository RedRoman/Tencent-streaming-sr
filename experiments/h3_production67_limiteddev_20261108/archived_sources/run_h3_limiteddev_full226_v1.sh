#!/usr/bin/env bash
set -euo pipefail
h3_eval_env_file=/path/to/h3-project/experiment/env.sh
h3_eval_env_sha=$(sha256sum "$h3_eval_env_file")
[[ "${h3_eval_env_sha%% *}" == 64451094fa9aca640bc5bfc9ba77080330f858d1706ba74798ca4eb8834e4524 ]]
source "$h3_eval_env_file"
export PYTHONNOUSERSITE=1
exec /path/to/model-assets/runtime/.venv-npu/bin/python /path/to/staging/h3_production67_limiteddev_eval_launch_20261107_v1/run_h3_limiteddev_full226_v1.py "$@"
