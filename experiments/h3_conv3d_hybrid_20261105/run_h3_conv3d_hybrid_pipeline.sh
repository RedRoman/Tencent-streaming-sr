#!/usr/bin/env bash
# Start only after the separate causal audit and pixel preflight have passed.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
physical="${1:?Supply a verified idle physical NPU ID}"
destination="$H3_CONV3D_HYBRID_HOME"
python - "$destination" <<'PY'
import json, sys, hashlib
from pathlib import Path
from evaluate_h3_cmn import module_source
p = Path(sys.argv[1])
s = json.loads((p/'preflight_v1/summary.json').read_text())
a = json.loads((p/'causal_prior_hf32_off.json').read_text())
assert s['status'] == 'preflight_completed' and s['completed_updates'] == 8
assert s['frozen_modules_unchanged'] and s['source_contract_unchanged']
assert s['execution_precision_unchanged']
assert s['initial_causality']['passed'] and s['causality']['passed']
assert a['all_passed'] and a['sources_unchanged'] and a['mode'] == 'actual_345m_checkpoint'
assert a['checkpoint']['sha256'] == s['contract']['initial_sha256']
assert a['execution_precision']['conv_allow_hf32'] is False
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
for name, digest in s['contract']['source_sha256'].items():
    assert sha(module_source(name)) == digest, name
for record in [*a['sources'].values(), a['precision_driver']]:
    assert sha(record['path']) == record['sha256'], record['path']
assert abs(s['initial_dev']['mean']['latent_mse'] - 0.10832573473453522) < 1e-6
default = json.loads((p/'causal_prior_long.json').read_text())
for name, check in default['checks'].items():
    if name.startswith(('future_', 'prefix_')) or name == 'input_gradient':
        assert check['passed'], name
assert not (p/'train.log').exists()
PY
bash "$EXP_CODE_DIR/run_train_h3_conv3d_hybrid.sh" "$physical" train
python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$physical" -- \
  python "$EXP_CODE_DIR/audit_h3_conv3d_precision.py" --hf32-conv false --device npu:0 \
  --upstream "$H3_UPSTREAM_DIR" \
  --weights "$H3_UPSCALER_WEIGHTS" \
  --initial "$destination/kd_pixel_steps2000/best.pt" --checkpoint-kind hybrid \
  --output "$destination/causal_final_hf32_off.json" \
  > "$destination/causal_final_hf32_off.log" 2>&1
python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$physical" -- \
  python "$EXP_CODE_DIR/evaluate_h3_conv3d_hybrid.py" --device npu:0 \
  > "$destination/evaluate_latents.log" 2>&1
python "$H3_REFINER_CODE_DIR/locked_run.py" --devices "$physical" -- \
  python "$EXP_CODE_DIR/evaluate_h3_conv3d_hybrid.py" --device npu:0 \
  --decode-only --num-shards 1 --shard-index 0 \
  --base-model "$H3_BASE_MODEL" \
  > "$destination/decode.log" 2>&1
python "$EXP_CODE_DIR/evaluate_h3_conv3d_hybrid.py" --finalize --num-shards 1 \
  > "$destination/finalize.log" 2>&1
python "$EXP_CODE_DIR/plot_h3_conv3d_hybrid.py" \
  --run "$destination/kd_pixel_steps2000" --output "$destination/training_curves.png" \
  > "$destination/plot.log" 2>&1
