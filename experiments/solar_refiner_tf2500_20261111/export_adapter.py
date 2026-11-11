"""CPU-only audited export; never publishes the original training checkpoint."""
import argparse, hashlib, json, math, os
from pathlib import Path
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import torch
from safetensors.torch import save_file, load_file

EXPECTED = 'aec47799fbdf3425b8d11828653eaccc0349faa011919a8bd2fbe9fd9a9de422'
CONFIG = 'a652cfef6cbf68b3377c287f4f5c65858706f63812829090a07342b039c6fd45'
MODEL_INDEX = '5ba637cf6e4cbab0fcfd223646cecddcaf831a46b16256109a5c92353d7e018e'

def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    r, out = a.root, a.output
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = r/'runs/task_tf2500_v2/checkpoint_2500.pt'
    receipt = json.loads(checkpoint.with_suffix('.json').read_text())
    status = json.loads((r/'runs/task_tf2500_v2/status.json').read_text())
    launch = json.loads((r/'launches/task_tf2500_v2/launch.json').read_text())
    assert status['optimizer_updates'] == 2500
    assert status['status'] == 'training_complete_evaluation_pending'
    assert launch['status'] == 'complete' and launch['exit_code'] == 0 and launch['child_reaped']
    assert all(not (Path('/proc')/str(launch[k])).exists() for k in ('parent_pid', 'child_pid'))
    assert checkpoint.stat().st_size == receipt['bytes'] == 482164949
    assert sha(checkpoint) == receipt['sha256'] == EXPECTED
    p = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert p['format'] == 'solar-task-lora-v2' and p['step'] == 2500
    assert p['config_sha256'] == CONFIG and p['pretrained_model_index_sha256'] == MODEL_INDEX
    state = p['adapter']
    assert len(state) == len(p['parameter_names']) == len(p['optimizer']['state']) == 400
    assert list(state) == p['parameter_names']
    for name, tensor in state.items():
        assert isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32
        assert tensor.device.type == 'cpu' and torch.isfinite(tensor).all(), name
        assert name.endswith(('.a', '.b')), name
    assert sum(t.numel() for t in state.values()) == 40140800
    for slot in p['optimizer']['state'].values():
        assert int(slot['step']) == 2500
        for key in ('exp_avg', 'exp_avg_sq'):
            assert torch.isfinite(slot[key]).all()
    assert p['torch_rng'].dtype == p['npu_rng'].dtype == torch.uint8
    rng_before = torch.get_rng_state()
    torch.set_rng_state(p['torch_rng'])
    assert torch.equal(torch.get_rng_state(), p['torch_rng'])
    torch.set_rng_state(rng_before)
    target = out/'solar_refiner_tf2500_lora_rank16_fp32.safetensors'
    save_file({n:t.contiguous() for n,t in state.items()}, str(target), metadata={
        'format':'solar-task-lora-inference-v1', 'step':'2500', 'rank':'16', 'alpha':'16',
        'training_checkpoint_sha256':EXPECTED, 'training_config_sha256':CONFIG,
        'pretrained_model_index_sha256':MODEL_INDEX, 'quality_accepted':'false',
        'upstream_revision':'0ca4f5e45b062cc9e9aee506de46aabdc22e4c3d'})
    readback = load_file(str(target), device='cpu')
    assert set(readback) == set(state)
    assert all(readback[n].dtype == t.dtype and readback[n].shape == t.shape and torch.equal(readback[n], t) for n,t in state.items())
    audit = {'status':'passed', 'step':2500, 'checkpoint_sha256':EXPECTED,
        'checkpoint_bytes':checkpoint.stat().st_size, 'config_sha256':CONFIG,
        'pretrained_model_index_sha256':MODEL_INDEX, 'inference_file':target.name,
        'inference_sha256':sha(target), 'inference_bytes':target.stat().st_size,
        'tensor_count':len(state), 'parameter_count':sum(t.numel() for t in state.values()),
        'dtype':'float32', 'optimizer_states_checked':400, 'optimizer_steps':[2500],
        'optimizer_moments_finite':True, 'cpu_rng_restore_exact':True,
        'npu_rng_saved_bytes':p['npu_rng'].numel(), 'npu_rng_restore_tested_here':False,
        'roundtrip_names_shapes_dtypes_values_exact':True, 'contains_optimizer_rng_cache':False,
        'npu_forwards':0, 'new_optimizer_updates':0, 'inference_quality_accepted':False,
        'next_schedule':p['next_schedule'], 'parameter_shapes':{n:list(t.shape) for n,t in state.items()}}
    (out/'adapter_export_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (out/'SHA256SUMS').write_text(sha(target)+'  '+target.name+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='parameter_shapes'},indent=2))

if __name__ == '__main__':
    main()
