"""Load inference-only causal H3 upscaler exports without pickle checkpoints.

Requires the experiment's published h3_conv3d_hybrid and causal_upscaler modules
on PYTHONPATH and the exact recorded Open-H3 upscaler.py/upscaler_node.py.
Already aligned kernels MUST NOT undergo aligned-past initialization again.
The original bidirectional node by itself does not reproduce this model.
"""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import torch
from safetensors import safe_open


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(part)
    return digest.hexdigest()


def load_release(config_path, upstream, device='cpu'):
    import causal_upscaler
    import h3_conv3d_hybrid
    from h3_conv3d_hybrid import CausalConv3dStudent, validate_adapter_config
    from prepare_data import load_author_upscaler
    config_path, upstream = Path(config_path), Path(upstream)
    config = json.loads(config_path.read_text())
    validate_adapter_config(config['adapter_config'])
    expected_model = dict(kind='h3_causal_conv3d',latent_channels=24,spatial_scale=2,
                          adapter_config=config['adapter_config'])
    if (config['schema_version'] != 1 or config['model_config'] != expected_model or
            config['state_dict_format'] != 'original_upscaler' or
            config['parameter_count'] != 345280216 or config['tensor_count'] != 322 or
            config['dtype'] != 'float32' or config['quantized'] is not False or
            config['already_aligned_weights'] is not True or config['reinitialization_forbidden'] is not True):
        raise ValueError('Wrong public architecture/precision/alignment contract')
    name = config['weights_file']
    if Path(name).name != name or not name.endswith('.safetensors'):
        raise ValueError('Weights must be the named sibling safetensors file')
    weights = config_path.parent/name
    if sha256(weights) != config['weights_sha256']:
        raise ValueError('Public weights SHA mismatch')
    for filename, module in (('causal_upscaler.py',causal_upscaler),
                             ('h3_conv3d_hybrid.py',h3_conv3d_hybrid)):
        if sha256(inspect.getfile(module)) != config['runtime_source_sha256'][filename]:
            raise ValueError('Causal runtime source differs: '+filename)
    for filename in ('upscaler.py','upscaler_node.py'):
        if sha256(upstream/'open_h3'/filename) != config['upstream_source_sha256'][filename]:
            raise ValueError('Upstream architecture/normalization source differs: '+filename)
    with safe_open(str(weights),framework='pt',device='cpu') as opened:
        metadata = opened.metadata()
        if (metadata['variant'] != config['variant'] or
                metadata['original_checkpoint_sha256'] != config['original_checkpoint_sha256'] or
                metadata['dtype'] != 'float32' or metadata['state_dict_format'] != 'original_upscaler' or
                len(opened.keys()) != 322):
            raise ValueError('Safetensors metadata differs from config')
    device = torch.device(device)
    node = load_author_upscaler(upstream,weights,device)
    original = node.load_model('upscaler',device,'fp32')
    # Existing saved tensors already underwent causal aligned-past surgery.
    # The flag records that fact; this assignment does not change any tensor.
    original._aligned_past_initialized = True
    student = CausalConv3dStudent(node,original,config['adapter_config'],activation_checkpointing=False)
    student.eval().requires_grad_(False)
    if (sum(value.numel() for value in student.parameters()) != 345280216 or
            any(value.dtype != torch.float32 for value in student.parameters())):
        raise ValueError('Loaded parameter count/dtype differs')
    return student


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--upstream',type=Path,required=True)
    args = parser.parse_args()
    # This diagnostic only constructs/loads on CPU and compares every tensor;
    # it does not run a forward pass or select/initialize an accelerator.
    torch.set_num_threads(4)
    model = load_release(args.config,args.upstream,'cpu')
    cfg = json.loads(args.config.read_text())
    state = model.original_state_dict()
    checked = 0
    with safe_open(str(args.config.parent/cfg['weights_file']),framework='pt',device='cpu') as opened:
        if set(opened.keys()) != set(state):raise AssertionError('Loaded state keys differ')
        for name,actual in state.items():
            restored = opened.get_tensor(name)
            if actual.dtype != restored.dtype or not torch.equal(actual,restored):
                raise AssertionError('Causal adapter/loader changed tensor: '+name)
            a = hashlib.sha256(actual.detach().contiguous().numpy().tobytes()).hexdigest()
            b = hashlib.sha256(restored.contiguous().numpy().tobytes()).hexdigest()
            if a != b:raise AssertionError('Loaded tensor bits differ: '+name)
            checked += 1
    print(json.dumps(dict(status='passed',scope='CPU architecture+adapter load and per-tensor equality; no forward/NPU',
        variant=cfg['variant'],tensors_checked=checked,parameter_count=345280216,
        all_tensors_bitwise_equal=True,aligned_past_reinitialization_called=False,
        frozen_inference_parameters=all(not p.requires_grad for p in model.parameters()))))


if __name__ == '__main__':
    main()
