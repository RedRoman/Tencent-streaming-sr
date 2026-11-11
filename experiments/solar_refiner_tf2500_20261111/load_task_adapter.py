"""Load the released task delta without merging away its FP32 branch arithmetic.

The caller supplies the exact frozen Ref2VA + causal-control + causal-4step model.
This module performs no downloads, training, device selection, or inference jobs.
"""
import hashlib
from pathlib import Path
import torch
from torch import nn
from safetensors.torch import load_file

EXPECTED_SHA256 = '9c5cd061f3f339c5939e9568fb9d4179e27aaec7d584d9ce68be2073c9c76fc3'
MODEL_INDEX_SHA256 = '5ba637cf6e4cbab0fcfd223646cecddcaf831a46b16256109a5c92353d7e018e'

def file_sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

class TaskLoRAInference(nn.Module):
    def __init__(self, base, a, b):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError('Expected an unwrapped Linear; duplicate attachment is rejected')
        if a.dtype != torch.float32 or b.dtype != torch.float32:
            raise ValueError('The published task tensors must remain FP32')
        if tuple(a.shape) != (16, base.in_features) or tuple(b.shape) != (base.out_features, 16):
            raise ValueError('Rank or base-model shape mismatch')
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('Non-finite task adapter')
        self.base = base.requires_grad_(False)
        self.register_buffer('a', a.to(base.weight.device).clone())
        self.register_buffer('b', b.to(base.weight.device).clone())

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        output = self.base(x)
        # alpha/rank = 1. Matches TrainingRuntime.TaskLoRALinear under no autocast.
        return output + ((x.float() @ self.a.T) @ self.b.T).to(output.dtype)

def attach_task_adapter(model, adapter_path, model_directory):
    if file_sha256(adapter_path) != EXPECTED_SHA256:
        raise ValueError('Task weight SHA-256 mismatch')
    index = Path(model_directory)/'diffusion_pytorch_model.safetensors.index.json'
    if file_sha256(index) != MODEL_INDEX_SHA256:
        raise ValueError('Expected the pinned, sequentially merged causal base index')
    state = load_file(str(adapter_path), device='cpu')
    if len(model.transformer_blocks) != 50:
        raise ValueError('Expected the 50-block H3 model')
    paths = [f'transformer_blocks.{i}.{name}' for i in range(50)
             for name in ('attn.to_q', 'attn.to_k', 'attn.to_v', 'attn.to_out.0')]
    expected = {path+suffix for path in paths for suffix in ('.a', '.b')}
    if set(state) != expected:
        raise ValueError('Task tensor keys differ from the pinned 400-tensor schema')
    # Validate all replacements before mutating the supplied model.
    replacements = [(path, TaskLoRAInference(model.get_submodule(path), state[path+'.a'], state[path+'.b']))
                    for path in paths]
    for path, module in replacements:
        parent, key = path.rsplit('.', 1)
        setattr(model.get_submodule(parent), key, module)
    return model.eval().requires_grad_(False)

if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser(description='CPU-only inspection of the pinned released adapter')
    ap.add_argument('adapter', type=Path)
    args = ap.parse_args()
    assert file_sha256(args.adapter) == EXPECTED_SHA256, 'SHA-256 mismatch'
    state = load_file(str(args.adapter), device='cpu')
    assert len(state) == 400 and sum(t.numel() for t in state.values()) == 40140800
    assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in state.values())
    print(json.dumps({'step':2500, 'tensors':400, 'parameters':40140800,
                     'sha256':EXPECTED_SHA256, 'quality_accepted':False}, indent=2))
