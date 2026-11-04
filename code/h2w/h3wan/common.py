import os, json, random
from pathlib import Path
import numpy as np
import torch
import torch_npu

ROOT = Path(os.environ.get('EXPERIMENT_ROOT', '/cache/songhua/h3_to_wan_latent'))
H3 = ROOT / 'assets/video_vae'
WAN = ROOT / 'assets/wan/Wan2.1_VAE.pth'

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)

def device():
    d = torch.device('npu:0')
    torch.npu.set_device(d)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.set_num_threads(8)
    return d

def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2)); tmp.replace(path)

def manifest(split):
    return json.loads((ROOT / 'manifests' / f'{split}.json').read_text())

def geometry(rgb, h3, main, tail):
    target = torch.cat((main, tail), dim=2)
    assert rgb.shape[:3] == (1, 3, 22), rgb.shape
    assert h3.shape[:3] == (1, 24, 7), h3.shape
    assert main.shape[:3] == (1, 16, 5), main.shape
    assert tail.shape[:3] == (1, 16, 2), tail.shape
    assert target.shape[:3] == (1, 16, 7), target.shape
    assert target.shape[-2] == h3.shape[-2] * 2
    assert target.shape[-1] == h3.shape[-1] * 2
    for x in (h3, main, tail):
        assert torch.isfinite(x).all().item(), 'nonfinite latent'
    return target
