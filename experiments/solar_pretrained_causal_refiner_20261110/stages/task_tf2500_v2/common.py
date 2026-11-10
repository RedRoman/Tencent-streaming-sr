import hashlib
import json
from pathlib import Path
import torch

ROOT = Path('/opt/refiner-cache/h3-pretrained-causal-refiner-20261109')
VENDOR = ROOT/'vendor/SolarRenderer-0ca4f5e45b062cc9e9aee506de46aabdc22e4c3d'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)

def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value

def schedule(step, config):
    import random
    update=step+1
    if update % config['full_frame_every']==0:
        ordinal=update//config['full_frame_every']-1
        return {'sample_index':ordinal%2,'latent_start':((ordinal//2)%12)*5,
                'latent_count':12,'crop':None,'full_frame':True,'noise_seed':2026111000+step}
    h,w=config['patch_grid']
    gen=random.Random(config['seed']+step//2)
    y=2*gen.randrange((68-h)//2+1);x=2*gen.randrange((120-w)//2+1)
    return {'sample_index':step%2,'latent_start':((step//2)%13)*5,
            'latent_count':7,'crop':[y,x,h,w],'full_frame':False,'noise_seed':2026111000+step}
