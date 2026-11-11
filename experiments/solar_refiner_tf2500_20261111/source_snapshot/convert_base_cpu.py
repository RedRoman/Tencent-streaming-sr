"""Convert the existing Ref2VA checkpoint into this experiment's private directory."""
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path('/opt/refiner-cache/h3-pretrained-causal-refiner-20261109')
SOURCE = Path('/opt/refiner-cache/models/MiniMax-H3/Ref2VA')
CONVERTER = Path('/opt/runtime-temp/h3_training_assets_20260921/runtime/diffusers-patched/scripts/convert_minimax_h3_to_diffusers.py')
OUTPUT = ROOT / 'models/ref2va/transformer'

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    import torch
    import diffusers
    torch.set_num_threads(8)
    started = time.time()
    if OUTPUT.exists():
        raise RuntimeError(f'Refusing to overwrite an existing conversion: {OUTPUT}')
    OUTPUT.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location('h3_official_converter', CONVERTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(json.dumps({'event':'conversion_start','time':started,'source':str(SOURCE),'output':str(OUTPUT)}), flush=True)
    mod.convert_transformer(str(SOURCE), str(OUTPUT), mod.MINIMAX_H3_TRANSFORMER_CONFIG, 5_000_000_000)
    mod.write_transformer_config(str(OUTPUT), mod.MINIMAX_H3_TRANSFORMER_CONFIG, diffusers.__version__)
    files = {p.name: {'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUTPUT.iterdir()) if p.is_file()}
    record = {'source':str(SOURCE),'source_index_sha256':sha(SOURCE/'transformer/model.safetensors.index.json'),
              'converter':str(CONVERTER),'converter_sha256':sha(CONVERTER),'files':files,
              'started_unix':started,'finished_unix':time.time(),'new_optimizer_updates':0}
    (ROOT/'evidence/base_conversion.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'event':'conversion_complete','time':time.time(),'output_files':len(files)}),flush=True)

if __name__ == '__main__':
    main()
