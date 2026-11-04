"""Materialize official refiner delta once on CPU, retaining original shard layout."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(24)
    base, assets, out = Path(args.base) / 'transformer', Path(args.assets), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((base / 'diffusion_pytorch_model.safetensors.index.json').read_text())
    lora_path = assets / 'lora/open_h3_regenerate_2k_lora.safetensors'
    with lora_path.open('rb') as stream:
        lora_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert lora_sha == '8e36a0912ab72f7f3b00f41e3c7c97c7dfdec21e66c88ab22f08b9b0c681bb04', 'Official LoRA SHA256 mismatch'
    raw = load_file(str(lora_path))
    pairs = {k[len('transformer.'):-len('.lora_A.weight')]:
             (v.float(), raw[k.replace('.lora_A.weight', '.lora_B.weight')].float())
             for k, v in raw.items() if k.endswith('.lora_A.weight')}
    assert len(pairs) == 312 and len(pairs) * 2 == len(raw)
    del raw
    seen, records = set(), []
    started = time.time()
    for filename in sorted(set(index['weight_map'].values())):
        shard = load_file(str(base / filename))
        changed = []
        for name, weight in shard.items():
            module = name[:-len('.weight')] if name.endswith('.weight') else ''
            if module not in pairs:
                continue
            # Own the modified storage explicitly; shared base files stay read-only.
            weight = weight.clone()
            shard[name] = weight
            a, b = pairs[module]
            assert weight.shape == (b.shape[0], a.shape[1])
            for start in range(0, len(weight), 512):
                stop = min(start + 512, len(weight))
                value = weight[start:stop].float()
                value.addmm_(b[start:stop], a)
                weight[start:stop].copy_(value.to(weight.dtype))
            seen.add(module)
            changed.append(module)
        temp = out / (filename + '.partial')
        save_file(shard, str(temp), metadata={'format': 'pt'})
        temp.replace(out / filename)
        records.append({'file': filename, 'changed': len(changed), 'bytes': (out / filename).stat().st_size})
        print(json.dumps({'event': 'merged_shard', **records[-1], 'elapsed': time.time() - started}), flush=True)
        del shard
    assert seen == set(pairs), f'Unmerged LoRA targets: {set(pairs) - seen}'
    for name in ('config.json', 'diffusion_pytorch_model.safetensors.index.json'):
        shutil.copy2(base / name, out / name)
    manifest = {'base': str(base), 'lora': str(lora_path), 'lora_sha256': lora_sha,
                'release_revision': '50c09cf1df5f9262b9279868c1c666bd5fe3476b', 'targets': len(seen),
                'scale': 1.0, 'merge': 'cast(W_fp32 + B_fp32 @ A_fp32, original_dtype)',
                'sharp_curve': 'installed separately at runtime, strength 1', 'shards': records}
    (out / 'MERGED.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
