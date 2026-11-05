"""Cache frozen original H3 teacher outputs for train/dev, never test selection."""
from pathlib import Path
import argparse
import json
import os
import time
import torch
from data_utils import select_device, tensor_load, synchronize
from prepare_data import load_author_upscaler
from train_upscaler_trial import atomic_json, atomic_torch_save, sha256, wrapper_forward
from train_upscaler_ultraflash import load_pairs


def main():
    root = Path(os.environ.get('EXP_ROOT', Path(__file__).resolve().parents[1]))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, default=root/'upscaler_ultraflash/pairs')
    p.add_argument('--upstream', type=Path, default=Path(os.environ.get('H3_UPSTREAM_DIR', root/'upstream')))
    p.add_argument('--teacher', type=Path, default=root/'assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors')
    p.add_argument('--output', type=Path, default=root/'upscaler_hybrid/teacher_cache')
    p.add_argument('--device', default='npu:0')
    args = p.parse_args()
    torch.set_num_threads(4)
    ready = json.loads((args.pairs/'READY.json').read_text())
    assert ready['status'] == 'ready' and ready['plan_sha256'] == sha256(args.pairs/'plan.json')
    for split in ('train', 'dev'):
        assert ready['manifests'][split]['sha256'] == sha256(args.pairs/f'{split}.jsonl')
    pairs = load_pairs(args.pairs/'train.jsonl') + load_pairs(args.pairs/'dev.jsonl')
    assert len(pairs) == 144
    contract = dict(plan_sha256=sha256(args.pairs/'plan.json'), teacher_sha256=sha256(args.teacher),
                    teacher='Unmodified original bidirectional H3, wrapper D(U(N(z))), FP32',
                    source_sha256=sha256(Path(__file__)))
    args.output.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    node = load_author_upscaler(args.upstream, args.teacher, device)
    model = node.load_model('upscaler', device, 'fp32').eval().requires_grad_(False)
    started = time.perf_counter()
    records = {}
    for i, pair in enumerate(pairs):
        path = args.output/f"{pair['sample_id']}.pt"
        expected = dict(sample_id=pair['sample_id'], split=pair['split'], pair_sha256=pair['pair_sha256'], **contract)
        if path.exists():
            saved = tensor_load(path)
            if saved['metadata'] != expected:
                raise ValueError(f'Stale teacher cache: {path}')
        else:
            with torch.no_grad():
                pred = wrapper_forward(node, model, pair['draft'][None].to(device), pair['reference'].shape[-3:])[0]
            synchronize(device)
            if not torch.isfinite(pred).all():
                raise FloatingPointError(pair['sample_id'])
            atomic_torch_save(path, dict(latent=pred.detach().cpu(), metadata=expected))
            del pred
        records[pair['sample_id']] = dict(path=path.name, sha256=sha256(path), split=pair['split'], pair_sha256=pair['pair_sha256'])
        if i % 10 == 0 or i == len(pairs)-1:
            print(json.dumps(dict(cached=i+1, total=len(pairs), seconds=time.perf_counter()-started)), flush=True)
    atomic_json(args.output/'READY.json', dict(status='ready', contract=contract, records=records, count=len(records)))


if __name__ == '__main__':
    main()
