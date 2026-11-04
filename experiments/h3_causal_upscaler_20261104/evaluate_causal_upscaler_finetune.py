"""Compare isolated causal upscaler before/after fine-tuning with a fixed VAE.

No refiner is loaded. Pixel quality does not establish end-to-end causality.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from data_utils import decode_latent, load_video_vae, select_device, synchronize, tensor_load, write_mp4
from decode_results import comparisons
from evaluate_causal_upscaler_pixels import metric, save_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--trial', type=Path, required=True)
    p.add_argument('--base-model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='npu:0')
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    rows = []
    for split in ('train', 'holdout'):
        for line in (args.pairs / f'{split}.jsonl').read_text().splitlines():
            if line.strip(): rows.append((split, json.loads(line)))
    labels = ('reference', 'trilinear', 'pretrained', 'causal_before', 'causal_after')
    ids = [row['sample_id'] for _,row in rows]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError('Empty or overlapping samples')

    def latent_path(sample_id, label):
        root = args.baseline if label in ('trilinear', 'pretrained') else args.trial
        return root / 'latents' / sample_id / f'{label}.pt'

    for _,row in rows:
        if Path(row['sample_id']).name != row['sample_id']:
            raise ValueError('Invalid sample ID')
        for label in labels:
            if not latent_path(row['sample_id'], label).is_file():
                raise FileNotFoundError(latent_path(row['sample_id'], label))
    args.output.mkdir(parents=True)
    device = select_device(args.device)
    vae = load_video_vae(args.base_model, device)
    summary = {'status': 'decoding', 'no_refiner': True,
        'primary_target': 'matched HR latent decoded by the same fixed FP32 VAE, uint8 RGB before compression',
        'secondary_target': 'source RGB, includes VAE reconstruction/alignment error',
        'causality_scope': 'Upstream VAE remains noncausal; pixel metrics are quality checks only.',
        'comparison_order': ['pretrained','causal_before','causal_after','reference'],
        'samples': {}}
    for split,row in rows:
        sample_id = row['sample_id']
        path = Path(row['path'])
        pair = tensor_load(path if path.is_absolute() else args.pairs / path)
        metadata = pair['metadata']
        if metadata['decoded_frames'] != 43 or metadata['fps'] != 24.:
            raise ValueError('Expected 43 frames at 24 fps')
        directory = args.output / sample_id
        directory.mkdir()
        frames, times = {}, {}
        for label in labels:
            value = tensor_load(latent_path(sample_id, label))['latent']
            if value.shape != (24,13,24,42) or not torch.isfinite(value).all():
                raise ValueError('Invalid latent')
            if label == 'reference' and not torch.equal(value.float(), pair['reference'].float()):
                raise ValueError('Reference mismatch')
            before = time.perf_counter()
            frames[label] = decode_latent(vae, value, device, keep_frames=43)
            synchronize(device)
            times[label] = time.perf_counter()-before
            write_mp4(frames[label], directory / f'{label}.mp4', fps=24.)
            print(json.dumps({'decoded':sample_id,'variant':label,'seconds':times[label]}),flush=True)
        record = {'split':split,'decode_seconds':times,'variants':{}}
        for label in labels[1:]:
            record['variants'][label] = {
                'vs_decoded_reference': metric(frames[label],frames['reference']),
                'vs_source_rgb_secondary': metric(frames[label],pair['source_pixels'])}
        record['reference_vs_source_rgb_secondary'] = metric(frames['reference'],pair['source_pixels'])
        comparisons({label:frames[label] for label in summary['comparison_order']},directory,
                    fps=24.,crf=16,comparison_video=True)
        summary['samples'][sample_id] = record
        save_summary(args.output/'pixel_metrics.json',summary)
        print(json.dumps({'sample_id':sample_id,**record}),flush=True)
    summary['status'] = 'completed'
    save_summary(args.output/'pixel_metrics.json',summary)


if __name__ == '__main__':
    main()
