"""Decode isolated upscaler outputs and measure pixels before video compression.

Primary target is the same VAE's HR-latent reconstruction. Source RGB PSNR is
secondary because it includes VAE reconstruction/temporal alignment error.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from data_utils import decode_latent, load_video_vae, select_device, tensor_load, write_mp4
from decode_results import comparisons


def metric(output, reference):
    a, b = output.float() / 255., reference.float() / 255.
    error = (a-b).square().mean().item()
    temporal = ((a[1:]-a[:-1]) - (b[1:]-b[:-1])).square().mean().item()
    return {'mse': error, 'psnr_db': -10*math.log10(max(error, 1e-15)),
            'temporal_difference_mse': temporal}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--trial', type=Path, required=True)
    p.add_argument('--base-model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='npu:0')
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    args.output.mkdir(parents=True)
    rows = []
    for split in ('train', 'holdout'):
        for line in (args.pairs / f'{split}.jsonl').read_text().splitlines():
            row = json.loads(line)
            rows.append((split, row))
    device = select_device(args.device)
    vae = load_video_vae(args.base_model, device)
    summary = {'primary_target': 'decoded matched HR reference latent, uint8 RGB before compression',
               'secondary_target': 'first43 source HR RGB frames; includes VAE reconstruction/alignment error',
               'no_refiner': True, 'samples': {}}
    for split, row in rows:
        sample_id = row['sample_id']
        path = Path(row['path'])
        pair = tensor_load(path if path.is_absolute() else args.pairs / path)
        metadata = pair['metadata']
        output_dir = args.output / sample_id
        output_dir.mkdir()
        frames, runtimes = {}, {}
        for label in ('reference', 'trilinear', 'pretrained', 'finetuned'):
            latent = tensor_load(args.trial / 'latents' / sample_id / f'{label}.pt')['latent']
            start = time.perf_counter()
            frames[label] = decode_latent(vae, latent, device, keep_frames=metadata['decoded_frames'])
            runtimes[label] = time.perf_counter()-start
            write_mp4(frames[label], output_dir / f'{label}.mp4', fps=metadata['fps'])
            print(json.dumps({'decoded': sample_id, 'variant': label, 'seconds': runtimes[label]}), flush=True)
        sample_metrics = {'split': split, 'decode_seconds': runtimes, 'variants': {}}
        for label in ('trilinear', 'pretrained', 'finetuned'):
            sample_metrics['variants'][label] = {
                'vs_decoded_reference': metric(frames[label], frames['reference']),
                'vs_source_rgb_secondary': metric(frames[label], pair['source_pixels'])}
        sample_metrics['reference_vs_source_rgb_secondary'] = metric(frames['reference'], pair['source_pixels'])
        ordered = {label: frames[label] for label in ('trilinear', 'pretrained', 'finetuned', 'reference')}
        comparisons(ordered, output_dir, fps=metadata['fps'], crf=16, comparison_video=True)
        summary['samples'][sample_id] = sample_metrics
        (args.output / 'pixel_metrics.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps({'sample_id': sample_id, **sample_metrics}), flush=True)


if __name__ == '__main__':
    main()
