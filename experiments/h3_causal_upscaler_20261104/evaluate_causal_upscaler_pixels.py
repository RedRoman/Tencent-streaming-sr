"""Decode and compare frozen causal-upscaler variants, without any refiner.

The primary target is the same FP32 VAE's matched HR-latent reconstruction.
Source RGB is secondary because it includes VAE reconstruction/alignment error.
The VAE remains noncausal: decoded pixels assess quality, not end-to-end causality.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from data_utils import (decode_latent, load_video_vae, select_device, synchronize,
                        tensor_load, video_tensor, write_mp4)
from decode_results import comparisons


VARIANTS = ('reference', 'trilinear', 'pretrained', 'causal_shifted', 'causal_aligned')
COMPARISON_ORDER = ('pretrained', 'causal_shifted', 'causal_aligned', 'reference')


def metric(output, reference):
    if output.shape != reference.shape or output.ndim != 4 or len(output) < 2:
        raise ValueError('Metrics require equal [T,H,W,C] videos with at least two frames')
    a, b = output.float() / 255., reference.float() / 255.
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise FloatingPointError('Nonfinite pixels')
    error = (a-b).square().mean().item()
    temporal = ((a[1:]-a[:-1]) - (b[1:]-b[:-1])).square().mean().item()
    return {'mse': error, 'psnr_db': -10*math.log10(max(error, 1e-15)),
            'temporal_difference_mse': temporal}


def save_summary(path, summary):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


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
    rows = []
    for split in ('train', 'holdout'):
        for line in (args.pairs / f'{split}.jsonl').read_text().splitlines():
            if line.strip():
                rows.append((split, json.loads(line)))
    ids = [row['sample_id'] for _, row in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Manifest sample IDs must be nonempty and train/holdout disjoint')
    for sample_id in ids:
        if not isinstance(sample_id, str) or Path(sample_id).name != sample_id or sample_id in ('.', '..'):
            raise ValueError('Sample IDs must be safe directory names')
        for label in VARIANTS:
            path = args.trial / 'latents' / sample_id / f'{label}.pt'
            if not path.is_file():
                raise FileNotFoundError(path)
    args.output.mkdir(parents=True)
    device = select_device(args.device)
    vae = load_video_vae(args.base_model, device)
    summary = {
        'status': 'decoding',
        'primary_target': 'decoded matched HR reference latent, uint8 RGB before compression',
        'secondary_target': 'first43 source HR RGB frames; includes VAE reconstruction/alignment error',
        'metric_domain': 'RGB values divided by 255; all pixels/frames/channels equally weighted',
        'vae_dtype': 'float32', 'no_refiner': True, 'no_optimization': True,
        'causality_scope': 'Only the upscaler relative to fixed input latents; VAE remains noncausal',
        'pixel_metrics_prove_causality': False,
        'variant_semantics': {
            'trilinear': 'Direct spatial interpolation of normalized input latents',
            'pretrained': 'Published noncausal upscaler with Open-H3 N/U/D convention',
            'causal_shifted': 'Left temporal padding and per-frame GroupNorm, original kernel values unchanged',
            'causal_aligned': 'Causal operators with old past/current kernel offsets retained and future taps discarded',
            'reference': 'Matched HR latent, comparison target only'},
        'comparison_order': list(COMPARISON_ORDER), 'samples': {},
        'limitations': [
            'No refiner, optimizer or adaptation is evaluated in this frozen surgery trial.',
            'Train/holdout labels are inherited from the paired-data trial; neither split is optimized here.',
            'Quality metrics and contact sheets cannot establish upscaler causality; use the separate latent probe.',
            'The noncausal VAE may mix time context; this is not an end-to-end pixel streaming test.',
            'Three source-video-derived pairs do not establish generalization to generated draft latents.']}
    summary_path = args.output / 'pixel_metrics.json'
    save_summary(summary_path, summary)
    for split, row in rows:
        sample_id = row['sample_id']
        path = Path(row['path'])
        pair = tensor_load(path if path.is_absolute() else args.pairs / path)
        metadata = pair['metadata']
        if metadata['decoded_frames'] != 43 or metadata['fps'] != 24.:
            raise ValueError('This controlled trial expects 43 decoded frames at 24 fps')
        output_dir = args.output / sample_id
        output_dir.mkdir()
        frames, runtimes, codecs = {}, {}, {}
        for label in VARIANTS:
            payload = tensor_load(args.trial / 'latents' / sample_id / f'{label}.pt')
            latent = video_tensor(payload['latent'], name=label)
            if tuple(latent.shape) != (24, 13, 24, 42):
                raise ValueError(f'{sample_id}/{label}: expected [24,13,24,42] latent')
            if label == 'reference' and not torch.equal(latent.float(), pair['reference'].float()):
                raise ValueError('Trial reference does not match the paired-data HR target')
            start = time.perf_counter()
            frames[label] = decode_latent(vae, latent, device, keep_frames=metadata['decoded_frames'])
            synchronize(device)
            runtimes[label] = time.perf_counter()-start
            if tuple(frames[label].shape) != (43, 384, 672, 3):
                raise ValueError(f'{sample_id}/{label}: unexpected decoded geometry {frames[label].shape}')
            codecs[label] = write_mp4(frames[label], output_dir / f'{label}.mp4', fps=metadata['fps'])
            print(json.dumps({'decoded': sample_id, 'variant': label, 'seconds': runtimes[label]}), flush=True)
        sample_metrics = {'split': split, 'decode_seconds': runtimes, 'video_codecs': codecs,
                          'reference_matches_pair': True, 'variants': {}}
        for label in VARIANTS:
            if label != 'reference':
                sample_metrics['variants'][label] = {
                    'vs_decoded_reference': metric(frames[label], frames['reference']),
                    'vs_source_rgb_secondary': metric(frames[label], pair['source_pixels'])}
        sample_metrics['reference_vs_source_rgb_secondary'] = metric(frames['reference'], pair['source_pixels'])
        comparisons({label: frames[label] for label in COMPARISON_ORDER}, output_dir,
                    fps=metadata['fps'], crf=16, comparison_video=True)
        summary['samples'][sample_id] = sample_metrics
        save_summary(summary_path, summary)
        print(json.dumps({'sample_id': sample_id, **sample_metrics}), flush=True)
    summary['status'] = 'completed'
    save_summary(summary_path, summary)
    print(json.dumps({'status': 'completed', 'samples': len(rows), 'summary': str(summary_path)}), flush=True)


if __name__ == '__main__':
    main()
