"""v4 expanded-data preparation: add new train clips under the exact 120->37 contract.

Reuses the released-pipeline contract from prepare_upscaler_5s_pairs.py unchanged
(strict FFmpeg 120-frame proof, 2x2 area-average LR, posterior seed 42). Existing
samples from upscaler_5s/pairs are copied bit-identically (holdout 0380588 must
stay byte-identical so v1/v2/v4 remain comparable); only genuinely new source
videos are encoded here.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import prepare_upscaler_5s_pairs as base
from data_utils import (encode_pixels, load_video_vae, released_decoded_frames,
                        select_device)

SOURCE_FRAMES = base.SOURCE_FRAMES
LATENT_FRAMES = base.LATENT_FRAMES
HEIGHT, WIDTH = base.HEIGHT, base.WIDTH


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--new-train-sources', nargs='+', required=True,
                   help='absolute paths of new 5s source mp4 clips for the train split')
    p.add_argument('--existing-data', type=Path, required=True,
                   help='existing upscaler_5s/pairs directory to inherit samples from')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-model', required=True)
    p.add_argument('--device', default='npu:0')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory; old pairs are never overwritten')

    # Inherit existing samples bit-identically (train + holdout).
    args.output.mkdir(parents=True)
    (args.output / 'samples').mkdir()
    (args.output / 'source_proof').mkdir()
    manifests = {'train': [], 'holdout': []}
    inherited = []
    for split in ('train', 'holdout'):
        for line in (args.existing_data / f'{split}.jsonl').read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            src = args.existing_data / item['path']
            if not src.is_file():
                raise FileNotFoundError(src)
            digest = base.sha256(src)
            summary_rows = json.loads((args.existing_data / 'pair_summary.json').read_text())
            reference_digest = {r['sample_id']: r['pair_file_sha256'] for r in summary_rows}
            sample_id = item['sample_id']
            if digest != reference_digest[sample_id]:
                raise ValueError(f'{sample_id}: inherited sample SHA256 mismatch; refusing to continue')
            shutil.copyfile(src, args.output / item['path'])
            manifests[split].append(dict(item))
            inherited.append({'sample_id': sample_id, 'split': split, 'sha256': digest,
                              'inherited': True})
    if len(manifests['holdout']) != 1:
        raise ValueError('Expected exactly one inherited holdout sample')

    new_sources = [Path(s) for s in args.new_train_sources]
    if len(new_sources) != len(set(new_sources)):
        raise ValueError('Duplicate new train sources')
    for source in new_sources:
        if not source.is_absolute() or not source.is_file():
            raise ValueError(f'New train source must be an absolute existing file: {source}')
    inherited_ids = {row['sample_id'] for row in inherited}
    for source in new_sources:
        if source.stem in inherited_ids:
            raise ValueError(f'{source.stem}: already present in inherited data')

    print(json.dumps({'inherited': inherited, 'new_train_clips': [s.name for s in new_sources]}), flush=True)

    # Strict source proof + read for new clips (same contract as the base script).
    rows = []
    for source in new_sources:
        sample_id = source.stem
        proof = {'sample_id': sample_id, 'source_video_path': str(source),
                 'source_video_sha256': base.sha256(source),
                 'original_metadata_sha256': None,
                 'full_source_count': base.count_source_frames(
                     source, args.output / 'source_proof', sample_id)}
        base.save_json(args.output / 'source_proof' / f'{sample_id}.json', proof)
        if not proof['full_source_count']['has_at_least_120_source_and_fps_frames']:
            raise ValueError(f'{sample_id}: source insufficient for 120 real frames at 24 fps')
        row = {'sample_id': sample_id, 'source': source, 'source_sha256': proof['source_video_sha256'],
               'original_sha256': None}
        row['pixels'], proof['strict_reader'] = base.read_first_120_strict(
            source, args.output / 'source_proof', sample_id)
        proof['passed'] = True
        base.save_json(args.output / 'source_proof' / f'{sample_id}.json', proof)
        row['proof'] = proof
        rows.append(row)
        print(json.dumps({'source_proof_passed': sample_id,
                          'counted_frames': proof['full_source_count']['counted_output_frames'],
                          'used_frames': SOURCE_FRAMES}), flush=True)

    device = select_device(args.device)
    vae = load_video_vae(args.base_model, device)
    if vae.config.clip_length != base.VAE_CLIP_LENGTH or \
            released_decoded_frames(LATENT_FRAMES) != base.RAW_DECODED_FRAMES:
        raise ValueError('Released VAE temporal geometry does not match the 120->37->124 contract')

    summaries = json.loads((args.existing_data / 'pair_summary.json').read_text())
    for row in rows:
        started = time.perf_counter()
        sample_id = row['sample_id']
        hr_pixels = row.pop('pixels')
        lr_pixels = F.avg_pool2d(hr_pixels.permute(0, 3, 1, 2).float(), 2, 2)
        lr_pixels = lr_pixels.round().clamp(0, 255).byte().permute(0, 2, 3, 1).contiguous()
        draft = encode_pixels(vae, lr_pixels, device, args.seed)
        reference = encode_pixels(vae, hr_pixels, device, args.seed)
        if tuple(draft.shape) != (24, LATENT_FRAMES, 12, 21) or \
                tuple(reference.shape) != (24, LATENT_FRAMES, 24, 42):
            raise ValueError(f'{sample_id}: VAE encoded unexpected shapes {draft.shape}, {reference.shape}')
        metadata = {
            'id': sample_id, 'sample_id': sample_id, 'split': 'train',
            'source_video_path': str(row['source']), 'source_video_sha256': row['source_sha256'],
            'original_video_sha256': row['source_sha256'],
            'source_frames': SOURCE_FRAMES, 'source_frames_encoded': SOURCE_FRAMES,
            'output_frames': SOURCE_FRAMES, 'decoded_frames': SOURCE_FRAMES,
            'latent_frames': LATENT_FRAMES, 'raw_decoded_frames': base.RAW_DECODED_FRAMES,
            'vae_internal_pad_frames': base.VAE_INTERNAL_PAD_FRAMES,
            'vae_padded_input_frames': base.VAE_PADDED_INPUT_FRAMES,
            'vae_internal_padding': 'Released VAE repeats its last input frame to a multiple of clip_length=17',
            'decode_tail_crop_frames': base.RAW_DECODED_FRAMES - SOURCE_FRAMES,
            'fps': float(base.FPS), 'output_duration_seconds': SOURCE_FRAMES / base.FPS,
            'draft_pixels': [WIDTH // 2, HEIGHT // 2], 'target_pixels': [WIDTH, HEIGHT],
            'posterior': 'sample', 'posterior_seed': args.seed,
            'normalization': '(raw-vae.latents_mean)/vae.latents_std',
            'pairing': 'Same first120 source-timed RGB frames at24fps; HR cover-area resize/center crop; exact2x2 area average rounded to uint8 for LR',
            'external_time_padding': False, 'external_slowdown': False,
            'source_proof_path': f'source_proof/{sample_id}.json',
            'base_model': str(args.base_model), 'new_training': True,
            'scope': 'v4 expanded train split; holdout inherited bit-identically'}
        target = args.output / 'samples' / f'{sample_id}.pt'
        temporary = target.with_suffix('.pt.tmp')
        torch.save({'draft': draft, 'reference': reference, 'source_pixels': hr_pixels,
                    'metadata': metadata}, temporary)
        temporary.replace(target)
        manifests['train'].append({'sample_id': sample_id, 'path': f'samples/{sample_id}.pt',
                                   'split': 'train'})
        summary = {'sample_id': sample_id, 'split': 'train', 'draft_shape': list(draft.shape),
                   'reference_shape': list(reference.shape), 'source_pixel_shape': list(hr_pixels.shape),
                   'pair_file_sha256': base.sha256(target), 'metadata': metadata,
                   'seconds': time.perf_counter() - started}
        summaries.append(summary)
        base.save_json(args.output / 'pair_summary.json', summaries)
        print(json.dumps({k: summary[k] for k in ('sample_id', 'split', 'seconds')}), flush=True)
        del draft, reference, hr_pixels, lr_pixels

    for split, items in manifests.items():
        (args.output / f'{split}.jsonl').write_text(
            ''.join(json.dumps(item) + '\n' for item in items))
    print(json.dumps({'train_clips': len(manifests['train']),
                      'holdout_clips': len(manifests['holdout'])}), flush=True)


if __name__ == '__main__':
    main()
