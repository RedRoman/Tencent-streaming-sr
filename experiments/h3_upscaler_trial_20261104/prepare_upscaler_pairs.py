"""Make matched source-video LR/HR latent pairs for an isolated upscaler trial.

Both resolutions use exactly the same 64 source frames and the same VAE
temporal context. The old cached full-video HR latents are not training targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data_utils import (encode_pixels, load_video_vae, read_video,
                        released_decoded_frames, select_device, tensor_load)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--existing-data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-model', required=True)
    p.add_argument('--device', default='npu:0')
    p.add_argument('--source-frames', type=int, default=64)
    p.add_argument('--latent-frames', type=int, default=13)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory; matched pairs are immutable')
    rows = []
    for split in ('train', 'holdout'):
        for line in (args.existing_data / f'{split}.jsonl').read_text().splitlines():
            item = json.loads(line)
            sample_path = Path(item['path'])
            if not sample_path.is_absolute():
                sample_path = args.existing_data / sample_path
            sample = tensor_load(sample_path)
            meta = sample['metadata']
            source = Path(meta['source_video_path'])
            actual_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            if meta.get('source_video_sha256') and actual_sha != meta['source_video_sha256']:
                raise ValueError(f'Source hash mismatch: {item["id"]}')
            rows.append((split, item['id'], source, actual_sha, sample))
    if len({row[1] for row in rows}) != len(rows):
        raise ValueError('Train/holdout overlap')
    args.output.mkdir(parents=True)
    (args.output / 'samples').mkdir()
    device = select_device(args.device)
    vae = load_video_vae(args.base_model, device)
    manifests = {'train': [], 'holdout': []}
    summaries = []
    for split, sample_id, source, sha, old in rows:
        start = time.perf_counter()
        hr_pixels = read_video(source, width=672, height=384,
                               frames=args.source_frames, fps=24.)
        # Exact 2x2 area average after the common cover-resize/crop, then uint8.
        lr_pixels = F.avg_pool2d(hr_pixels.permute(0, 3, 1, 2).float(), 2, 2)
        lr_pixels = lr_pixels.round().clamp(0, 255).byte().permute(0, 2, 3, 1).contiguous()
        draft = encode_pixels(vae, lr_pixels, device, args.seed)[:, :args.latent_frames].contiguous()
        reference = encode_pixels(vae, hr_pixels, device, args.seed)[:, :args.latent_frames].contiguous()
        if draft.shape != (24, args.latent_frames, 12, 21) or reference.shape != (24, args.latent_frames, 24, 42):
            raise ValueError('Unexpected paired latent geometry')
        decoded_frames = released_decoded_frames(args.latent_frames)
        meta = {'id': sample_id, 'split': split, 'source_video_path': str(source),
                'source_video_sha256': sha, 'source_frames_encoded': args.source_frames,
                'latent_frames': args.latent_frames, 'decoded_frames': decoded_frames,
                'fps': 24., 'draft_pixels': [336, 192], 'target_pixels': [672, 384],
                'posterior': 'sample', 'posterior_seed': args.seed,
                'normalization': '(raw-vae.latents_mean)/vae.latents_std',
                'pairing': 'Common HR RGB cover-area resize/center crop at 672x384; exact 2x2 RGB area average and round to uint8 for LR; both VAE-encode the same first64 source frames then crop first13 latent frames',
                'base_model': str(args.base_model),
                'scope': 'two training clips and one internally held-out clip; source manifest itself is a training set; source VAE encodes are proxies for generated draft latents'}
        target = args.output / 'samples' / f'{sample_id}.pt'
        torch.save({'draft': draft, 'reference': reference,
                    'source_pixels': hr_pixels[:decoded_frames].contiguous(), 'metadata': meta}, target)
        manifests[split].append({'sample_id': sample_id, 'path': str(target), 'split': split})
        summary = {'sample_id': sample_id, 'split': split,
                   'draft_shape': list(draft.shape), 'reference_shape': list(reference.shape),
                   'old_reference_difference_rms': float((reference - old['reference'].float()).square().mean().sqrt()),
                   'old_draft_difference_rms': float((draft - old['draft'].float()).square().mean().sqrt()),
                   'pair_file_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                   'seconds': time.perf_counter() - start, 'metadata': meta}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    for split, items in manifests.items():
        (args.output / f'{split}.jsonl').write_text(''.join(json.dumps(item)+'\n' for item in items))
    (args.output / 'pair_summary.json').write_text(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
