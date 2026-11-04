"""Prepare actual 5-second source-video pairs without externally padding time.

All three source clips must supply 120 output frames at 24 fps. FFmpeg counting
and strict reading use no tpad, looping, slowdown, or frame-hold fallback. The
released H3 VAE itself internally repeats the last frame: 120 -> 136 inputs,
37 latents -> 124 decoded frames; evaluation must crop the last four to 120.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import (encode_pixels, ffmpeg_executable, load_video_vae,
                        released_decoded_frames, select_device, tensor_load)


FPS = 24
SOURCE_FRAMES = 120
LATENT_FRAMES = 37
HEIGHT, WIDTH = 384, 672
VAE_CLIP_LENGTH = 17
VAE_PADDED_INPUT_FRAMES = 136
VAE_INTERNAL_PAD_FRAMES = 16
RAW_DECODED_FRAMES = 124


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def fps_statistics(stderr):
    matches = re.findall(
        r'(\d+) frames in, (\d+) frames out; (\d+) frames dropped, (\d+) frames duplicated', stderr)
    if not matches:
        raise RuntimeError('FFmpeg did not expose fps-filter counts; cannot prove source-frame availability')
    frames_in, frames_out, dropped, duplicated = map(int, matches[-1])
    return {'input_frames': frames_in, 'output_frames': frames_out,
            'dropped_frames': dropped, 'duplicated_frames': duplicated}


def count_source_frames(source, proof_directory, sample_id):
    # No output -r: frame timing is solely the fps filter applied to source PTS.
    # Passthrough output sync prevents the output layer from duplicating frames.
    command = [ffmpeg_executable(), '-hide_banner', '-nostdin', '-loglevel', 'verbose',
               '-nostats', '-i', str(source), '-map', '0:v:0', '-an', '-sn', '-dn',
               '-vf', f'fps={FPS}', '-vsync', '0', '-progress', 'pipe:1', '-f', 'null', '-']
    result = subprocess.run(command, capture_output=True, text=True)
    (proof_directory / f'{sample_id}.count.log').write_text(result.stderr)
    (proof_directory / f'{sample_id}.count.progress').write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'{sample_id}: FFmpeg source counting failed; see count.log')
    stats = fps_statistics(result.stderr)
    frame_counts = [int(value) for value in re.findall(r'^frame=(\d+)\s*$', result.stdout, flags=re.MULTILINE)]
    if not frame_counts or frame_counts[-1] != stats['output_frames']:
        raise RuntimeError(f'{sample_id}: FFmpeg progress and fps-filter frame counts disagree')
    sufficient = stats['input_frames'] >= SOURCE_FRAMES and stats['output_frames'] >= SOURCE_FRAMES
    return {'filter': f'fps={FPS}', 'output_sync': 'passthrough', 'fps_filter': stats,
            'counted_output_frames': frame_counts[-1], 'required_source_output_frames': SOURCE_FRAMES,
            'has_at_least_120_source_and_fps_frames': sufficient,
            'external_tpad': False, 'external_loop': False, 'setpts_slowdown': False,
            'note': 'Full source count; duplicate-free first120 reader is checked separately'}


def read_first_120_strict(source, proof_directory, sample_id):
    filters = (f'fps={FPS},scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase:flags=area,'
               f'crop={WIDTH}:{HEIGHT}')
    command = [ffmpeg_executable(), '-hide_banner', '-nostdin', '-loglevel', 'verbose',
               '-nostats', '-i', str(source), '-map', '0:v:0', '-an', '-sn', '-dn',
               '-vf', filters, '-frames:v', str(SOURCE_FRAMES), '-vsync', '0',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    result = subprocess.run(command, capture_output=True)
    stderr = result.stderr.decode('utf-8', errors='replace')
    (proof_directory / f'{sample_id}.read.log').write_text(stderr)
    if result.returncode:
        raise RuntimeError(f'{sample_id}: strict FFmpeg reader failed; see read.log')
    stats = fps_statistics(stderr)
    frame_bytes = WIDTH * HEIGHT * 3
    expected_bytes = SOURCE_FRAMES * frame_bytes
    proof = {'filter': filters, 'requested_frames': SOURCE_FRAMES,
             'received_full_frames': len(result.stdout) // frame_bytes,
             'received_bytes': len(result.stdout), 'expected_bytes': expected_bytes,
             'fps_filter': stats, 'output_sync': 'passthrough',
             'exact_120_frames': len(result.stdout) == expected_bytes,
             'fps_generated_duplicate_frames': stats['duplicated_frames'],
             'external_tpad': False, 'external_loop': False, 'setpts_slowdown': False}
    save_json(proof_directory / f'{sample_id}.reader.json', proof)
    if len(result.stdout) != expected_bytes:
        raise ValueError(f'{sample_id}: source yields {proof["received_full_frames"]} complete frames; '
                         '120 real frames required, padding/slowdown is forbidden')
    if stats['duplicated_frames'] != 0 or stats['input_frames'] < SOURCE_FRAMES:
        raise ValueError(f'{sample_id}: fps=24 reader duplicated {stats["duplicated_frames"]} frames '
                         f'from {stats["input_frames"]} source frames; refuse synthetic frame repetition')
    pixels = np.frombuffer(result.stdout, dtype=np.uint8).reshape(SOURCE_FRAMES, HEIGHT, WIDTH, 3).copy()
    return torch.from_numpy(pixels), proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--existing-data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-model', required=True)
    p.add_argument('--device', default='npu:0')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory; old pairs are never overwritten')
    rows = []
    for split in ('train', 'holdout'):
        for line in (args.existing_data / f'{split}.jsonl').read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            sample_id = item.get('sample_id', item.get('id'))
            if not isinstance(sample_id, str) or Path(sample_id).name != sample_id or sample_id in ('.', '..'):
                raise ValueError('Invalid sample ID')
            sample_path = Path(item['path'])
            sample = tensor_load(sample_path if sample_path.is_absolute() else args.existing_data / sample_path)
            metadata = sample['metadata']
            source = Path(metadata['source_video_path'])
            if not source.is_absolute():
                raise ValueError('Source-video metadata must give an explicit absolute path')
            digest = sha256(source)
            original_digest = metadata.get('source_video_sha256')
            if original_digest and digest != original_digest:
                raise ValueError(f'{sample_id}: original source-video SHA256 mismatch')
            rows.append({'split': split, 'sample_id': sample_id, 'source': source,
                         'source_sha256': digest, 'original_sha256': original_digest})
            del sample
    identifiers = [row['sample_id'] for row in rows]
    if len(rows) != 3 or len(set(identifiers)) != 3:
        raise ValueError('This follow-up requires the same three distinct source clips')
    if sum(row['split'] == 'train' for row in rows) != 2:
        raise ValueError('Expected inherited 2 train / 1 holdout labels; this script does not train')
    args.output.mkdir(parents=True)
    proof_directory = args.output / 'source_proof'
    proof_directory.mkdir()
    (args.output / 'samples').mkdir()

    # Establish source availability before model/device initialization. Retaining
    # three 120-frame RGB tensors on CPU avoids re-reading with a different path.
    for row in rows:
        sample_id = row['sample_id']
        proof = {'sample_id': sample_id, 'source_video_path': str(row['source']),
                 'source_video_sha256': row['source_sha256'],
                 'original_metadata_sha256': row['original_sha256'],
                 'full_source_count': count_source_frames(row['source'], proof_directory, sample_id)}
        save_json(proof_directory / f'{sample_id}.json', proof)
        if not proof['full_source_count']['has_at_least_120_source_and_fps_frames']:
            raise ValueError(f'{sample_id}: source is insufficient for 120 real frames at 24 fps; '
                             'no repetition, slowdown, or last-frame hold fallback is allowed')
        row['pixels'], proof['strict_reader'] = read_first_120_strict(row['source'], proof_directory, sample_id)
        proof['passed'] = True
        save_json(proof_directory / f'{sample_id}.json', proof)
        row['proof'] = proof
        print(json.dumps({'source_proof_passed': sample_id,
                          'counted_frames': proof['full_source_count']['counted_output_frames'],
                          'used_frames': SOURCE_FRAMES, 'fps_duplicates_in_used_reader': 0}), flush=True)

    device = select_device(args.device)
    vae = load_video_vae(args.base_model, device)
    if vae.config.clip_length != VAE_CLIP_LENGTH or released_decoded_frames(LATENT_FRAMES) != RAW_DECODED_FRAMES:
        raise ValueError('Released VAE temporal geometry does not match the 120->37->124 contract')
    manifests, summaries = {'train': [], 'holdout': []}, []
    for row in rows:
        started = time.perf_counter()
        split, sample_id = row['split'], row['sample_id']
        hr_pixels = row.pop('pixels')
        lr_pixels = F.avg_pool2d(hr_pixels.permute(0, 3, 1, 2).float(), 2, 2)
        lr_pixels = lr_pixels.round().clamp(0, 255).byte().permute(0, 2, 3, 1).contiguous()
        # Keep the actual 37 latent frames; never crop back to the old 13-frame trial.
        draft = encode_pixels(vae, lr_pixels, device, args.seed)
        reference = encode_pixels(vae, hr_pixels, device, args.seed)
        if tuple(draft.shape) != (24, LATENT_FRAMES, 12, 21) or tuple(reference.shape) != (24, LATENT_FRAMES, 24, 42):
            raise ValueError(f'{sample_id}: actual VAE encoded unexpected shapes {draft.shape}, {reference.shape}')
        metadata = {
            'id': sample_id, 'sample_id': sample_id, 'split': split,
            'source_video_path': str(row['source']), 'source_video_sha256': row['source_sha256'],
            'original_video_sha256': row['original_sha256'] or row['source_sha256'],
            'source_frames': SOURCE_FRAMES, 'source_frames_encoded': SOURCE_FRAMES,
            'output_frames': SOURCE_FRAMES, 'decoded_frames': SOURCE_FRAMES,
            'latent_frames': LATENT_FRAMES, 'raw_decoded_frames': RAW_DECODED_FRAMES,
            'vae_internal_pad_frames': VAE_INTERNAL_PAD_FRAMES,
            'vae_padded_input_frames': VAE_PADDED_INPUT_FRAMES,
            'vae_internal_padding': 'Released VAE repeats its last input frame to a multiple of clip_length=17',
            'decode_tail_crop_frames': RAW_DECODED_FRAMES-SOURCE_FRAMES,
            'fps': float(FPS), 'output_duration_seconds': SOURCE_FRAMES/FPS,
            'draft_pixels': [WIDTH//2, HEIGHT//2], 'target_pixels': [WIDTH, HEIGHT],
            'posterior': 'sample', 'posterior_seed': args.seed,
            'normalization': '(raw-vae.latents_mean)/vae.latents_std',
            'pairing': 'Same first120 source-timed RGB frames at24fps; HR cover-area resize/center crop; exact2x2 area average rounded to uint8 for LR',
            'external_time_padding': False, 'external_slowdown': False,
            'source_proof_path': f'source_proof/{sample_id}.json',
            'base_model': str(args.base_model), 'new_training': False,
            'scope': 'Re-evaluate the same three source clips at5seconds with existing upscaler weights; inherited split labels only'}
        target = args.output / 'samples' / f'{sample_id}.pt'
        temporary = target.with_suffix('.pt.tmp')
        torch.save({'draft': draft, 'reference': reference, 'source_pixels': hr_pixels,
                    'metadata': metadata}, temporary)
        temporary.replace(target)
        manifests[split].append({'sample_id': sample_id, 'path': f'samples/{sample_id}.pt', 'split': split})
        summary = {'sample_id': sample_id, 'split': split, 'draft_shape': list(draft.shape),
                   'reference_shape': list(reference.shape), 'source_pixel_shape': list(hr_pixels.shape),
                   'pair_file_sha256': sha256(target), 'metadata': metadata,
                   'seconds': time.perf_counter()-started}
        summaries.append(summary)
        save_json(args.output / 'pair_summary.json', summaries)
        print(json.dumps(summary), flush=True)
        del draft, reference, hr_pixels, lr_pixels
    for split, items in manifests.items():
        (args.output / f'{split}.jsonl').write_text(''.join(json.dumps(item)+'\n' for item in items))


if __name__ == '__main__':
    main()
