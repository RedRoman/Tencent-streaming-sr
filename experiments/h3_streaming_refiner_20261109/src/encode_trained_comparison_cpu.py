"""Package a completed trained first-dev result; no wait, NPU use or acceptance.

Keep the reused dense receipt under its original protocol and prove that the
derived manifest changes only the evaluation-protocol binding. Produce a main
stream/dense/GT viewing copy and a separate no-refiner ablation.
"""
import argparse
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from encode_preview_cpu import checked, encode, sha
from run_evaluation_continuation_v2 import (
    validate_latent, validate_pixels, validate_dense_latent, validate_dense_pixels,
)
from run_tf2500_evaluation import validate_protocol


def pinned(path, digest):
    if sha(path) != digest:
        raise ValueError('Input receipt SHA changed: ' + Path(path).name)
    return json.loads(Path(path).read_text())


def receipt(desc):
    return pinned(desc['path'], desc['sha256'])


def validate_derived(original, derived, protocol_sha):
    expected = dict(original, protocol_sha256=protocol_sha)
    if expected != derived:
        raise ValueError('Evaluation manifest changes more than the protocol binding')


def main():
    parser = argparse.ArgumentParser()
    for name in ('stage', 'inputs', 'inputs-sha', 'original-data', 'derived-data',
                 'stream-latent', 'stream-latent-sha', 'stream-pixels', 'stream-pixels-sha', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--step', type=int, choices=(1000, 2000, 2500), required=True)
    args = parser.parse_args()
    started = time.time()
    inputs = pinned(args.inputs, args.inputs_sha)
    stage = Path(args.stage)
    sources = pinned(stage / 'manifest.json', inputs['manifest_sha256'])
    for name, digest in sources['files'].items():
        if sha(stage / name) != digest:
            raise ValueError('Frozen evaluation source changed')
    policy = pinned(stage / 'config/evaluation_tf2500_v1.json', inputs['protocol_sha256'])
    parent = pinned(stage / 'config/evaluation_parent_v1.json', sources['parent_protocol_sha256'])
    validate_protocol(policy, parent)
    original = pinned(args.original_data, sources['parent_dev_manifest_sha256'])
    derived = pinned(args.derived_data, inputs['data_manifest_sha256'])
    validate_derived(original, derived, inputs['protocol_sha256'])
    sample_id = policy['dev'][0]
    sample = next(r for r in derived['samples'] if r['sample_id'] == sample_id)
    checkpoint_sha = inputs['checkpoint_sha256'][str(args.step)]
    stream_latent = pinned(args.stream_latent, args.stream_latent_sha)
    stream_pixel = pinned(args.stream_pixels, args.stream_pixels_sha)
    validate_latent(stream_latent, args.step, checkpoint_sha, sample_id,
                    inputs['data_manifest_sha256'], inputs['protocol_sha256'])
    validate_pixels(stream_pixel, args.step, checkpoint_sha, sample_id,
                    inputs['data_manifest_sha256'], args.stream_latent_sha)
    dense = inputs['reused_dense']
    dense_latent = receipt(dense['latent'])
    dense_pixel = receipt(dense['pixels'])
    gate = pinned(Path(dense['latent']['path']).parent / 'upstream_gate.json', sources['dense_gate_sha256'])
    validate_dense_latent(dense_latent, sample_id, sources['parent_dev_manifest_sha256'],
                          sources['parent_protocol_sha256'], gate)
    validate_dense_pixels(dense_pixel, sample_id, sources['parent_dev_manifest_sha256'], dense['latent']['sha256'])
    provenance = {'stream': stream_pixel['RGB'], 'dense': dense_pixel['RGB'],
                  'no_refiner': sample['rgb']['teacher_student'], 'GT': sample['rgb']['native_hr']}
    arrays = {key: checked(desc) for key, desc in provenance.items()}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    fontpath = next(str(p) for p in (Path('/usr/share/fonts/dejavu/DejaVuSans.ttf'),
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')) if p.is_file())
    labels = {'stream': f'Streaming refiner | TF{args.step}',
              'dense': 'Released bidirectional refiner', 'no_refiner': 'Frozen upsampler | no refiner',
              'GT': 'Actual native-HR GT'}
    def panels(keys, width, height, total_height, font_size):
        font = ImageFont.truetype(fontpath, font_size)
        small = ImageFont.truetype(fontpath, max(11, font_size - 5))
        top = 44 if total_height == 620 else 26
        for i in range(226):
            canvas = Image.new('RGB', (width * 3, total_height), (14, 19, 29))
            draw = ImageDraw.Draw(canvas)
            for j, key in enumerate(keys):
                draw.text((j * width + 10, 5), labels[key], font=font, fill=(240, 244, 250))
                canvas.paste(Image.fromarray(arrays[key][i]).resize((width, height), Image.Resampling.LANCZOS),
                             (j * width, top))
            draw.text((10, top + height + 5),
                f'{sample_id} | {i+1:03d}/226 | 24 fps | Candidate preview; quality review pending; decoder provisional',
                font=small, fill=(195, 205, 220))
            if total_height == 620 and i in (0, 75, 150, 225):
                canvas.save(out / f'main_frame_{i:03d}.jpg', quality=94)
            yield canvas
    main_keys = ('stream', 'dense', 'GT')
    ablation_keys = ('stream', 'no_refiner', 'GT')
    videos = {
        'native1080': encode(out / f'tf{args.step}_native1080.mp4', (1920, 1080), iter(arrays['stream'])),
        'main': encode(out / f'tf{args.step}_dense_gt.mp4', (2880, 620), panels(main_keys, 960, 540, 620, 25)),
        'quickview': encode(out / f'tf{args.step}_dense_gt_quickview.mp4', (1440, 326), panels(main_keys, 480, 270, 326, 15)),
        'ablation': encode(out / f'tf{args.step}_no_refiner_gt.mp4', (1440, 326), panels(ablation_keys, 480, 270, 326, 15)),
    }
    audit = {'status': 'full226_video_technical_audit_passed', 'training_step': args.step,
             'checkpoint_sha256': checkpoint_sha, 'sample_id': sample_id, 'frames': 226, 'fps': 24,
             'main_columns': [labels[k] for k in main_keys], 'ablation_columns': [labels[k] for k in ablation_keys],
             'files': videos, 'input_RGB': provenance, 'evaluation_inputs_sha256': args.inputs_sha,
             'stream_latent_receipt_sha256': args.stream_latent_sha, 'stream_pixel_receipt_sha256': args.stream_pixels_sha,
             'original_dense_receipts_preserved': True, 'derived_manifest_only_protocol_change': True,
             'accepted_refiner': None, 'continuous_visual_reviewed': False,
             'strict_streaming_decoder_validated': False, 'new_optimizer_updates': 0,
             'new_accelerator_forwards': 0, 'seconds': time.time() - started}
    (out / 'video_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'status': audit['status'], 'files': videos}), flush=True)


if __name__ == '__main__':
    main()
