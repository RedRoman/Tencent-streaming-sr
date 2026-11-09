"""CPU-only complete226 comparison of rejected step200, native dense, and GT.

This is an interim diagnostic comparison, never a selected-model deliverable.
The existing no-refiner comparison remains a separate ablation. Metrics use
lossless arrays, not viewing videos; technical audits do not claim motion QA.
"""
import argparse, json, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from encode_preview_cpu import checked, encode, sha


def read_receipt(receipt):
    assert sha(receipt['path']) == receipt['sha256'], 'Receipt payload changed'
    return json.loads(Path(receipt['path']).read_text())


def validate_pair(stream, dense, sample_id, data_sha):
    assert stream['arm'] == 'strict_streaming_step200'
    assert stream['training_step'] == 200
    assert stream['checkpoint_sha256'] == 'b1bd7d862dc98e12bebdd6ea58257fbfd069d8d5173f39bebef7290c2a382ab1'
    assert dense['arm'] == 'released_dense_refiner'
    assert dense['training_step'] is None and dense['checkpoint_sha256'] is None
    for r in (stream, dense):
        assert r['status'] == 'full226_nativeHR_metrics_complete_visual_review_pending'
        assert r['sample_id'] == sample_id and r['data_manifest_sha256'] == data_sha
        assert r['raw_decode_shape'] == [226, 1088, 1920, 3]
        assert r['RGB']['shape'] == [226, 1080, 1920, 3]
        assert r['fps'] == 24 and r['crop_y'] == [4, 1084]
        assert r['decoder'] == 'frozen_official_overlap_provisional_for_quality_comparison'
        assert r['optimizer_updates'] == 0 and not r['strict_streaming_decoder_validated']


def main():
    p = argparse.ArgumentParser()
    for n in ('streaming-preview', 'streaming-preview-sha', 'dense-preview', 'dense-preview-sha',
              'data-manifest', 'data-manifest-sha', 'sample-id', 'output'):
        p.add_argument('--' + n, required=True)
    a = p.parse_args(); started = time.time()
    assert sha(a.data_manifest) == a.data_manifest_sha
    data = json.loads(Path(a.data_manifest).read_text())
    rec = next(r for r in data['samples'] if r['sample_id'] == a.sample_id)
    sr = read_receipt({'path': a.streaming_preview, 'sha256': a.streaming_preview_sha})
    dr = read_receipt({'path': a.dense_preview, 'sha256': a.dense_preview_sha})
    assert sr['status'] == 'full226_engineering_preview_pixels_complete'
    assert dr['status'] == 'full226_dense_reference_complete_visual_review_pending'
    sl = read_receipt(sr['latent']); dl = read_receipt(dr['latent'])
    sp = read_receipt(sr['pixels']); dp = read_receipt(dr['pixels'])
    validate_pair(sp, dp, a.sample_id, a.data_manifest_sha)
    for wrapper, latent, pixel in ((sr, sl, sp), (dr, dl, dp)):
        assert pixel['prediction_sha256'] == wrapper['latent']['sha256']
        assert latent['protocol_sha256'] == data['protocol_sha256']
        assert latent['sample_id'] == a.sample_id
        assert sha(latent['output']['path']) == latent['output']['sha256']
    assert dl['same_draft_noise_condition_sigmas'] and not dl['trained_streaming_adapter_loaded']
    gate_path = Path(dr['latent']['path']).parent / 'upstream_gate.json'
    gate = json.loads(gate_path.read_text())
    assert 0 <= gate['max_abs'] <= .03 and 0 <= gate['relative_rms'] <= .005
    assert gate['scope'] == 'full_model_two_latent_dense_vs_upstream_forward'
    provenance = [sp['RGB'], dp['RGB'], rec['rgb']['native_hr']]
    arrays = [checked(d) for d in provenance]
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    fontpath = next(str(q) for q in (Path('/usr/share/fonts/dejavu/DejaVuSans.ttf'),
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')) if q.is_file())
    labels = ['Streaming refiner | TF200 REJECTED', 'Released bidirectional refiner', 'Actual native-HR GT']

    def panels(panel_width, panel_height, height, font_size):
        font = ImageFont.truetype(fontpath, font_size)
        small = ImageFont.truetype(fontpath, max(12, font_size - 4))
        top = height - panel_height - 36 if height == 620 else 24
        for i in range(226):
            canvas = Image.new('RGB', (panel_width * 3, height), (14, 19, 29))
            draw = ImageDraw.Draw(canvas)
            for j, (frames, label) in enumerate(zip(arrays, labels)):
                draw.text((j * panel_width + 10, 5), label, font=font, fill=(240, 244, 250))
                canvas.paste(Image.fromarray(frames[i]).resize((panel_width, panel_height),
                    Image.Resampling.LANCZOS), (j * panel_width, top))
            footer = (f'{a.sample_id} | {i+1:03d}/226 | 24 fps | '
                      'Interim comparison; no accepted refiner; decoder provisional')
            draw.text((10, top + panel_height + 5), footer, font=small, fill=(195, 205, 220))
            if height == 620 and i in (0, 75, 150, 225):
                canvas.save(out / f'frame_{i:03d}.jpg', quality=94)
            yield canvas

    videos = {
        'comparison': encode(out / 'step200_dense_gt_comparison.mp4', (2880, 620), panels(960, 540, 620, 25)),
        'quickview': encode(out / 'step200_dense_gt_quickview.mp4', (1440, 320), panels(480, 270, 320, 15))
    }
    audit = {'status': 'full226_video_technical_audit_passed', 'columns': labels,
        'sample_id': a.sample_id, 'fps': 24, 'frames': 226, 'input_RGB': provenance,
        'data_manifest_sha256': a.data_manifest_sha, 'protocol_sha256': data['protocol_sha256'],
        'streaming_preview_sha256': a.streaming_preview_sha, 'dense_preview_sha256': a.dense_preview_sha,
        'dense_upstream_gate_sha256': sha(gate_path), 'files': videos,
        'step200_fidelity_passed': False, 'accepted_refiner': None,
        'continuous_visual_reviewed': False, 'strict_streaming_decoder_validated': False,
        'optimizer_updates': 0, 'seconds': time.time() - started}
    (out / 'video_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'status': audit['status'], 'files': videos}), flush=True)


if __name__ == '__main__':
    main()
