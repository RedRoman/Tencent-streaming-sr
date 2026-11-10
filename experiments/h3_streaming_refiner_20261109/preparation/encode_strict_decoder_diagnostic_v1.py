"""CPU-only video packaging of completed decoder ablation; no model execution."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
from fractions import Fraction
import tarfile
import traceback

ROOT = Path('<AUTHORIZED_CACHE>/h3-streaming-refiner-20261109')
OUT = ROOT / 'media/strict_decoder_diagnostic_v1'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD') == '0'
    OUT.mkdir(parents=True, exist_ok=False)
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import imageio_ffmpeg
    result_path = ROOT / 'validation/strict_decoder_pretrained_v1/result.json'
    assert sha(result_path) == '079d0052455865ce5375295188c93221c3fa333c576491e4c9e80610af4fc592'
    result = json.loads(result_path.read_text())
    manifest_path = ROOT / 'runs/dev_v1/manifest.json'
    assert sha(manifest_path) == result['data_manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    rec = next(x for x in manifest['samples'] if x['sample_id'] == result['sample_id'])
    sources = [result['RGB'], rec['rgb']['teacher_student'], rec['rgb']['native_hr']]
    for s in sources:
        assert sha(s['path']) == s['sha256']
    arrays = [np.load(s['path'], mmap_mode='r', allow_pickle=False) for s in sources]
    assert all(a.shape == (226, 1080, 1920, 3) and a.dtype == np.uint8 for a in arrays)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    videos = [('strict_decoder_native1080.mp4', (1920, 1080)),
              ('strict_official_gt_full226.mp4', (1440, 310))]
    workers = []
    for name, (w, h) in videos:
        cmd = [ffmpeg, '-v', 'error', '-nostdin', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{w}x{h}', '-r', '24', '-i', '-', '-an', '-c:v', 'libx264',
               '-threads', '2', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
               '-map_metadata', '-1', '-movflags', '+faststart', str(OUT / name)]
        workers.append(subprocess.Popen(cmd, stdin=subprocess.PIPE))
    font = ImageFont.load_default()
    stills = []
    try:
        for i in range(226):
            workers[0].stdin.write(arrays[0][i].tobytes())
            canvas = Image.new('RGB', (1440, 310), (16, 18, 22))
            draw = ImageDraw.Draw(canvas)
            for j, label in enumerate(['Strict streaming decoder', 'Official overlap decoder', 'Actual native HR']):
                canvas.paste(Image.fromarray(arrays[j][i]).resize((480, 270), Image.Resampling.LANCZOS), (480*j, 40))
                draw.text((480*j+10, 9), label, fill='white', font=font)
            draw.text((1050, 25), f'Decoder ablation - no refiner | frame {i}/225', fill='#cccccc', font=font)
            workers[1].stdin.write(canvas.tobytes())
            if i in [0, 72, 144, 192]:
                stills.append(canvas.copy())
    finally:
        for p in workers:
            p.stdin.close()
        for p in workers:
            assert p.wait() == 0
    sheet = Image.new('RGB', (1440, 1240))
    for i, im in enumerate(stills):
        sheet.paste(im, (0, i*310))
    sheet.save(OUT/'static_frames_0_72_144_192.jpg', quality=94)
    files = {}
    for name, dims in videos:
        p = OUT/name
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-count_frames', '-show_entries', 'stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration',
            '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(p)], text=True))
        s = probe['streams'][0]
        pts = [float(x['best_effort_timestamp_time']) for x in probe['frames']]
        assert int(s['nb_read_frames']) == 226 and Fraction(s['avg_frame_rate']) == 24
        assert (s['width'], s['height']) == dims and len(pts) == 226
        assert max(abs(t-i/24) for i,t in enumerate(pts)) < 1e-5
        subprocess.run([ffmpeg, '-v', 'error', '-xerror', '-i', str(p), '-f', 'null', '-'], check=True)
        files[name] = {'sha256': sha(p), 'bytes': p.stat().st_size, 'probe': s,
                       'all226_pts_checked': True, 'full_decode_passed': True}
    report = {'status': 'CPU_encoding_full226_technical_audit_complete', 'sample_id': result['sample_id'],
              'input_arm': result['input_arm'], 'source_result_sha256': sha(result_path), 'sources': sources,
              'files': files, 'comparison_columns': ['strict_streaming_decoder', 'official_overlap_decoder', 'nativeHR'],
              'refiner_applied': False, 'strict_decoder_accepted': False, 'continuous_visual_reviewed': False,
              'optimizer_updates': 0, 'accelerator_forwards': 0,
              'note': 'Full226 video; metrics remain from 1080p lossless RGB. Contact sheet is static sampling only.'}
    (OUT/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    bundle = ROOT/'evidence/strict_decoder_diagnostic_media_v1.tar'
    assert not bundle.exists()
    with tarfile.open(bundle, 'w') as t:
        for p in sorted(OUT.iterdir()):
            t.add(p, arcname=p.name)
    receipt = {'path': str(bundle), 'bytes': bundle.stat().st_size, 'sha256': sha(bundle)}
    (ROOT/'evidence/strict_decoder_diagnostic_media_bundle_v1.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        if OUT.is_dir():
            (OUT/'failure.json').write_text(json.dumps({'traceback': traceback.format_exc(), 'accelerator_forwards': 0}, indent=2)+'\n')
        raise
