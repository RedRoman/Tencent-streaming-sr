"""Audited compact viewing copy of an already complete full226 comparison."""
import argparse, json, subprocess
from fractions import Fraction
from pathlib import Path
import imageio_ffmpeg
from encode_preview_cpu import sha


def probe(path):
    p = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_frames', '-show_entries', 'stream=width,height,avg_frame_rate,duration,nb_read_frames',
        '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(path)]))
    s = p['streams'][0]
    assert int(s['nb_read_frames']) == 226 and Fraction(s['avg_frame_rate']) == 24
    assert abs(float(s['duration']) - 226/24) < 1e-5
    pts = [float(f['best_effort_timestamp_time']) for f in p['frames']]
    assert len(pts) == 226 and all(abs(t-i/24) < 1e-5 for i, t in enumerate(pts))
    return s


def main():
    p = argparse.ArgumentParser()
    for name in ('input', 'input-sha', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args(); source = Path(a.input); out = Path(a.output)
    assert sha(source) == a.input_sha and not out.exists()
    before = probe(source)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, '-hide_banner', '-loglevel', 'error', '-n', '-i', str(source), '-an',
        '-map_metadata', '-1', '-c:v', 'libx264', '-preset', 'medium', '-crf', '25',
        '-maxrate', '1800k', '-bufsize', '3600k', '-threads', '4', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(out)], check=True)
    after = probe(out)
    assert (before['width'], before['height']) == (after['width'], after['height'])
    subprocess.run([ff, '-v', 'error', '-xerror', '-i', str(out), '-f', 'null', '-'], check=True)
    report = {'status': 'full226_compact_copy_technical_audit_passed',
        'source_sha256': a.input_sha, 'sha256': sha(out), 'bytes': out.stat().st_size,
        'stream': after, 'all226_pts_checked': True, 'full_decode_checked': True,
        'continuous_visual_reviewed': False, 'quality_acceptance': False, 'optimizer_updates': 0}
    out.with_suffix('.audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
