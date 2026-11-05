"""Create audited H264 display copies for the thirteen additional test clips.

Read-only toward raw evaluation evidence; existing target videos are accepted
only through a prior successful per-conversion receipt. No overwrite flag.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def exclusive_json(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--auditor', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    s = read(root / 'summary.json')
    assert s['status'] == 'completed' and s['decode_complete']
    ids = s['new_test_ids']
    assert len(ids) == len(set(ids)) == 13
    assert s['comparison_order'] == ['pretrained','old_causal_ewarp','cmn','cmn_hybrid','conv3d_hybrid','reference']
    jobs = []
    for ident in ids:
        assert Path(ident).name == ident and ident not in ('.','..')
        for label in ('comparison','conv3d_hybrid'):
            source = root / 'decoded' / ident / f'{label}.mp4'
            expected = s['media_audit'][ident][f'{label}.mp4']
            assert sha(source) == expected['sha256']
            assert (expected['width'],expected['height']) == ((4032,412) if label == 'comparison' else (672,384))
            jobs.append((source, source.with_name(f'{label}.h264.mp4'), expected['sha256']))
    audit_path = root / 'h264_decode_audit.json'
    manifest_path = root / 'h264_conversion_manifest.json'
    assert not audit_path.exists() and not manifest_path.exists()
    receipts = root / 'h264_conversion_receipts'
    receipts.mkdir(exist_ok=True)

    def convert(job):
        source, target, digest = job
        receipt = receipts / f'{source.parent.name}_{source.stem}.json'
        if receipt.exists():
            prior = read(receipt)
            assert prior['exit_code'] == 0 and prior['source_sha256'] == sha(source) == digest
            assert prior['output_sha256'] == sha(target)
            return prior
        assert not target.exists(), f'Unreceipted output retained for inspection: {target}'
        started = time.monotonic()
        r = subprocess.run(['/usr/bin/avconvert','--source',str(source),'--output',str(target),
                            '--preset','PresetHighestQuality'],capture_output=True,text=True)
        entry = {'source':str(source.relative_to(root)),'source_sha256':digest,
                 'source_bytes':source.stat().st_size,'output':str(target.relative_to(root)),
                 'preset':'PresetHighestQuality','exit_code':r.returncode,
                 'seconds':time.monotonic()-started,'stdout':r.stdout,'stderr':r.stderr}
        if target.is_file():
            entry.update(output_sha256=sha(target),output_bytes=target.stat().st_size)
        exclusive_json(receipt,entry)
        print(json.dumps({'output':entry['output'],'exit_code':r.returncode,'seconds':entry['seconds']}),flush=True)
        assert r.returncode == 0 and entry.get('output_bytes',0) > 0, receipt
        return entry

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(convert,jobs))
    candidate = root / f'h264_decode_audit_attempt_{time.time_ns()}.json'
    r = subprocess.run([str(args.auditor),'--root',str(root),'--output',str(candidate),
                        '--expect-frames','120','--expect-fps','24','--expect-duration','5',
                        *[str(target) for _,target,_ in jobs]],capture_output=True,text=True)
    assert r.returncode == 0, f'Audit failed; evidence preserved: {candidate}\n{r.stderr}'
    audit = read(candidate)
    assert audit['status'] == 'passed' and audit['video_count'] == len(audit['videos']) == 26
    assert {v['path'] for v in audit['videos']} == {str(t.relative_to(root)) for _,t,_ in jobs}
    for v in audit['videos']:
        comparison = Path(v['path']).name == 'comparison.h264.mp4'
        assert v['passed'] and v['decoded_frames'] == 120 and v['nominal_fps'] == 24
        assert v['asset_duration']['seconds'] == v['track_time_range']['duration']['seconds'] == 5
        assert v['codecs'] == ['avc1'] and v['cadence_errors'] == 0
        assert v['reader_status'] == 'completed' and v['errors'] == []
        assert (v['width'],v['height']) == ((3840,392) if comparison else (672,384))
    with audit_path.open('xb') as f:
        f.write(candidate.read_bytes())
    exclusive_json(manifest_path,{
        'schema_version':1,'status':'completed_with_documented_display_resize',
        'source_evaluation_summary_sha256':sha(root/'summary.json'),
        'script_sha256':sha(__file__),'audit_tool_sha256':sha(args.auditor),
        'decode_audit_sha256':sha(audit_path),'max_parallel_conversions':2,
        'scope':'Thirteen additional test clips, two H264 display copies each. Metrics use original uncompressed RGB.',
        'acceptance':{'video_count':26,'fully_decoded_frames_each':120,'fps':24,'duration_seconds':5,
            'standalone_dimensions':[672,384],'comparison_source_dimensions':[4032,412],
            'comparison_output_dimensions':[3840,392],
            'comparison_resize':'PresetHighestQuality display scaling to3840 width and392 even height; not metric inputs'},
        'records':records})
    print(json.dumps({'status':'completed','videos':26,'root':str(root)}),flush=True)


if __name__ == '__main__':
    main()
