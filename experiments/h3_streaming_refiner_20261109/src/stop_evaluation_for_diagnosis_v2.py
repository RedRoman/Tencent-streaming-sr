"""One-shot owned evaluation stop after all three TF2500 outputs are verified.

No training, no waiter, no frozen-source mutation. Preserve the brief step1000
evaluation started by the prior controller's conservative fallback.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def proc(pid):
    p = Path('/proc', str(pid))
    row = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'ppid': int(row[1]), 'pgid': int(row[2]),
            'starttime': int(row[19]), 'cmdline_sha256': sha(p / 'cmdline')}


def main(root):
    root = Path(root)
    out = root / 'control/diagnostic_priority_v2'
    out.mkdir(exist_ok=False)
    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + '\n')
    launch_path = root / 'launches/tf2500_evaluation_v1/launch.json'
    launch = json.loads(launch_path.read_text())
    assert launch['status'] == 'running' and launch['parent_pid'] == 1118689 and launch['child_pid'] == 1118705
    identities = {str(pid): proc(pid) for pid in [1118689, 1118705, 1391764]}
    assert identities['1118689']['starttime'] == 757884804
    assert identities['1118705']['starttime'] == 757884925
    assert identities['1391764']['ppid'] == 1118705 and identities['1391764']['pgid'] == 1118705
    cmd = Path('/proc/1391764/cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert str(root / 'stages/tf2500_evaluation_v1/src/evaluate_streaming.py') in cmd
    assert 'step1000_WingIt_dev_3586_3811_latent' in cmd
    assert sha(root / 'stages/tf2500_evaluation_v1/src/run_locked.py') == '0dd684ffaf2b172426566dcfdee74b204052c6ad3d53622b9dc6fb7aea7bb7c6'
    failure = root / 'control/diagnostic_priority_v1/failure.json'
    assert failure.exists() and not Path('/proc/1390341').exists()
    request = {'authorization': 'User stopped training and prioritized diagnosis; preserve all completed2500 results and defer remaining evaluation.',
               'identities': identities, 'evaluation_launch': str(launch_path),
               'prior_controller_failure_sha256': sha(failure), 'training_updates': 2500,
               'deferred': ['step1000 three dev clips', 'step2000 three dev clips', 'remaining dense two clips'],
               'partial_step1000_preserved': True, 'created_unix': time.time()}
    write('request.json', request)
    verified = []
    for sid in ['WingIt_dev_3586_3811', 'ElephantsDream_dev_13070_13295', 'WingIt_dev_300_525']:
        base = root / 'evaluations/tf2500_v1'
        lp = base / ('step2500_' + sid + '_latent/result.json')
        pp = base / ('step2500_' + sid + '_pixels/result.json')
        lr, pr = [json.loads(p.read_text()) for p in (lp, pp)]
        assert lr['sample_id'] == pr['sample_id'] == sid and lr['training_step'] == 2500
        assert lr['pipeline_shape'] == [1, 24, 67, 68, 120] and lr['chunk_invariance_first4']['max_abs'] == 0
        assert pr['arm'] == 'strict_streaming_step2500' and pr['nativeHR_metrics']['regions']['full226']['frame_count'] == 226
        assert pr['RGB']['shape'] == [226, 1080, 1920, 3]
        assert sha(lr['output']['path']) == lr['output']['sha256'] and sha(pr['RGB']['path']) == pr['RGB']['sha256']
        verified.append({'sample_id': sid, 'latent_receipt_sha256': sha(lp),
                         'pixel_receipt_sha256': sha(pp), 'RGB_sha256': pr['RGB']['sha256']})
    smi = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, check=True, timeout=20).stdout
    (out / 'npu_smi_before.txt').write_text(smi)
    assert re.findall(r'^\|\s*7\s+\d+\s*\|\s*(\d+)\s*\|', smi, re.M) == ['1391764']
    for name in [f'{root.parent}/device_locks/npu7.lock', '/tmp/yunfeng_npu_7.lock', '/tmp/codex-h3-vae-npu-7.lock']:
        st = Path(name).stat()
        key = f'{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}'
        assert any(key in line and 'WRITE 1118689 ' in line for line in Path('/proc/locks').read_text().splitlines())
    members = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            row = proc(int(p.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        if row['pgid'] == 1118705:
            members[row['pid']] = row
    for pid in members:
        cursor = pid
        visited = set()
        while cursor != 1118705:
            assert cursor in members and cursor not in visited
            visited.add(cursor)
            cursor = members[cursor]['ppid']
    write('verified_before_stop.json', {'completed_step2500': verified, 'owned_process_group': members})
    for pid, expected in identities.items():
        assert proc(int(pid)) == expected
    os.kill(1118689, signal.SIGTERM)  # Verified launcher relays to its own child group only.
    deadline = time.monotonic() + 40
    while True:
        launch = json.loads(launch_path.read_text())
        absent = all(not Path('/proc', str(pid)).exists() for pid in [1118689, *members])
        if absent and launch.get('child_reaped'):
            break
        if time.monotonic() > deadline:
            raise TimeoutError('Owned evaluation has not fully exited; no diagnosis may launch')
        time.sleep(.5)
    assert launch['status'] == 'failed' and launch['exit_code'] == -15
    report = {'status': 'evaluation_deferred_for_user_requested_diagnosis',
              'request_sha256': sha(out / 'request.json'), 'controller_source_sha256': sha(__file__),
              'evaluation_launch_sha256': sha(launch_path), 'raw_exit_code': -15,
              'child_reaped': True, 'all_owned_processes_absent': True,
              'completed_step2500': verified, 'deferred': request['deferred'],
              'actual_training_updates': 2500, 'new_optimizer_updates': 0,
              'partial_step1000_preserved': True, 'prior_controller_failure_sha256': sha(failure),
              'original_sources_and_receipts_unchanged': True, 'finished_unix': time.time()}
    write('complete.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
