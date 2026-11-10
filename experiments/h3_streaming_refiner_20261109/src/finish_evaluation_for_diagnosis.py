"""Finish one already-running pixel worker, then stop only its held dispatch.

CPU-only one-shot controller. All sources, process identities and three locks
are verified. Original evaluation outputs and launcher records are preserved.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', required=True)
    p.add_argument('--request-sha', required=True)
    p.add_argument('--identity-source', required=True)
    a = p.parse_args()
    if sha(a.request) != a.request_sha:
        raise ValueError('Priority request changed')
    if sha(a.identity_source) != '695c8763c644c975a763ed33d9ca94cc77c465f12d373cf5e9aef9fa574d5cf9':
        raise ValueError('Previously tested process identity helper changed')
    spec = importlib.util.spec_from_file_location('identity_helper', a.identity_source)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    request = json.loads(Path(a.request).read_text())
    out = Path(a.request).parent
    with (out / 'controller_claim.json').open('x') as f:
        json.dump({'pid': os.getpid(), 'source_sha256': sha(__file__), 'request_sha256': a.request_sha}, f)
    def write(name, value):
        temp = out / (name + '.tmp')
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(out / name)
    def validate(name, allow_zombie=False):
        expected = request[name]
        current = helper.process(expected['pid'])
        if allow_zombie and current['state'] == 'Z':
            current_for_check = dict(current, state='S')
        else:
            current_for_check = current
        helper.validate_identity(current_for_check, expected)
        return current
    def check_locks():
        rows = Path('/proc/locks').read_text().splitlines()
        for item in request['locks']:
            s = Path(item['path']).stat()
            identity = f'{os.major(s.st_dev):02x}:{os.minor(s.st_dev):02x}:{s.st_ino}'
            if identity != item['identity'] or not any(identity in v and f"WRITE {request['parent']['pid']} " in v for v in rows):
                raise RuntimeError('Parent lock ownership changed')
    terminated = False
    try:
        validate('parent')
        if validate('orchestrator')['state'] != 'T':
            raise RuntimeError('Evaluation dispatch must already be held')
        validate('worker', True)
        check_locks()
        write('status.json', {'status': 'waiting_for_active_full226_pixel_receipt', 'controller_pid': os.getpid(),
                             'request_sha256': a.request_sha, 'new_optimizer_updates': 0})
        deadline = time.monotonic() + request['max_wait_seconds']
        target = Path(request['worker_output']) / 'result.json'
        while True:
            validate('parent')
            if validate('orchestrator')['state'] != 'T':
                raise RuntimeError('Dispatch hold lost')
            state = validate('worker', True)['state']
            check_locks()
            if target.exists() and state == 'Z':
                break
            if state == 'Z':
                raise RuntimeError('Pixel worker exited without a complete result')
            if time.monotonic() > deadline:
                raise TimeoutError('Bounded active worker completion wait expired')
            time.sleep(2)
        verified = []
        for sid in ['WingIt_dev_3586_3811', 'ElephantsDream_dev_13070_13295', 'WingIt_dev_300_525']:
            base = Path(request['evaluation_output'])
            latent = base / ('step2500_' + sid + '_latent/result.json')
            pixel = base / ('step2500_' + sid + '_pixels/result.json')
            lr, pr = [json.loads(v.read_text()) for v in (latent, pixel)]
            if (lr['sample_id'] != sid or lr['training_step'] != 2500
                    or lr['pipeline_shape'] != [1, 24, 67, 68, 120]
                    or lr['chunk_invariance_first4']['max_abs'] != 0
                    or pr['sample_id'] != sid or pr['arm'] != 'strict_streaming_step2500'
                    or pr['nativeHR_metrics']['regions']['full226']['frame_count'] != 226):
                raise ValueError('Incomplete step2500 sample result')
            if sha(lr['output']['path']) != lr['output']['sha256']:
                raise ValueError('Latent payload changed')
            # Pixel receipt binds the lossless RGB payload and full226 metric pass.
            rgb = pr['RGB']
            if rgb['shape'] != [226, 1080, 1920, 3] or sha(rgb['path']) != rgb['sha256']:
                raise ValueError('RGB payload changed')
            verified.append({'sample_id': sid, 'latent_receipt_sha256': sha(latent),
                             'pixel_receipt_sha256': sha(pixel), 'RGB_sha256': rgb['sha256']})
        if verified[-1]['latent_receipt_sha256'] != request['latent_receipt_sha256']:
            raise ValueError('Active worker input changed')
        smi = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, check=True, timeout=20).stdout
        (out / 'npu_smi_before_finish.txt').write_text(smi)
        pids = re.findall(r'^\|\s*7\s+\d+\s*\|\s*(\d+)\s*\|', smi, re.M)
        if any(pid != str(request['worker']['pid']) for pid in pids):
            raise RuntimeError('Unexpected card process before owned shutdown')
        check_locks()
        validate('orchestrator')
        validate('parent')
        os.kill(request['parent']['pid'], signal.SIGTERM)
        terminated = True
        validate('orchestrator')
        os.kill(request['orchestrator']['pid'], signal.SIGCONT)
        deadline = time.monotonic() + 120
        while True:
            launch = json.loads(Path(request['evaluation_launch']).read_text())
            absent = all(not Path('/proc', str(request[k]['pid'])).exists() for k in ['parent', 'orchestrator', 'worker'])
            if absent and launch.get('child_reaped') is True:
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Owned evaluation processes have not exited/reaped')
            time.sleep(.5)
        if launch.get('status') != 'failed' or launch.get('exit_code') != -15:
            raise ValueError('Unexpected original launcher terminal record')
        report = {'status': 'evaluation_deferred_for_user_requested_diagnosis',
                  'request_sha256': a.request_sha, 'controller_source_sha256': sha(__file__),
                  'evaluation_launch_sha256': sha(request['evaluation_launch']),
                  'raw_exit_code': -15, 'child_reaped': True, 'all_owned_processes_absent': True,
                  'completed_step2500': verified, 'deferred': request['deferred'],
                  'actual_training_updates': 2500, 'new_optimizer_updates': 0,
                  'original_sources_and_receipts_unchanged': True, 'finished_unix': time.time()}
        write('complete.json', report)
        write('status.json', report)
        print(json.dumps(report), flush=True)
    except BaseException as error:
        write('failure.json', {'error': str(error), 'traceback': traceback.format_exc(),
                             'termination_sent': terminated, 'new_optimizer_updates': 0})
        # On failed preparation restore only this owned dispatcher; no new job starts here.
        try:
            if validate('orchestrator')['state'] == 'T':
                os.kill(request['orchestrator']['pid'], signal.SIGCONT)
        except (ProcessLookupError, FileNotFoundError):
            pass
        raise


if __name__ == '__main__':
    main()
