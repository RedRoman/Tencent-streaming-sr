"""One bounded CPU-only controller for an explicitly authorized owned trainer.

Never edits the running worker or its raw receipts. A verified committed checkpoint
is required before stopping; failed checkpoint audit resumes the paused child.
Uses pidfds when available; on the deployed Linux4.19 kernel, revalidates PID,
start time, command, PPID and PGID immediately before each owned-process signal.
"""
import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def process(pid, proc=Path('/proc')):
    folder = proc / str(pid)
    row = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'state': row[0], 'ppid': int(row[1]),
            'pgid': int(row[2]), 'starttime': int(row[19]),
            'cmdline_sha256': sha(folder / 'cmdline')}


def validate_identity(actual, expected):
    for key in ('pid', 'ppid', 'pgid', 'starttime', 'cmdline_sha256'):
        if actual[key] != expected[key]:
            raise ValueError('Owned process identity changed: ' + key)
    if actual['state'] in ('Z', 'X'):
        raise ValueError('Owned process is no longer live')


class OwnedProcess:
    def __init__(self, expected):
        self.expected = expected
        self.fd = None
        if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
            self.fd = os.pidfd_open(expected['pid'])
        validate_identity(process(expected['pid']), expected)

    def send(self, sig):
        validate_identity(process(self.expected['pid']), self.expected)
        if self.fd is not None:
            signal.pidfd_send_signal(self.fd, sig)
        else:
            os.kill(self.expected['pid'], sig)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)


def identities(request):
    result = {}
    for name in ('parent', 'child'):
        expected = request[name]
        result[name] = process(expected['pid'])
        validate_identity(result[name], expected)
    owners = []
    for row in Path('/proc/locks').read_text().splitlines():
        parts = row.split()
        if len(parts) >= 6 and parts[1:4] == ['FLOCK', 'ADVISORY', 'WRITE'] and parts[4] == str(request['parent']['pid']):
            major, minor, inode = parts[5].split(':')
            owners.append((int(major, 16), int(minor, 16), int(inode)))
    for path in request['lock_paths']:
        stat = Path(path).stat()
        if (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino) not in owners:
            raise ValueError('Owned launcher no longer holds all three locks')
    result['three_kernel_locks_verified'] = True
    return result


def verify_sources(request):
    for path, digest in request['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Pinned training/control source changed')
    launch = read(request['training_launch'])
    if any(launch.get(k + '_pid') != request[k]['pid'] for k in ('parent', 'child')):
        raise ValueError('Launch identity changed')
    if launch['status'] != 'running' or launch['physical_npu'] != request['device']:
        raise ValueError('Trainer not running on expected device')


def audit(request, receipt_path):
    from tf_resume_utils import load_cpu, validate_bundle, schedule
    import torch
    target = request['target']
    root = Path(request['training_output'])
    receipt = read(receipt_path)
    if receipt['step'] != target or receipt['private_state_CPU_tensors'] is not True:
        raise ValueError('Checkpoint receipt is not the requested committed checkpoint')
    adapter = root / f'adapter_step{target:04d}.pt'
    private = root / f'private_resume_step{target:04d}.pt'
    for path, key in ((adapter, 'adapter_bytes'), (private, 'private_optimizer_bytes')):
        if path.stat().st_size != receipt[key]:
            raise ValueError('Checkpoint size mismatch')
    a = load_cpu(adapter, receipt['adapter_sha256'])
    b = load_cpu(private, receipt['private_optimizer_sha256'])
    result = validate_bundle(a, b, target)
    if b['parameter_names'] != list(a['adapter']) or b['next_schedule'] != schedule(target) or b['resume_origin_step'] != 200:
        raise ValueError('Parameter order, next schedule or resume origin changed')
    rng = {}
    for key in ('torch_rng_state', 'npu_rng_state'):
        state = b[key]
        if state.device.type != 'cpu' or state.dtype != torch.uint8 or state.ndim != 1 or not state.numel():
            raise ValueError('Missing/invalid saved RNG state')
        rng[key] = {'bytes': state.numel(), 'sha256': hashlib.sha256(state.numpy().tobytes()).hexdigest()}
    generator = torch.Generator(device='cpu')
    generator.set_state(b['torch_rng_state'])
    if not torch.equal(generator.get_state(), b['torch_rng_state']):
        raise ValueError('CPU RNG restore failed')
    if 'torch_npu' in sys.modules:
        raise RuntimeError('CPU audit unexpectedly imported accelerator runtime')
    result.update(status='passed', checkpoint_receipt=receipt,
                  checkpoint_receipt_sha256=sha(receipt_path), next_schedule=b['next_schedule'],
                  RNG=rng, CPU_rng_restore_exact=True, NPU_rng_saved_CPU_only=True,
                  NPU_rng_runtime_restore_tested=False, new_optimizer_updates=0,
                  new_accelerator_forwards=0)
    return result


def completed_updates(request):
    rows = []
    for line in Path(request['training_log']).read_text().splitlines():
        if not line.startswith('{'):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get('event') == 'optimizer_update':
            rows.append(row)
    from tf_resume_utils import schedule
    if not rows or [r['step'] for r in rows] != list(range(201, rows[-1]['step'] + 1)):
        raise ValueError('Optimizer log is not contiguous from restored step200')
    for row in rows:
        spec = schedule(row['step'] - 1)
        if any(row[k] != spec[k] for k in ('latent_index', 'history', 'sigma', 'noise_seed')):
            raise ValueError('Training schedule changed')
        if any(not math.isfinite(row[k]) for k in ('fm', 'draft_l1', 'loss', 'grad_norm')) or row['grad_norm'] <= 0:
            raise ValueError('Invalid recorded training update')
    status = read(Path(request['training_output']) / 'status.json')
    if status['optimizer_updates'] != rows[-1]['step'] or status['new_optimizer_updates'] != rows[-1]['step'] - 200:
        raise ValueError('Raw status and committed optimizer log disagree')
    return rows[-1]['step'], status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--request-sha', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if sha(args.request) != args.request_sha:
        raise ValueError('Stop request changed')
    request = read(args.request)
    if request['target'] != 2500 or request['resume_origin'] != 200:
        raise ValueError('Unexpected stop target')
    verify_sources(request)
    identity = identities(request)
    if args.preflight_only:
        print(json.dumps({'status': 'CPU_preflight_passed', 'identities': identity,
                          'checkpoint_ready': (Path(request['training_output']) / 'checkpoint_step2500.json').exists(),
                          'signals_sent': 0, 'new_optimizer_updates': 0}), flush=True)
        return
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)  # Durable one-shot claim; never reuse an attempt.
    paused = False
    parent_fd = child_fd = None
    started = time.time()
    def interrupted(signum, frame):
        raise RuntimeError('Controller interrupted: ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        parent_fd = OwnedProcess(request['parent'])
        child_fd = OwnedProcess(request['child'])
        identities(request)  # Bind open pidfds to the observed identities.
        write(out / 'status.json', {'status': 'waiting_for_committed_checkpoint',
              'controller_pid': os.getpid(), 'request_sha256': args.request_sha,
              'started_unix': started, 'target': 2500, 'identities': identity,
              'signal_identity_method': 'pidfd' if child_fd.fd is not None else 'immediate_full_proc_identity_revalidation_linux4_19'})
        receipt_path = Path(request['training_output']) / 'checkpoint_step2500.json'
        while not receipt_path.exists():
            if time.time() - started > request['max_wait_seconds']:
                raise TimeoutError('Bounded checkpoint wait expired')
            identities(request)
            time.sleep(.5)
        verify_sources(request)
        smi = subprocess.run(['npu-smi', 'info'], text=True, capture_output=True, check=True, timeout=20).stdout
        (out / 'npu_smi_before_stop.txt').write_text(smi)
        device_pids = re.findall(r'^\|\s*' + str(request['device']) + r'\s+\d+\s*\|\s*(\d+)\s*\|', smi, re.M)
        if device_pids != [str(request['child']['pid'])]:
            raise ValueError('Fresh NPU identity does not match sole owned child')
        identities(request)
        child_fd.send(signal.SIGSTOP)
        paused = True
        deadline = time.time() + 20
        while process(request['child']['pid'])['state'] != 'T':
            if time.time() > deadline:
                raise TimeoutError('Owned child did not pause')
            time.sleep(.1)
        identities(request)
        write(out / 'status.json', {'status': 'owned_child_paused_for_CPU_checkpoint_audit',
              'controller_pid': os.getpid(), 'request_sha256': args.request_sha, 'target': 2500})
        checkpoint_audit = audit(request, receipt_path)
        write(out / 'checkpoint_audit.json', checkpoint_audit)
        actual, raw_status = completed_updates(request)
        if actual < request['target']:
            raise ValueError('Requested training updates not reached')
        identities(request)
        write(out / 'stop_intent.json', {'status': 'checkpoint_verified_before_user_requested_stop',
              'actual_logged_optimizer_updates': actual, 'checkpoint_step': 2500,
              'checkpoint_audit_sha256': sha(out / 'checkpoint_audit.json'),
              'request_sha256': args.request_sha, 'raw_training_status': raw_status})
        parent_fd.send(signal.SIGTERM)  # Pinned launcher relays only to its child PGID.
        child_fd.send(signal.SIGCONT)
        paused = False
        deadline = time.time() + 120
        while True:
            launch = read(request['training_launch'])
            absent = all(not Path('/proc', str(request[k]['pid'])).exists() for k in ('parent', 'child'))
            if launch.get('child_reaped') is True and absent:
                break
            if time.time() > deadline:
                raise TimeoutError('Owned launcher has not reaped/exited; evaluation must remain blocked')
            time.sleep(.5)
        if launch.get('exit_code') != -signal.SIGTERM or launch.get('status') != 'failed':
            raise ValueError('Unexpected original launcher exit; retain evidence for review')
        final_actual, final_status = completed_updates(request)
        if final_actual != actual or final_status != raw_status:
            raise ValueError('Unexpected additional update while stopping')
        smi = subprocess.run(['npu-smi', 'info'], text=True, capture_output=True, check=True, timeout=20).stdout
        (out / 'npu_smi_after_stop.txt').write_text(smi)
        result = {'status': 'user_requested_checkpoint_stop_complete', 'target_optimizer_updates': 2500,
                  'actual_logged_optimizer_updates': actual, 'overshoot_updates': actual - 2500,
                  'checkpoint_step': 2500, 'checkpoint_audit_sha256': sha(out / 'checkpoint_audit.json'),
                  'checkpoint_receipt_sha256': sha(receipt_path), 'request_sha256': args.request_sha,
                  'controller_source_sha256': sha(__file__), 'training_launch_sha256': sha(request['training_launch']),
                  'training_status_sha256': sha(Path(request['training_output']) / 'status.json'),
                  'raw_launcher_status': launch['status'], 'raw_launcher_exit_code': launch['exit_code'],
                  'child_reaped': True, 'parent_and_child_absent': True,
                  'original_raw_receipts_unchanged': True, 'next_action': 'evaluate2500_1000_2000_no_training_resume',
                  'new_optimizer_updates_by_controller': 0, 'finished_unix': time.time()}
        write(out / 'complete.json', result)
        write(out / 'status.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        write(out / 'failure.json', {'error': str(error), 'traceback': traceback.format_exc(),
              'request_sha256': args.request_sha, 'paused_child_will_be_resumed': paused})
        raise
    finally:
        if paused and child_fd is not None:
            try:
                child_fd.send(signal.SIGCONT)
            except ProcessLookupError:
                pass
        for fd in (parent_fd, child_fd):
            if fd is not None:
                fd.close()


if __name__ == '__main__':
    main()
