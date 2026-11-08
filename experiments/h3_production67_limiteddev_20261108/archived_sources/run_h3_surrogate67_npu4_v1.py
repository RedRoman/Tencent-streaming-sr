"""Reviewed-stage launcher candidate for the exact67/226 surrogate diagnostic.

No all-stage mode, retries, training, or checkpoint selection. The CPU command
runs self-test/check/prepare/check with accelerators hidden. Each model stage
requires a separate invocation after its preceding evidence has been inspected.
Only this launcher's start_new_session child process group is ever signalled.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path('/path/to/h3-project')
SCRIPT = Path('/path/to/staging/diagnose_h3_720to1080_v3.py')
SCRIPT_SHA = '1dfc47a50b85fc67119629b9e1be2b3907a890eeaef7ebc97f46010e376ad269'
GEOMETRY = Path('/path/to/staging/h3_720to1080_surrogate_geometry_v3.json')
GEOMETRY_SHA = 'a9d8061c61358c519a47fc88d25334d7de58923c19445ac54a06711c51d6b212'
VAE_SOURCE = Path('/path/to/staging/h3_720_vae_source_72cf83.py')
VAE_SOURCE_SHA = '72cf83c0b8c055b67d7d26abd62769edea752c86adb4a94254a3ed7266feb5d2'
ENV_SCRIPT = ROOT / 'experiment/env.sh'
ENV_SHA = '64451094fa9aca640bc5bfc9ba77080330f858d1706ba74798ca4eb8834e4524'
RUNTIME = Path('/path/to/model-assets/runtime/.venv-npu/bin/python')
OUTPUT = ROOT / 'upscaler_v5_motion_20261106/diagnostics/surrogate67_704to1080_v3'
LAUNCHES = OUTPUT.parent / 'surrogate67_704to1080_v3_launches'
LOCKS = ('/path/to/device-locks/npu4.lock', '/path/to/staging/SHARED_LOCK_OWNER_npu_4.lock', '/path/to/staging/codex-h3-vae-npu-4.lock')
MODEL_STAGES = ('latent', 'pixels')
STAGE_ORDER = ('cpu', 'latent', 'pixels', 'finalize')
MIGRATION = LAUNCHES / 'migration_to_npu4_20261106.json'
PIXEL_ENTRY = Path('/path/to/staging/run_h3_surrogate67_pixels_npu4_v1.py')
PIXEL_ENTRY_SHA = '0f55fdf780243f2deb18fbeb48bb4fb12bb6688f3f0b57231cbca27cf7758496'
MIGRATION_SHA = '435dc3e2459269361db45083e0982bbc2a4bb44a4a0afd39ff90de136dba55b3'
PRIOR_RECEIPT_SHA = {'cpu': 'd01e3d40b1a3b0c230be0b9ced700c8ad6106b68fe63a8fda6b7fb48fcecfef8', 'latent': '62d4bad94d69ba143741720800b5801a0eafc8af35bde04d0034bbe7f557ba7c'}
PRIOR_WRAPPERS = {Path('/path/to/staging/run_h3_surrogate67_npu7_v2.py'): '3da256a995025ac008d0b593f1800cc869652bb3cad8b45a77838f2276ef8311', Path('/path/to/staging/run_h3_surrogate67_npu7_v2.sh'): 'e5bda3a15ece203d8359f17209589d35ee190e6981675fe93d96f6b64fb3cc69'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def record(path):
    p = Path(path)
    require(p.is_file() and not p.is_symlink(), 'Missing/nonregular/symlink input: ' + str(p))
    return dict(path=str(p), sha256=digest(p), bytes=p.stat().st_size)


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp_' + str(os.getpid()))
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def fingerprints():
    fixed = ((PIXEL_ENTRY, PIXEL_ENTRY_SHA), (MIGRATION, MIGRATION_SHA), (SCRIPT, SCRIPT_SHA), (GEOMETRY, GEOMETRY_SHA), (VAE_SOURCE, VAE_SOURCE_SHA), (ENV_SCRIPT, ENV_SHA))
    for path, expected in fixed:
        require(digest(path) == expected, 'Frozen input SHA differs: ' + str(path))
    paths = [x[0] for x in fixed] + [Path(__file__).resolve(), Path(__file__).resolve().with_suffix('.sh')]
    return {str(path): record(path) for path in paths}


class ControlledTermination(RuntimeError):
    def __init__(self, signum):
        self.signum = signum
        super().__init__('Controlled stage termination requested: ' + str(signum))


class Cancellation:
    """Handlers only set a flag: Popen can safely publish ownership first."""
    def __init__(self):
        self.signum = None
        self.previous_handlers = {}
        self.initial_mask = set()

    def handler(self, signum, _frame):
        if self.signum is None:
            self.signum = signum

    def install(self):
        self.previous_handlers = {s: signal.signal(s, self.handler) for s in (signal.SIGTERM, signal.SIGINT)}
        # A parent shell/wrapper may have blocked these. Never propagate that
        # mask across exec; flag-only handlers need no Popen-time blocking.
        self.initial_mask = set(signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT}))
        require(not ({signal.SIGTERM, signal.SIGINT} & set(signal.pthread_sigmask(signal.SIG_BLOCK, set()))),
                'Cancellation signals remain blocked')

    def check(self):
        if self.signum is not None:
            raise ControlledTermination(self.signum)

    def restore_handlers(self):
        for sig, handler in self.previous_handlers.items():
            signal.signal(sig, handler)
        # Intentionally do not restore an inherited blocked TERM/INT mask.


def child_environment(model):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='', ASCEND_VISIBLE_DEVICES='4' if model else '',
               ASCEND_RT_VISIBLE_DEVICES='4' if model else '')
    if model:
        env.pop('TORCH_DEVICE_BACKEND_AUTOLOAD', None)
    else:
        env['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    return env


def acquire_file_lock(path):
    # Existing lock text is retained. No stale-PID heuristic and no unlink.
    handle = Path(path).open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        handle.close()
        raise
    return handle


def require_npu4_idle(smi):
    require('No running processes found in NPU 4' in smi, 'NPU4 has processes or unknown process state')
    # Reject an explicit process row even if another section contains an idle marker.
    for line in smi.splitlines():
        require(not re.match(r'^\|\s*4\s+\d+\s*\|\s*\d+\s*\|', line), 'NPU4 process row contradicts idle marker')
    lines = smi.splitlines()
    entries = [i for i, line in enumerate(lines) if re.match(r'^\|\s*4\s+910B3\s*\|', line)]
    require(len(entries) == 1, 'Cannot uniquely read NPU4 hardware row')
    index = entries[0]
    require(lines[index].split('|')[2].strip() == 'OK', 'NPU4 health is not OK')
    require(index + 1 < len(lines), 'NPU4 chip usage row missing')
    fields = lines[index + 1].split('|')
    require(len(fields) == 5, 'Unexpected NPU4 chip usage format')
    values = [int(x) for x in re.findall(r'\d+', fields[3])]
    require(len(values) == 5 and values[0] == 0 and values[-1] - values[-2] >= 60000,
            'NPU4 must be idle with at least 60000 MiB HBM free')
    return dict(physical_npu=4, health='OK', aicore_percent=values[0],
                hbm_used_mb=values[-2], hbm_total_mb=values[-1], no_running_processes=True)


def live_group_members(pgid):
    """Read OS process state; zombies no longer execute or retain device work."""
    require(pgid > 1 and pgid != os.getpgrp(), 'Refusing own/invalid process group')
    members = []
    proc = Path('/proc')
    if proc.is_dir() and (proc / 'self/stat').exists():
        for path in proc.glob('[0-9]*/stat'):
            try:
                content = path.read_text()
                tail = content[content.rfind(')') + 2:].split()
                if int(tail[2]) == pgid and tail[0] != 'Z':
                    members.append(int(path.parent.name))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
    else:
        result = subprocess.run(['ps', '-axo', 'pid=,pgid=,stat='], capture_output=True, text=True, timeout=5)
        require(result.returncode == 0, 'Cannot verify owned group quiescence')
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) >= 3 and int(fields[1]) == pgid and not fields[2].startswith('Z'):
                members.append(int(fields[0]))
    return members


def terminate_owned(process, term_timeout=15):
    """Terminate this owned session group and reap its leader before unlock."""
    if process is None:
        return dict(pid=None, reaped=True, signals=[], live_group_members_at_end=[])
    pgid = process.pid
    require(pgid > 1 and pgid != os.getpgrp(), 'Owned group cannot be the launcher group')
    leader_identity = 'already_exited'
    if process.poll() is None:
        try:
            require(os.getpgid(pgid) == pgid, 'Owned group identity changed; refusing unrelated signal')
            leader_identity = 'owned_session_leader'
        except ProcessLookupError:
            # The leader may exit/be reaped between poll and getpgid. The
            # start_new_session PGID still belongs to this launch; descendants
            # can remain after its leader disappears. Continue group cleanup.
            leader_identity = 'exited_between_poll_and_getpgid'
    sent = []
    live = live_group_members(pgid)
    if live:
        try:
            os.killpg(pgid, signal.SIGTERM)
            sent.append('SIGTERM')
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + term_timeout
        while live_group_members(pgid) and time.monotonic() < deadline:
            time.sleep(.05)
        if live_group_members(pgid):
            try:
                os.killpg(pgid, signal.SIGKILL)
                sent.append('SIGKILL')
            except ProcessLookupError:
                pass
    # Do not release the device locks until all members are no longer running,
    # including a child left behind after a session leader exits. SIGKILL can
    # remain pending in an uninterruptible kernel operation; waiting is safer
    # than falsely declaring the accelerator free.
    while live_group_members(pgid):
        time.sleep(.1)
    process.wait()
    return dict(pid=pgid, pgid=pgid, signals=sent, reaped=True, returncode=process.returncode,
                live_group_members_at_end=[], leader_identity=leader_identity)


def await_owned_quiescence(process):
    """Never unlock on a cleanup exception while an owned group can run."""
    while True:
        try:
            if not live_group_members(process.pid):
                process.wait()
                # Check again after reaping before allowing active=None.
                if not live_group_members(process.pid):
                    return
        except (OSError, RuntimeError, ValueError):
            # Unknown process state is not permission to release device locks.
            pass
        time.sleep(.1)


class ChildRunner:
    def __init__(self, cancellation):
        self.cancellation = cancellation
        self.active = None
        self.cleanup = None

    def run(self, command, environment, stream, on_started=None, timeout=None):
        self.cancellation.check()
        started = time.monotonic()
        require(self.active is None, 'Only one owned child is permitted')
        try:
            # Handler cannot raise in the assignment interval. If cancellation
            # arrives during Popen, active is set before the next flag check.
            self.active = subprocess.Popen(command, env=environment, stdout=stream,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            if on_started is not None:
                on_started(self.active)
            while True:
                self.cancellation.check()
                require(timeout is None or time.monotonic() - started < timeout, "Owned command exceeded its explicit timeout")
                try:
                    code = self.active.wait(timeout=0.2)
                    self.cancellation.check()
                    return code
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if self.active is not None:
                try:
                    self.cleanup = terminate_owned(self.active)
                except BaseException:
                    # Keep the parent, stage mutex and NPU locks alive. This
                    # barrier does not rerun science or signal any new PGID; it
                    # permits failure propagation only after the owned group
                    # is known quiescent. External termination can resolve a
                    # stuck group, but uncertainty never becomes a free device.
                    print('Owned-child cleanup failed; holding locks until the owned group is quiescent', file=sys.stderr, flush=True)
                    raise
                finally:
                    await_owned_quiescence(self.active)
                    self.active = None


def stage_command(stage):
    command = [str(RUNTIME), str(PIXEL_ENTRY if stage == 'pixels' else SCRIPT), '--root', str(ROOT), '--geometry-spec', str(GEOMETRY),
               '--output', str(OUTPUT), '--stage', stage, '--device', 'npu:0' if stage in MODEL_STAGES else 'cpu']
    if stage == 'self-test':
        command.extend(['--vae-source', str(VAE_SOURCE)])
    return command


def expected_result(stage):
    return ('prepare.json', 'prepared') if stage == 'cpu' else \
        ('latent_summary.json', 'latent_completed') if stage == 'latent' else ('summary.json', 'completed')


def successful_stage(stage, before):
    found = []
    for path in LAUNCHES.glob(stage + '_attempt*.json'):
        value = read(path)
        if value.get('status') == 'completed':
            require(value.get('stage') == stage and value.get('exit_code') == 0 and value.get('sources_unchanged') is True,
                    'Malformed prior stage completion')
            if stage in PRIOR_RECEIPT_SHA:
                require(path.name == stage + '_attempt1.json' and digest(path) == PRIOR_RECEIPT_SHA[stage], 'Prior receipt differs from inspected completion')
                expected = {k: v for k, v in before.items() if k not in {str(PIXEL_ENTRY), str(MIGRATION), str(Path(__file__).resolve()), str(Path(__file__).resolve().with_suffix('.sh'))}}
                for wrapper, expected_sha in PRIOR_WRAPPERS.items():
                    require(digest(wrapper) == expected_sha, 'Preserved original launcher changed')
                    expected[str(wrapper)] = record(wrapper)
                require(value.get('source_fingerprints') == expected, 'Prior scientific source or pinned original launcher differs')
            else:
                require(value.get('source_fingerprints') == before, 'Prior stage used different launcher/scientific source')
            summary = value['stage_summary']
            require(record(summary['path']) == summary, 'Prior stage summary changed')
            found.append(dict(status_file=record(path), result=value))
    require(len(found) <= 1, 'Multiple completions for one stage are forbidden')
    return found[0] if found else None


def verify_predecessors(stage, before):
    require(successful_stage(stage, before) is None, 'Stage already completed; do not rerun')
    predecessors = STAGE_ORDER[:STAGE_ORDER.index(stage)]
    prior = {}
    for name in predecessors:
        found = successful_stage(name, before)
        require(found is not None, 'Complete and inspect preceding stage first: ' + name)
        prior[name] = found['status_file']
    if stage == 'cpu':
        require(not OUTPUT.exists(), 'CPU stage requires fresh output; partial/completed evidence must be retained')
        return prior, None
    contract = read(OUTPUT / 'contract.json')
    contract_sha = digest(OUTPUT / 'contract.json')
    require(contract['files']['script']['sha256'] == SCRIPT_SHA and contract['files']['geometry_spec']['sha256'] == GEOMETRY_SHA,
            'Contract scientific source/geometry differs')
    require(contract['geometry_spec'] == read(GEOMETRY), 'Contract geometry differs')
    prepare = read(OUTPUT / 'prepare.json')
    require(prepare['status'] == 'prepared' and prepare['contract_sha256'] == contract_sha and
            prepare['no_select_device_or_model_loaded'] is True, 'CPU preparation receipt invalid')
    if stage == 'latent':
        require(not (OUTPUT / 'encodings').exists() and not (OUTPUT / 'predictions').exists(),
                'Prior latent partial evidence forbids automatic recovery')
    if stage == 'pixels':
        require(not (OUTPUT / 'pixels').exists(), 'Prior pixels partial evidence forbids automatic recovery')
    return prior, contract_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('pixels', 'finalize'))
    parser.add_argument('--attempt', type=int, default=1,
                        help='Fresh attempt number after independent review; no auto-retry or overwrite.')
    args = parser.parse_args()
    require(args.attempt > 0, 'Positive attempt number required')
    require(Path(sys.executable).resolve() == RUNTIME.resolve(), 'Use the pinned runtime via the reviewed shell launcher')
    cancellation = Cancellation()
    cancellation.install()
    runner = ChildRunner(cancellation)
    mutex = None
    handles = []
    status_path = None
    rec = None
    code = 1
    try:
        cancellation.check()
        before = fingerprints()
        LAUNCHES.mkdir(parents=True, exist_ok=True)
        mutex = acquire_file_lock(LAUNCHES / 'stage.lock')
        cancellation.check()
        prefix = f'{args.stage}_attempt{args.attempt}'
        status_path = LAUNCHES / (prefix + '.json')
        log_path = LAUNCHES / (prefix + '.log')
        require(not status_path.exists() and not log_path.exists(), 'Attempt evidence exists; do not overwrite')
        with status_path.open('x'):
            pass
        rec = dict(schema_version=1, status='starting', stage=args.stage, attempt=args.attempt, pid=os.getpid(),
            started_utc=now(), source_fingerprints=before, output=str(OUTPUT), log=str(log_path), commands=[],
            device_locks=[], physical_npu=None, no_training=True,
            inherited_blocked_signals=sorted(int(x) for x in cancellation.initial_mask),
            child_signal_policy='SIGINT/SIGTERM unblocked before exec; parent handlers set only a cancellation flag')
        atomic_json(status_path, rec)
        predecessors, contract_start = verify_predecessors(args.stage, before)
        rec['preceding_stage_receipts'] = predecessors
        if contract_start:
            rec['contract_sha256_start'] = contract_start
        model = args.stage in MODEL_STAGES
        environment = child_environment(model)
        if model:
            for lock_path in LOCKS:
                handles.append(acquire_file_lock(lock_path))
            rec.update(device_locks=list(LOCKS), physical_npu=4, logical_device='npu:0')
            with (LAUNCHES / (prefix + '_npu_smi_before.txt')).open('x') as smi_log:
                smi_command = ['npu-smi', 'info']
                smi_code = runner.run(smi_command, child_environment(False), smi_log, timeout=30)
            smi_path = Path(smi_log.name)
            require(smi_code == 0, 'npu-smi failed; refusing model launch')
            rec['npu_smi_before'] = record(smi_path)
            rec['device_idle'] = require_npu4_idle(smi_path.read_text())
            cancellation.check()
        stages = ('self-test', 'check', 'prepare', 'check') if args.stage == 'cpu' else (args.stage,)
        rec['status'] = 'running'
        atomic_json(status_path, rec)
        with log_path.open('x') as stream:
            for stage in stages:
                cancellation.check()
                require(fingerprints() == before, 'Pinned source/environment changed before child launch')
                item = dict(stage=stage, command=stage_command(stage), started_utc=now(),
                            accelerator_hidden=not model, physical_npu=4 if model else None)
                rec['commands'].append(item)
                def own(process):
                    item.update(pid=process.pid, pgid=process.pid)
                    atomic_json(status_path, rec)
                try:
                    result = runner.run(item['command'], environment, stream, own)
                finally:
                    item.update(ended_utc=now(), owned_child_cleanup=runner.cleanup)
                    atomic_json(status_path, rec)
                item['exit_code'] = result
                atomic_json(status_path, rec)
                require(result == 0, 'Child stage failed; inspect preserved log: ' + str(log_path))
        cancellation.check()
        require(fingerprints() == before, 'Pinned source/environment changed during stage')
        contract_sha = digest(OUTPUT / 'contract.json')
        if contract_start:
            require(contract_sha == contract_start, 'Scientific contract changed during stage')
        expected_path, expected_status = expected_result(args.stage)
        value = read(OUTPUT / expected_path)
        require(value['status'] == expected_status and value['contract_sha256'] == contract_sha,
                'Stage completion summary invalid')
        rec.update(status='completed', sources_unchanged=True, contract_sha256_end=contract_sha,
                   stage_summary=record(OUTPUT / expected_path))
        code = 0
    except BaseException as exc:
        code = 128 + exc.signum if isinstance(exc, ControlledTermination) else 1
        if rec is not None:
            rec.update(status='failed', controlled_termination=isinstance(exc, ControlledTermination),
                       error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        # ChildRunner never clears active before wait(); never release device
        # or stage locks while a model child is still active.
        require(runner.active is None, 'Owned child has not been reaped; device locks must remain held')
        if rec is not None:
            rec.update(exit_code=code, ended_utc=now(), owned_child_cleanup=runner.cleanup)
            atomic_json(status_path, rec)
        for handle in reversed(handles):
            handle.close()
        if mutex is not None:
            mutex.close()
        cancellation.restore_handlers()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
