"""One-shot launch; no waiting and no repeated attempts. Run only after predecessor exit."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    root = Path('<AUTHORIZED_CACHE>/h3-streaming-refiner-20261109')
    config = root / 'config/refiner_long137_launch_after_eval2000_v1.json'
    assert sha(config) == '100441626a0fed66bbb07b4de6226c109ad1281dc75dab68140b03ef7f8be5c7'
    plan = json.loads(config.read_text())
    stage = Path(plan['validation_stage'])
    assert sha(stage / 'manifest.json') == plan['manifest_sha256']
    manifest = json.loads((stage / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha(stage / name) == digest, name
    worker = stage / 'src/validate_refiner_long_stream_v1.py'
    spec = importlib.util.spec_from_file_location('long_validator', worker)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result, _, _ = module.admission(SimpleNamespace(validation_stage=str(stage), manifest_sha=plan['manifest_sha256']))
    assert result['ready'] and not result['torch_imported'] and not result['accelerator_imported']
    for key in ('attempt', 'output', 'log'):
        assert not Path(plan[key]).exists(), 'Never duplicate an attempt or overwrite evidence'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cmd = (proc / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        assert str(worker).encode() not in cmd, 'Validator already present'
    available = next(int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()
                     if s.startswith('MemAvailable:'))
    assert available > plan['minimum_available_GiB'] * 1024 * 1024
    device = plan['device']
    smi = subprocess.check_output(['npu-smi', 'info'], text=True)
    assert f'No running processes found in NPU {device}' in smi
    lock_runner = stage / 'src/run_locked.py'
    assert sha(lock_runner) == '0dd684ffaf2b172426566dcfdee74b204052c6ad3d53622b9dc6fb7aea7bb7c6'
    argv = ['python', str(lock_runner), '--device', str(device), '--attempt', plan['attempt'],
            '--lock-path', f'<AUTHORIZED_CACHE>/device_locks/npu{device}.lock',
            '--lock-path', f'/tmp/yunfeng_npu_{device}.lock',
            '--lock-path', f'/tmp/codex-h3-vae-npu-{device}.lock',
            'python', str(worker), '--validation-stage', str(stage),
            '--manifest-sha', plan['manifest_sha256'], '--output', plan['output']]
    with open(plan['log'], 'x') as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        os.execvpe(argv[0], argv, os.environ)


if __name__ == '__main__':
    main()
