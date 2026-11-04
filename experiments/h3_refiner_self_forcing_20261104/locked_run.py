"""Cooperate with existing per-device advisory locks, retaining them until exit."""
import argparse
import fcntl
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--devices', default='0,1,2,3,4,5,6,7')
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
lock_dir = Path(os.environ.get("NPU_LOCK_DIR", "/tmp/h3-npu-device-locks"))
lock_dir.mkdir(parents=True, exist_ok=True)
handles = []
for number in args.devices.split(','):
    handle = open(lock_dir/f'npu{number}.lock', 'a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    handles.append(handle)
snapshot = subprocess.check_output(['npu-smi', 'info'], text=True)
busy = [number for number in args.devices.split(',')
        if f'No running processes found in NPU {number}' not in snapshot]
if busy:
    raise RuntimeError(f'Selected physical NPUs are no longer idle: {busy}')
command = args.command[1:] if args.command and args.command[0] == '--' else args.command
environment = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=args.devices)
raise SystemExit(subprocess.call(command, env=environment))
