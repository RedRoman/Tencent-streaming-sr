"""Start this experiment's CPU merge when its verified asset transfer completes."""
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path


root = Path(os.environ['EXP_ROOT'])
lock = (root / '.merge.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
expected = '8e36a0912ab72f7f3b00f41e3c7c97c7dfdec21e66c88ab22f08b9b0c681bb04'
name = 'lora/open_h3_regenerate_2k_lora.safetensors'
status_path = root / 'assets/bridge_import_status.json'
deadline = time.monotonic() + 3600
print(json.dumps({'event': 'waiting_for_verified_lora', 'unix': time.time()}), flush=True)
while time.monotonic() < deadline:
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    asset = status.get('files', {}).get(name, {})
    path = root / 'assets' / name
    if asset.get('status') == 'verified' and asset.get('sha256') == expected and path.exists():
        assert path.stat().st_size == 12452449080
        print(json.dumps({'event': 'verified_lora_ready', 'unix': time.time()}), flush=True)
        subprocess.run(['bash', str(Path(__file__).resolve().parent / 'run_merge.sh')], check=True)
        print(json.dumps({'event': 'merge_completed', 'unix': time.time()}), flush=True)
        break
    time.sleep(15)
else:
    raise TimeoutError('Verified LoRA did not arrive within the CPU merge wait window')
