"""Prepare pinned public SolarRenderer assets in this independent experiment only."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import subprocess
import tarfile
import time
import urllib.request

ROOT = Path('/opt/refiner-cache/h3-pretrained-causal-refiner-20261109')
CODE_REV = '0ca4f5e45b062cc9e9aee506de46aabdc22e4c3d'
MODEL_REV = 'c5512be825f297ed8db390cbeda0289fb0c24779'
WEIGHTS = {
    'solarrenderer_causal_control_lora.safetensors': (4150865680, '49704c2948a111f958f3a0fc2cb9802081f9b5c129d91f30da6c8758787a8888'),
    'solarrenderer_causal_4step_lora.safetensors': (4150865728, '10d275abf164f2115e682fd62bd1676e9b7db3c2fddaf8a66275651304d19d83'),
}


def record(event, **kw):
    print(json.dumps({'event': event, 'time_unix': time.time(), **kw}), flush=True)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def download(url, path):
    partial = path.with_name(path.name + '.partial')
    result = subprocess.run(['curl', '-fL', '--silent', '--show-error', '--retry', '2',
                             '--connect-timeout', '20', '--max-time', '1800', '-C', '-',
                             '-o', str(partial), url], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'download failed for {path.name}, curl exit {result.returncode}')
    partial.replace(path)


def weight(name, expected):
    size, sha = expected
    p = ROOT / 'weights' / name
    record('weight_start', name=name)
    if not p.exists():
        download(f'https://hf-mirror.com/Enderfga/SolarRenderer/resolve/{MODEL_REV}/{name}', p)
    if p.stat().st_size != size or digest(p) != sha:
        raise RuntimeError(f'integrity failure: {name}')
    record('weight_verified', name=name, bytes=size, sha256=sha)
    return {'file': str(p), 'bytes': size, 'sha256': sha}


def main():
    for name in ['weights', 'logs', 'evidence', 'vendor', 'runs', 'launches', 'data']:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    archive = ROOT / 'vendor' / f'SolarRenderer-{CODE_REV}.tar.gz'
    if not archive.exists():
        download(f'https://codeload.github.com/Enderfga/SolarRenderer/tar.gz/{CODE_REV}', archive)
    dest = ROOT / 'vendor' / f'SolarRenderer-{CODE_REV}'
    if not dest.exists():
        with tarfile.open(archive) as tf:
            for item in tf.getmembers():
                target = (ROOT / 'vendor' / item.name).resolve()
                if not target.is_relative_to((ROOT / 'vendor').resolve()) or item.issym() or item.islnk():
                    raise RuntimeError('unsafe archive member')
            tf.extractall(ROOT / 'vendor', filter='data')
    record('code_ready', path=str(dest), commit=CODE_REV, archive_sha256=digest(archive))
    for p in dest.rglob('AGENTS.md'):
        record('repository_instructions_found', path=str(p))
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fs = [pool.submit(weight, n, e) for n, e in WEIGHTS.items()]
        for f in concurrent.futures.as_completed(fs):
            results.append(f.result())
    manifest = {'status': 'verified', 'code_revision': CODE_REV, 'model_revision': MODEL_REV,
                'source': 'Enderfga/SolarRenderer', 'download_transport': 'hf-mirror.com',
                'weights': results, 'new_optimizer_updates': 0}
    (ROOT / 'evidence' / 'public_assets.json').write_text(json.dumps(manifest, indent=2) + '\n')
    record('assets_complete', weights=len(results))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        record('failed', error=str(exc))
        raise
