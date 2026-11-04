#!/usr/bin/env python3
"""Download only an explicitly supplied pinned manifest; dry-run by default.

Standard library only, no Hugging Face hub cache, no model imports. Execute with:
  python3 -B download_h3_manifest.py --project-dir /approved/project --execute
The adjacent manifest is the approved 35-file/82,169,587,116-byte DMD8 set.
Atomic progress/status JSON is written to <project>/download_state/. Payloads use
an adjacent .partial file, range resumption, integrity checks, then atomic rename.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

GIB = 1024 ** 3
BLOCK = 4 * 1024 ** 2
APPROVED_BUDGET = 150 * GIB
META_RESERVE = 8 * 1024 ** 2
STOP = threading.Event()


class DownloadError(Exception):
    pass


class IntegrityError(DownloadError):
    pass


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def verify_file(path, item):
    """Return all measured digests; Git blob SHA1 includes the Git blob header."""
    size = path.stat().st_size
    if size != item['bytes']:
        raise IntegrityError(f"size mismatch: expected {item['bytes']}, observed {size}")
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f'blob {size}\0'.encode()) if item.get('git_blob_sha1') else None
    with path.open('rb') as stream:
        while data := stream.read(BLOCK):
            if STOP.is_set():
                raise DownloadError('interrupted during integrity verification')
            sha256.update(data)
            if blob:
                blob.update(data)
    actual = sha256.hexdigest()
    for field in ('lfs_sha256', 'sha256'):
        if item.get(field) and actual != item[field]:
            raise IntegrityError(f'{field} mismatch')
    result = {'bytes': size, 'sha256': actual}
    if blob:
        if blob.hexdigest() != item['git_blob_sha1']:
            raise IntegrityError('git_blob_sha1 mismatch')
        result['git_blob_sha1'] = blob.hexdigest()
    return result


def read_manifest(path, allow_local_test=False):
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get('schema_version') != 1:
        raise DownloadError('unsupported manifest schema')
    files = manifest.get('files', [])
    if not files:
        raise DownloadError('empty manifest')
    seen = set()
    for item in files:
        name = item['path']
        parts = PurePosixPath(name).parts
        if (not parts or PurePosixPath(name).is_absolute() or '..' in parts
                or '\\' in name or name.endswith('.partial')
                or parts[0] == 'download_state' or name in seen):
            raise DownloadError(f'unsafe or duplicate manifest path: {name}')
        seen.add(name)
        if not isinstance(item['bytes'], int) or item['bytes'] <= 0:
            raise DownloadError(f'invalid byte count: {name}')
        url = urllib.parse.urlparse(item['url'])
        normal = (url.scheme == 'https' and url.hostname in
                  ('huggingface.co', 'raw.githubusercontent.com'))
        local = (allow_local_test and url.scheme == 'http'
                 and url.hostname in ('127.0.0.1', 'localhost', '::1'))
        if not normal and not local:
            raise DownloadError(f'nonofficial URL in manifest: {name}')
        if url.username or url.password:
            raise DownloadError('credentials may not be embedded in a URL')
        for key, length in [('lfs_sha256', 64), ('sha256', 64), ('git_blob_sha1', 40)]:
            if item.get(key) and not re.fullmatch(r'[0-9a-f]{%d}' % length, item[key]):
                raise DownloadError(f'invalid {key} for {name}')
    if manifest.get('total_bytes') != sum(f['bytes'] for f in files):
        raise DownloadError('manifest total does not match file sizes')
    if manifest.get('file_count') != len(files):
        raise DownloadError('manifest file count mismatch')
    budget = manifest.get('project_budget_bytes', APPROVED_BUDGET)
    if not isinstance(budget, int) or not 0 < budget <= APPROVED_BUDGET:
        raise DownloadError('budget exceeds approved 150 GiB')
    return manifest, hashlib.sha256(raw).hexdigest()


def safe_target(root, name):
    path = root / name
    for component in [path, *path.parents]:
        if component == root.parent:
            break
        if component.is_symlink():
            raise DownloadError(f'refusing symlink within download destination: {name}')
    if not path.resolve().is_relative_to(root):
        raise DownloadError(f'path escapes project: {name}')
    return path


def project_size(root):
    if not root.exists():
        return 0
    total = 0
    for directory, subdirs, names in os.walk(root, followlinks=False):
        subdirs[:] = [s for s in subdirs if not (Path(directory) / s).is_symlink()]
        for name in names:
            entry = Path(directory) / name
            if not entry.is_symlink() and entry.is_file():
                total += entry.stat().st_size
    return total


def space_plan(root, manifest):
    used = project_size(root)
    remaining = 0
    for item in manifest['files']:
        final = safe_target(root, item['path'])
        part = safe_target(root, item['path'] + '.partial')
        if final.exists():
            # Integrity verification happens before execution skips this file.
            if not final.is_file() or final.stat().st_size != item['bytes']:
                raise DownloadError(f'existing final file has unexpected size: {item["path"]}')
            if part.exists():
                raise DownloadError(f'both final and partial exist: {item["path"]}')
            continue
        offset = part.stat().st_size if part.exists() else 0
        if offset > item['bytes']:
            raise DownloadError(f'partial larger than manifest: {item["path"]}')
        remaining += item['bytes'] - offset
    disk_root = root
    while not disk_root.exists():
        disk_root = disk_root.parent
    free = shutil.disk_usage(disk_root).free
    expected_peak = used + remaining + META_RESERVE
    budget = manifest.get('project_budget_bytes', APPROVED_BUDGET)
    return {'existing_project_bytes': used, 'remaining_transfer_bytes': remaining,
            'expected_project_peak_bytes': expected_peak, 'project_budget_bytes': budget,
            'filesystem_free_bytes': free, 'required_new_disk_bytes': remaining + META_RESERVE,
            'within_budget': expected_peak <= budget,
            'enough_disk_space': free >= remaining + META_RESERVE}


class State:
    def __init__(self, root, manifest, digest, plan, interval):
        self.root, self.manifest, self.interval = root, manifest, interval
        self.lock = threading.RLock()
        self.started = time.monotonic()
        self.last_write = 0.0
        self.total_downloaded = 0
        self.path = root / 'download_state'
        self.records = {f['path']: {'path': f['path'], 'expected_bytes': f['bytes'],
                                  'downloaded_bytes': 0, 'status': 'pending'}
                        for f in manifest['files']}
        for item in manifest['files']:
            final = safe_target(root, item['path'])
            part = safe_target(root, item['path'] + '.partial')
            existing = final if final.exists() else part
            if existing.exists():
                self.records[item['path']]['downloaded_bytes'] = existing.stat().st_size
        self.status = {'schema_version': 1, 'status': 'running', 'started_at_utc': utc(),
                       'updated_at_utc': utc(), 'manifest_sha256': digest,
                       'repo_id': manifest.get('repo_id'), 'revision': manifest.get('revision'),
                       'total_bytes': manifest['total_bytes'], 'file_count': len(self.records),
                       'space_plan': plan}

    def write(self, force=False):
        with self.lock:
            now = time.monotonic()
            if not force and now - self.last_write < self.interval:
                return
            self.last_write = now
            entries = list(self.records.values())
            completed = [f for f in entries if f['status'] in ('complete', 'verified_existing')]
            self.status.update(updated_at_utc=utc(), elapsed_seconds=now - self.started,
                               transferred_this_run_bytes=self.total_downloaded,
                               completed_files=len(completed),
                               verified_bytes=sum(f['expected_bytes'] for f in completed),
                               on_disk_payload_bytes=sum(f['downloaded_bytes'] for f in entries),
                               files=entries)
            self.path.mkdir(parents=True, exist_ok=True)
            for name, content in [('progress.json', self.status),
                                  ('status.json', {k:v for k,v in self.status.items() if k != 'files'})]:
                target = self.path / name
                temp = self.path / (name + '.tmp')
                temp.write_text(json.dumps(content, indent=2) + '\n')
                os.replace(temp, target)

    def update(self, name, force=False, transferred=0, **fields):
        with self.lock:
            self.records[name].update(fields)
            self.total_downloaded += transferred
            self.write(force)


def transfer(item, final, part, state, timeout):
    offset = part.stat().st_size if part.exists() else 0
    if offset == item['bytes']:
        return
    headers = {'User-Agent': 'Sol-H3-Pinned-Manifest-Downloader/1.0',
               'Accept-Encoding': 'identity'}
    if offset:
        headers['Range'] = f'bytes={offset}-'
    request = urllib.request.Request(item['url'], headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = response.getcode()
        if status not in (200, 206):
            raise DownloadError(f'unexpected HTTP status {status}')
        encoding = response.headers.get('Content-Encoding', 'identity')
        if encoding.lower() != 'identity':
            raise DownloadError('encoded payload cannot be range-verified')
        if status == 206:
            value = response.headers.get('Content-Range', '')
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', value)
            if not match or int(match[1]) != offset or int(match[3]) != item['bytes']:
                raise DownloadError('invalid Content-Range start or total')
            if int(match[2]) != item['bytes'] - 1:
                raise DownloadError('server returned an incomplete requested range')
        elif offset:
            # Only our own .partial is reset when the server ignores Range.
            offset = 0
            state.update(item['path'], force=True, resume_ignored=True)
        length = response.headers.get('Content-Length')
        if length is not None and int(length) != item['bytes'] - offset:
            raise IntegrityError('Content-Length differs from pinned size')
        final.parent.mkdir(parents=True, exist_ok=True)
        state.update(item['path'], force=True, status='downloading', downloaded_bytes=offset)
        with part.open('ab' if offset else 'wb') as output:
            while True:
                if STOP.is_set():
                    raise DownloadError('interrupted; partial retained for resumption')
                data = response.read(min(BLOCK, item['bytes'] - offset + 1))
                if not data:
                    break
                if offset + len(data) > item['bytes']:
                    raise IntegrityError('payload exceeds pinned size')
                output.write(data)
                offset += len(data)
                state.update(item['path'], transferred=len(data), downloaded_bytes=offset)
            output.flush()
            os.fsync(output.fileno())
        if offset != item['bytes']:
            raise DownloadError(f'transfer ended early at {offset}/{item["bytes"]} bytes')


def download_one(item, root, state, attempts, timeout):
    name = item['path']
    final, part = safe_target(root, name), safe_target(root, name + '.partial')
    if final.exists():
        try:
            state.update(name, force=True, status='verifying_existing', downloaded_bytes=final.stat().st_size)
            measured = verify_file(final, item)
            state.update(name, force=True, status='verified_existing', integrity=measured)
            return True
        except (OSError, DownloadError) as error:
            state.update(name, force=True, status='failed', error=str(error))
            return False  # Never overwrite or delete an existing final file.
    for attempt in range(1, attempts + 1):
        if STOP.is_set():
            break
        state.update(name, force=True, attempt=attempt)
        try:
            transfer(item, final, part, state, timeout)
            state.update(name, force=True, status='verifying', downloaded_bytes=part.stat().st_size)
            measured = verify_file(part, item)
            os.replace(part, final)
            state.update(name, force=True, status='complete', integrity=measured, error=None)
            return True
        except IntegrityError as error:
            state.update(name, force=True, status='retrying', error=str(error))
            if attempt < attempts and part.exists():
                # A failed digest cannot be repaired by appending; truncate only
                # this downloader's partial before its bounded retry.
                with part.open('wb'):
                    pass
                state.update(name, force=True, downloaded_bytes=0, integrity_reset=True)
        except (OSError, DownloadError, urllib.error.URLError,
                http.client.HTTPException, ValueError) as error:
            state.update(name, force=True, status='retrying', error=str(error))
        if attempt < attempts:
            STOP.wait(min(2 ** (attempt - 1), 15))
    state.update(name, force=True, status='interrupted' if STOP.is_set() else 'failed')
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('h3_download_manifest_20261008.json'))
    parser.add_argument('--execute', action='store_true', help='download; default only prints the plan and does not write')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=2)
    parser.add_argument('--attempts', type=int, choices=range(1, 6), default=3)
    parser.add_argument('--request-timeout', type=float, default=30)
    parser.add_argument('--progress-interval', type=float, default=5)
    parser.add_argument('--allow-local-test-http', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.project_dir.is_absolute():
        parser.error('--project-dir must be an explicit absolute path')
    if args.project_dir.is_symlink():
        parser.error('--project-dir cannot be a symlink')
    if args.request_timeout <= 0 or args.progress_interval <= 0:
        parser.error('timeouts and interval must be positive')
    root = args.project_dir.resolve()
    try:
        manifest, digest = read_manifest(args.manifest, args.allow_local_test_http)
        plan = space_plan(root, manifest)
        output = {'mode': 'execute' if args.execute else 'dry_run',
                  'manifest_sha256': digest, 'file_count': manifest['file_count'],
                  'total_bytes': manifest['total_bytes'], 'workers': args.workers,
                  'space_plan': plan}
        print(json.dumps(output), flush=True)
        if not plan['within_budget'] or not plan['enough_disk_space']:
            raise DownloadError('insufficient project budget or filesystem space')
        if not args.execute:
            return 0
        root.mkdir(parents=True, exist_ok=True)
        state_root = safe_target(root, 'download_state')
        state_root.mkdir(parents=True, exist_ok=True)
        lock_path = state_root / 'download.lock'
        if lock_path.is_symlink():
            raise DownloadError('lock file is a symlink')
        with lock_path.open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise DownloadError('another manifest download is active in this project')
            # Re-check after acquiring lock so concurrent invocations cannot
            # reserve the same remaining space or overwrite the same partials.
            plan = space_plan(root, manifest)
            if not plan['within_budget'] or not plan['enough_disk_space']:
                raise DownloadError('space budget changed before execution')
            state = State(root, manifest, digest, plan, args.progress_interval)
            (state_root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            state.write(True)
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, lambda *_: STOP.set())
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(download_one, item, root, state,
                                       args.attempts, args.request_timeout)
                           for item in manifest['files']]
                success = all([future.result() for future in futures])
            state.status['status'] = 'complete' if success else ('interrupted' if STOP.is_set() else 'failed')
            state.status['ended_at_utc'] = utc()
            state.write(True)
            print(json.dumps({k:v for k,v in state.status.items() if k != 'files'}), flush=True)
            return 0 if success else (130 if STOP.is_set() else 1)
    except (DownloadError, OSError, ValueError, KeyError) as error:
        print(json.dumps({'status': 'failed', 'error': str(error)}), file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    sys.exit(main())
