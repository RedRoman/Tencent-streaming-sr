#!/usr/bin/env python3
"""Read-only35-file download receipt: cached receiver digests plus current stat.

Default reads no payload bytes. --rehash-small additionally hashes only files
<=16MiB (metadata/licenses/official prompt), never scans the82GB weight payload.
Writes nothing and acquires no download lock. stdout is the receipt; the caller
may save it. This does not query/stop/restart any active download or NPU process.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys

APPROVED_MANIFEST_SHA256 = 'f476c5cbbddd15c07a04fcc1ea449f9d7f8b3c6bee300bd0c913f4bd826a42ba'
APPROVED_REPO = 'OpenVDN/vdn-minimax-h3'
APPROVED_REVISION = '2ce18deb1d1c6b115d6bb3732f3750fdbac95173'
APPROVED_COUNT = 35
APPROVED_BYTES = 82169587116
SMALL_LIMIT = 16 * 1024 ** 2
BLOCK = 4 * 1024 ** 2


def safe_path(root, name):
    path = PurePosixPath(name)
    if (not path.parts or path.is_absolute() or '..' in path.parts or '\\' in name
            or path.parts[0] not in ('ckpts', 'prompts')):
        raise ValueError('unsafe or unexpected manifest path')
    target = root / name
    for component in [target, *target.parents]:
        if component == root.parent:
            break
        if component.is_symlink():
            raise ValueError('symlink in approved payload path')
    if not target.resolve().is_relative_to(root):
        raise ValueError('payload outside project')
    return target


def snapshot(path):
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError('payload is not a regular file')
    return {'device': s.st_dev, 'inode': s.st_ino, 'bytes': s.st_size,
            'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns, 'mode': s.st_mode}


def timestamp_ns(value):
    if not isinstance(value, str):
        return None
    try:
        date = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
        if date.tzinfo is None:
            return None
        return int(date.timestamp() * 10 ** 9)
    except ValueError:
        return None


def hash_payload(path, item):
    sha = hashlib.sha256()
    blob = hashlib.sha1(f'blob {item["bytes"]}\0'.encode()) if item.get('git_blob_sha1') else None
    read_bytes = 0
    with path.open('rb') as stream:
        while block := stream.read(BLOCK):
            sha.update(block)
            if blob:
                blob.update(block)
            read_bytes += len(block)
    result = {'bytes': read_bytes, 'sha256': sha.hexdigest()}
    if blob:
        result['git_blob_sha1'] = blob.hexdigest()
    return result


def record_entries(path):
    raw = path.read_bytes()
    doc = json.loads(raw)
    entries = doc.get('files', {})
    if isinstance(entries, dict):
        pairs = list(entries.items())
    elif isinstance(entries, list):
        pairs = [(item['path'], item) for item in entries]
    else:
        raise ValueError('unsupported progress files representation')
    if len({name for name, _ in pairs}) != len(pairs):
        raise ValueError('duplicate progress file records')
    result = []
    for name, entry in pairs:
        if entry.get('path') != name:
            raise ValueError('progress map key/path disagreement')
        result.append((name, entry, entry.get('updated_at_utc') or doc.get('updated_at_utc')))
    return result, {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                    'bytes': len(raw), 'reported_status': doc.get('status'),
                    'reported_completed_files': doc.get('completed_files')}


def cached_digest_valid(item, record):
    if record.get('status') not in ('complete', 'verified_existing'):
        return False, 'record_not_complete'
    actual = record.get('actual_bytes', record.get('downloaded_bytes'))
    integrity = record.get('integrity') or {}
    if (record.get('expected_bytes') != item['bytes'] or actual != item['bytes']
            or integrity.get('bytes') != item['bytes']):
        return False, 'cached_byte_counts_mismatch'
    measured = integrity.get('sha256')
    if not isinstance(measured, str) or not re.fullmatch('[0-9a-f]{64}', measured):
        return False, 'cached_SHA256_missing_or_invalid'
    for key in ('lfs_sha256', 'sha256'):
        if item.get(key) and measured != item[key]:
            return False, key + '_cached_mismatch'
    if item.get('git_blob_sha1') and integrity.get('git_blob_sha1') != item['git_blob_sha1']:
        return False, 'git_blob_sha1_cached_mismatch'
    return True, None


def inspect_completed(root, manifest, record_paths, *, rehash_small=False, approved=True):
    root = Path(root)
    if not root.is_absolute() or root.is_symlink():
        raise ValueError('explicit absolute nonsymlink project required')
    root = root.resolve()
    items = manifest['files']
    if (manifest.get('file_count') != len(items) or manifest.get('total_bytes') != sum(i['bytes'] for i in items)
            or len({i['path'] for i in items}) != len(items)):
        raise ValueError('invalid manifest count/size/duplicates')
    if approved and (manifest['file_count'] != APPROVED_COUNT or manifest['total_bytes'] != APPROVED_BYTES
                     or manifest.get('repo_id') != APPROVED_REPO or manifest.get('revision') != APPROVED_REVISION):
        raise ValueError('not the approved fixed35-file manifest')
    records, evidence, errors = {}, [], []
    for path in map(Path, record_paths):
        if not path.exists():
            continue
        if path.is_symlink():
            raise ValueError('progress evidence is a symlink')
        entries, source = record_entries(path)
        evidence.append(source)
        for name, entry, verified_at in entries:
            records.setdefault(name, []).append((entry, verified_at, str(path)))
    if not evidence:
        errors.append('no_cached_receiver_integrity_records')
    receipt_files, initial_stats, payload_read = [], {}, 0
    for item in items:
        name = item['path']
        row = {'path': name, 'expected_bytes': item['bytes'], 'accepted': False,
               'fresh_payload_hash_this_run': False, 'payload_bytes_read_this_run': 0}
        row_errors = []
        try:
            target = safe_path(root, name)
            partial = safe_path(root, name + '.partial')
            if partial.exists() or partial.is_symlink():
                row_errors.append('partial_still_present')
            current = snapshot(target)
            initial_stats[name] = current
            row['current_stat'] = current
            if current['bytes'] != item['bytes']:
                row_errors.append('current_size_mismatch')
            completed_records = [(r, stamp, source) for r, stamp, source in records.get(name, [])
                                 if r.get('status') in ('complete', 'verified_existing')]
            valid = []
            for cached, stamp, source in completed_records:
                okay, error = cached_digest_valid(item, cached)
                if okay:
                    valid.append((cached, stamp, source))
                else:
                    row_errors.append(error)
            if not valid:
                row_errors.append('no_valid_complete_cached_record')
            elif len({record['integrity']['sha256'] for record, _, _ in valid}) != 1:
                row_errors.append('complete_cached_records_disagree_on_content')
            else:
                # Latest available metadata timestamp; old failed records do not
                # disqualify a later successful checksum-verified receiver record.
                cached, stamp, source = max(valid, key=lambda v: timestamp_ns(v[1]) or -1)
                row.update(cached_integrity=dict(cached['integrity']), cached_record_source=source,
                           cached_record_timestamp=stamp, cached_digest_verified_in_this_run=False)
                fresh = None
                if rehash_small and item['bytes'] <= SMALL_LIMIT:
                    fresh = hash_payload(target, item)
                    payload_read += fresh['bytes']
                    row.update(fresh_payload_hash_this_run=True, payload_bytes_read_this_run=fresh['bytes'],
                               fresh_integrity=fresh)
                    for key in ('bytes', 'sha256', 'git_blob_sha1'):
                        if key in cached['integrity'] and fresh.get(key) != cached['integrity'][key]:
                            row_errors.append('fresh_' + key + '_differs_from_cached_verification')
                stamp_ns = timestamp_ns(stamp)
                changed = stamp_ns is not None and current['mtime_ns'] > stamp_ns + 1000000
                row['modified_after_cached_record'] = changed
                if changed and fresh is None:
                    row_errors.append('payload_mtime_newer_than_cached_verification_requires_fresh_hash')
                row['official_content_digest_available'] = bool(item.get('lfs_sha256') or item.get('sha256') or item.get('git_blob_sha1'))
                row['origin_digest_kind'] = ('LFS_SHA256' if item.get('lfs_sha256') else
                    'official_manifest_SHA256' if item.get('sha256') else
                    'Git_blob_SHA1_header_plus_content' if item.get('git_blob_sha1') else
                    'pinned_URL_and_size_only_no_official_content_digest')
        except (OSError, ValueError) as error:
            row_errors.append(type(error).__name__ + ': ' + str(error))
        row['errors'] = row_errors
        row['accepted'] = not row_errors
        receipt_files.append(row)
    stable = True
    for row in receipt_files:
        name = row['path']
        if name not in initial_stats:
            continue
        try:
            if snapshot(safe_path(root, name)) != initial_stats[name]:
                row['errors'].append('payload_stat_changed_during_receipt')
                row['accepted'] = False
                stable = False
        except (OSError, ValueError):
            row['errors'].append('payload_disappeared_during_receipt')
            row['accepted'] = False
            stable = False
    for source in evidence:
        if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() != source['sha256']:
            stable = False
            errors.append('progress_evidence_changed_during_receipt')
    accepted = sum(row['accepted'] for row in receipt_files)
    actual_bytes = sum(row.get('current_stat', {}).get('bytes', 0) for row in receipt_files)
    passed = not errors and stable and accepted == manifest['file_count'] and actual_bytes == manifest['total_bytes']
    report = {'schema_version': 1, 'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status': 'complete_with_cached_receiver_integrity' if passed else 'incomplete_or_evidence_mismatch',
              'passed': passed, 'approved_fixed_manifest': approved, 'repo_id': manifest.get('repo_id'),
              'revision': manifest.get('revision'), 'expected_file_count': manifest['file_count'],
              'accepted_file_count': accepted, 'expected_payload_bytes': manifest['total_bytes'],
              'current_payload_bytes_sum': actual_bytes, 'payload_bytes_read_this_run': payload_read,
              'fresh_rehashed_file_count': sum(r['fresh_payload_hash_this_run'] for r in receipt_files),
              'cache_only_file_count': sum(r['accepted'] and not r['fresh_payload_hash_this_run'] for r in receipt_files),
              'metadata_stat_snapshot_stable': stable, 'record_sources': evidence,
              'errors': errors, 'files': receipt_files,
              'actions': {'download_started': False, 'files_modified': False, 'download_lock_acquired': False,
                          'processes_queried_or_controlled': False, 'NPU_initialized': False},
              'limits': ['Existing receiver/download logs provide cached digest evidence; default does not freshly hash current payload bytes.',
                         'Existing records contain no file inode/mtime fingerprint from checksum time; current stat and timestamp detect obvious changes only.',
                         'Current receipt does not prove writer process exit. Parent must separately confirm unique downloader completion.',
                         'Files without an official content digest retain pinned-URL/size provenance; cached SHA256 records the downloaded content.',
                         'This checks exactly the35 approved paths, not absence of other project files or weight usability in the model.']}
    report['receipt_sha256'] = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-dir', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--records', type=Path, nargs='*', help='defaults to project/download_state/{bridge_progress,progress}.json')
    p.add_argument('--rehash-small', action='store_true')
    args = p.parse_args()
    raw = args.manifest.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != APPROVED_MANIFEST_SHA256:
        raise ValueError('manifest differs from the approved fixed35-file manifest SHA256')
    manifest = json.loads(raw)
    paths = args.records if args.records is not None else [
        args.project_dir / 'download_state/bridge_progress.json',
        args.project_dir / 'download_state/progress.json']
    report = inspect_completed(args.project_dir, manifest, paths, rehash_small=args.rehash_small)
    report['manifest_sha256'] = digest
    report.pop('receipt_sha256')
    report['receipt_sha256'] = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({'status': 'failed', 'error': str(error)}), file=sys.stderr)
        sys.exit(2)
