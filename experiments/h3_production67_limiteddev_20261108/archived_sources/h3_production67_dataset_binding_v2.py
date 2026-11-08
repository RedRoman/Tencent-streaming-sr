"""Immutable video/PNG multi-pair envelope over frozen production67 reader v2.

Strict science equality remains unchanged, including generation precision and
posterior/teacher mode hashes. This is not a drop-in replacement for frozen
paired2000 core or sparse evaluator, which explicitly import binder v1.

This file neither generates data nor changes any child training_ready flag.
Its audit certifies bytes/reader compatibility, not native-source approval,
formal training admission, or test-model access.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from urllib.parse import urlsplit

SCHEMA = 'h3-production67-multipair-registry-v2'
AUDIT_SCHEMA = 'h3-production67-multipair-payload-audit-v2'
LOADER_SHA = '1b7b1f41996b5cc00fe45e33a35261e5d159ce03b96fb43f1f3b0ab6be00d47d'
SOURCE_SPECIFIC = {'source_split_contract_sha256', 'selected_windows_sha256',
                   'native_rgb_reader_sha256'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def strict_loader():
    path = Path(__file__).with_name('h3_production67_pair_loader_v2.py')
    require(file_sha(path) == LOADER_SHA, 'Frozen strict loader changed')
    spec = importlib.util.spec_from_file_location('h3_multi_child_loader_v2', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def directory(root, relative):
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir(),
            'Real absolute registry root required')
    require(type(relative) is str and relative and not Path(relative).is_absolute()
            and all(p not in ('', '.', '..') for p in relative.split('/')),
            'Unsafe relative cache directory')
    path = root / relative
    require(path.resolve() == path and path.is_dir() and path.is_relative_to(root),
            'Missing, escaping or symlink cache directory')
    return path


def checked_json(root, desc):
    loader = strict_loader()
    require(type(desc) is dict and set(desc) == {'path', 'bytes', 'sha256'},
            'Exact JSON descriptor required')
    path = loader.safe_member(Path(root), desc['path'])
    require(type(desc['bytes']) is int and path.stat().st_size == desc['bytes'],
            'JSON byte count changed')
    return loader.checked_json(Path(root), desc['path'], desc['sha256'])


def common_science(contract):
    """Keep numeric recipe identical while allowing distinct child provenance."""
    loader = strict_loader()
    loader.validate_contract(contract)
    value = {k: copy.deepcopy(v) for k, v in contract.items()
             if k not in ('generation_binding', 'generation_binding_sha256')}
    value['numeric_generation_binding'] = {
        k: contract['generation_binding'][k]
        for k in loader.BINDING_HASHES if k not in SOURCE_SPECIFIC}
    return value


def source_identity(pair):
    """Whole-source collision keys, never a window manifest as a film identity.

    PNG index SHA and publisher sequence directory are independent collision
    keys, so changing one alias cannot bypass a same-source overlap/split check.
    Film-family splits still come from the externally frozen registry mapping;
    this helper cannot infer equivalence of unrelated video/PNG representations.
    """
    loader = strict_loader()
    kind = pair.get('source_kind', 'video_source')
    require(kind in ('video_source', 'native_png_sequence'), 'Unknown source kind')
    ids = pair['source_time']['source_frame_indices']
    loader.validate_time(pair['source_time'])
    if kind == 'video_source':
        require('native_png_source' not in pair, 'Video cannot carry ambiguous PNG identity')
        keys = [['video_source_sha256', loader.sha_value(pair['source_video_sha256'])]]
        basis = 'video_source_frame_index_zero_based'
    else:
        provenance = loader.validate_png_source_identity(pair)
        bases = set()
        for row in provenance['frames']:
            raw = row['publisher_url']; u = urlsplit(raw)
            require(u.scheme == 'https' and u.hostname and u.netloc == u.hostname
                    and not u.query and not u.fragment and '%' not in u.path
                    and all(v not in ('.', '..') for v in u.path.split('/')),
                    'Exact canonical HTTPS publisher sequence URL required')
            require(u.path.endswith('/' + str(row['native_number_one_based']) + '.png'),
                    'Publisher URL does not bind original ordinal')
            base = 'https://' + u.netloc + u.path.rsplit('/', 1)[0] + '/'
            require('//' not in u.path, 'Publisher path alias is ambiguous')
            bases.add(base)
        require(len(bases) == 1, 'One native PNG window mixes publisher sequences')
        keys = [['png_publisher_index_sha256', provenance['publisher_sha_index']['sha256']],
                ['png_publisher_sequence_directory', next(iter(bases))]]
        basis = 'publisher_native_sequence_index_zero_based_not_proxy_PTS'
    return {'source_kind': kind, 'source_collision_keys': keys,
            'source_frame_index_domain': basis, 'window_inclusive': [ids[0], ids[-1]]}


def read_registry(root, descriptor):
    root = Path(root)
    value = checked_json(root, descriptor)
    require(set(value) == {'schema', 'status', 'training_ready',
            'formal_training_allowed', 'source_selection_sha256',
            'source_group_splits', 'samples'}, 'Unexpected registry schema')
    require(value['schema'] == SCHEMA and value['status'] == 'frozen_candidate_not_training_ready'
            and value['training_ready'] is False and value['formal_training_allowed'] is False,
            'Registry cannot upgrade a child cache to READY')
    loader = strict_loader()
    loader.sha_value(value['source_selection_sha256'])
    groups = value['source_group_splits']
    require(type(groups) is dict and groups and
            all(type(k) is str and v in ('train', 'dev', 'test') for k, v in groups.items()),
            'Frozen film-level partition required')
    require(type(value['samples']) is list and value['samples'], 'Empty registry')
    rows, ids, windows, source_splits, science = [], set(), {}, {}, None
    identities = {}
    for row in value['samples']:
        require(set(row) == {'sample_id', 'source_group', 'split', 'contract',
                'pair_root', 'pair', 'teacher_root', 'teacher', 'native_acceptance'},
                'Unexpected child descriptor schema')
        require(row['sample_id'] not in ids, 'Duplicate sample ID')
        ids.add(row['sample_id'])
        contract = checked_json(root, row['contract'])
        numeric = common_science(contract)
        if science is None:
            science = numeric
        require(science == numeric, 'Numeric recipes/geometry/precision differ between pairs')
        require(row['source_group'] in groups and groups[row['source_group']] == row['split'],
                'Registry source split leakage')
        for group, split in contract['generation_binding']['source_group_splits'].items():
            require(group in groups and groups[group] == split, 'Child/aggregate film split conflict')
        pair_root = directory(root, row['pair_root'])
        pair = checked_json(pair_root, row['pair'])
        loader.validate_pair_record(pair, contract)
        require(all(row[k] == pair[k] for k in ('sample_id', 'source_group', 'split')),
                'Registry/pair sample identity differs')
        # Preserve and hash manual/native evidence; its scientific acceptance is
        # an external admission obligation, not inferred from a filename/status.
        checked_json(root, row['native_acceptance'])
        identity = source_identity(pair)
        if identity['source_kind'] == 'native_png_sequence':
            require(row['native_acceptance']['sha256'] ==
                    pair['native_png_source']['native_acceptance']['sha256'],
                    'Registry/native PNG acceptance identity differs')
        identities[row['sample_id']] = identity
        lo, hi = identity['window_inclusive']
        for key in identity['source_collision_keys']:
            key = tuple(key)
            require(source_splits.get(key, row['split']) == row['split'],
                    'One original source crosses splits under different group aliases')
            source_splits[key] = row['split']
            for a, b in windows.setdefault(key, []):
                require(hi < a or lo > b, 'Overlapping source226 windows')
            windows[key].append((lo, hi))
        if row['split'] == 'test':
            require(row['teacher'] is None and row['teacher_root'] is None,
                    'Test teacher generation/access requires a later separate protocol')
        else:
            teacher_root = directory(root, row['teacher_root'])
            teacher = checked_json(teacher_root, row['teacher'])
            loader.validate_teacher_record(teacher, pair, contract, row['pair']['sha256'])
        rows.append(copy.deepcopy(row))
    require(set(groups) == {r['source_group'] for r in rows}, 'Unrepresented source group in registry')
    return {'root': str(root), 'descriptor': copy.deepcopy(descriptor),
            'registry': value, 'rows': rows, 'source_identities': identities, 'common_science': science,
            'common_science_sha256': canonical_sha(science),
            'scope': 'immutable_candidate_registry_not_formal_admission'}


def rows_for(dataset, split):
    require(split in ('train', 'dev'), 'Test model access is locked in this training adapter')
    return [copy.deepcopy(r) for r in dataset['rows'] if r['split'] == split]


def load_bundles(dataset, sample_id, *, split):
    """Strict per-child read; no shared-contract relabeling or implicit casting."""
    matches = [r for r in rows_for(dataset, split) if r['sample_id'] == sample_id]
    require(len(matches) == 1, 'Missing sample or wrong requested split')
    row = matches[0]
    root, loader = Path(dataset['root']), strict_loader()
    contract = checked_json(root, row['contract'])
    require(common_science(contract) == dataset['common_science'], 'Child science drift')
    pair = loader.load_pair(directory(root, row['pair_root']), row['pair']['path'],
                            row['pair']['sha256'], contract)
    teacher = loader.load_teacher(directory(root, row['teacher_root']), row['teacher']['path'],
                                  row['teacher']['sha256'], pair, contract)
    require(all(pair['record'][k] == row[k] for k in ('sample_id', 'source_group', 'split')),
            'Loaded sample identity differs')
    return pair, teacher, contract


def payload_snapshot(dataset):
    """Current full-file SHA inventory; arrays stay on disk and are not rewritten."""
    root, loader = Path(dataset['root']), strict_loader()
    paths = {}

    def add(base, desc):
        p = loader.safe_member(base, desc['path'])
        actual = {'bytes': p.stat().st_size, 'sha256': file_sha(p)}
        require(actual['sha256'] == desc['sha256'] and
                ('bytes' not in desc or actual['bytes'] == desc['bytes']), 'Snapshot identity differs')
        paths[str(p.relative_to(root))] = actual

    add(root, dataset['descriptor'])
    for row in dataset['rows']:
        add(root, row['contract']); add(root, row['native_acceptance'])
        base = directory(root, row['pair_root']); add(base, row['pair'])
        pair = checked_json(base, row['pair'])
        for d in list(pair['arrays'].values()) + list(pair['evidence'].values()):
            add(base, d)
        if pair.get('source_kind') == 'native_png_sequence':
            for role in ('publisher_sha_index', 'native_acceptance', 'registration_report'):
                add(base, pair['native_png_source'][role])
        if row['teacher'] is not None:
            base = directory(root, row['teacher_root']); add(base, row['teacher'])
            teacher = checked_json(base, row['teacher'])
            add(base, teacher['latent'])
            add(base, {'path': teacher['generation_receipt_path'],
                       'sha256': teacher['generation_receipt_sha256']})
    return paths


def audit_payloads(dataset):
    """Explicit CPU-only actual-data audit, never invoked automatically on import.

    Test pairs are checked for bytes/provenance only, without teacher or model.
    Native manual judgments and formal preflight remain separate required gates.
    """
    before = payload_snapshot(dataset)
    root, loader = Path(dataset['root']), strict_loader()
    for row in dataset['rows']:
        contract = checked_json(root, row['contract'])
        pair = loader.load_pair(directory(root, row['pair_root']), row['pair']['path'],
                                row['pair']['sha256'], contract)
        if row['split'] != 'test':
            teacher = loader.load_teacher(directory(root, row['teacher_root']), row['teacher']['path'],
                                          row['teacher']['sha256'], pair, contract)
            del teacher
        del pair
    after = payload_snapshot(dataset)
    require(before == after, 'Payload changed during full audit')
    return {'schema': AUDIT_SCHEMA, 'status': 'all_child_payloads_verified_not_formal_admission',
            'registry_sha256': dataset['descriptor']['sha256'],
            'checker_sha256': file_sha(__file__), 'strict_loader_sha256': LOADER_SHA,
            'common_science_sha256': dataset['common_science_sha256'],
            'sample_order': [r['sample_id'] for r in dataset['rows']],
            'counts': {s: sum(r['split'] == s for r in dataset['rows']) for s in ('train', 'dev', 'test')},
            'snapshot_before': before, 'snapshot_after': after,
            'child_training_ready_flags_rewritten': False, 'formal_training_allowed': False,
            'manual_native_acceptance_issued_here': False, 'model_inference_count': 0}
