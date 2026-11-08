"""Explicit two-cohort metadata/router candidate; no merged common_science.

Each subgroup is validated by the unchanged frozen binder2. Global source/split/
overlap checks are applied again across subgroups. No registry, cache, approval,
model or training state is created by this module. Actual payload reads occur
only in the explicit train/dev load_bundles accessor through the frozen binder.
"""
from __future__ import annotations
import copy
import importlib.util
from pathlib import Path

SCHEMA = 'h3-production67-two-cohort-router-v1-candidate'
POLICY_SHA = '3141c281a426ffd6b5d25feaec7118a066ba8fe4ccff01b7a3ad48c4fe9d0f76'
BINDER_SHA = '4fb6fe4c9823dbdd233c726f06e398f29854a3e507875e7b56ea6683b6236941'
SCIENCE_SHA = {
    'original_generation_mode_unbound': 'd0af8222ed095516b01bd5897a66c4b4ec9b898c97fc7ddf4d79034ca6afe2db',
    'strictdet_generation': '59743f52860e421d6c7037cd6508e135553bd5a74f9917617e8d75f50722fc2a'}
MODE_HASH_FIELDS = {'execution_precision_sha256', 'posterior_recipe_sha256',
                    'teacher_adapter_sha256'}
MAX_METADATA_BYTES = 4 << 20


def require(ok, message):
    if not ok:
        raise ValueError(message)


def frozen_binder():
    import hashlib
    p = Path(__file__).with_name('h3_production67_dataset_binding_v2.py')
    require(hashlib.sha256(p.read_bytes()).hexdigest() == BINDER_SHA,
            'Frozen binder2 source changed')
    s = importlib.util.spec_from_file_location('h3_cohort_frozen_binder_v2', p)
    b = importlib.util.module_from_spec(s)
    s.loader.exec_module(b)
    return b


def load_policy():
    import hashlib, json
    p = Path(__file__).with_name('h3_production67_mixed_cohort_policy_v1.json')
    raw = p.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == POLICY_SHA, 'Frozen cohort policy changed')
    v = json.loads(raw)
    b = frozen_binder()
    require(v['binder_sha256'] == BINDER_SHA and set(v['cohorts']) == set(SCIENCE_SHA),
            'Policy cohort identities changed')
    for cid, expected in SCIENCE_SHA.items():
        row = v['cohorts'][cid]
        require(row['common_science_sha256'] == expected and
                b.canonical_sha(row['common_science']) == expected,
                'Policy full science identity changed')
    old = v['cohorts']['original_generation_mode_unbound']['common_science']
    new = v['cohorts']['strictdet_generation']['common_science']
    require(set(old) == set(new), 'Policy science structure differs')
    differences = []
    for k in old:
        if k != 'numeric_generation_binding':
            require(b.canonical_sha(old[k]) == b.canonical_sha(new[k]),
                    'Non-mode policy science difference')
    a, z = old['numeric_generation_binding'], new['numeric_generation_binding']
    require(set(a) == set(z), 'Policy binding keys differ')
    for k in sorted(a):
        if a[k] != z[k]:
            differences.append({'field': '/numeric_generation_binding/' + k,
                                'operation': 'changed', 'old': a[k], 'new': z[k]})
    require({r['field'].rsplit('/', 1)[1] for r in differences} == MODE_HASH_FIELDS
            and differences == v['exact_cross_cohort_differences'],
            'Only the three pinned mode-derived SHA differences are allowed')
    require(v['training_ready'] is False and v['formal_training_allowed'] is False,
            'Policy cannot issue admission')
    return v


def validate_recipe(recipe, pair, contract, cohort_id, policy):
    """Validate original JSON, never strip/rewrite it for child admission."""
    b = frozen_binder()
    require(type(recipe) is dict, 'Original per-pair recipe required')
    keys = set(policy['recipe_base_field_names'])
    strict = cohort_id == 'strictdet_generation'
    if strict:
        keys |= {'runtime_mode', 'new_strict_mode_scope'}
    if pair.get('source_kind', 'video_source') == 'native_png_sequence':
        keys |= {'source_kind', 'source_time_basis'}
        require(recipe.get('source_kind') == 'native_png_sequence' and
                recipe.get('source_time_basis') == 'publisher_nominal_24fps_PNG_has_no_intrinsic_PTS',
                'PNG source/time identity cannot become video')
    require(set(recipe) == keys, 'Unexpected original recipe fields')
    require(type(recipe['schema']) is str and recipe['schema'] and
            type(recipe['source_acceptance_required']) is str and
            recipe['source_acceptance_required'], 'Recipe identity text missing')
    for k, value in policy['shared_recipe_fields'].items():
        require(b.canonical_sha(recipe[k]) == b.canonical_sha(value),
                'Shared original recipe field differs: ' + k)
    for k, value in policy['cohorts'][cohort_id]['recipe_mode_fields'].items():
        require(b.canonical_sha(recipe[k]) == b.canonical_sha(value),
                'Original cohort recipe mode changed: ' + k)
    if strict:
        require(b.canonical_sha(recipe['runtime_mode']) == b.canonical_sha(policy['runtime_mode']) and
                type(recipe['new_strict_mode_scope']) is str and
                recipe['new_strict_mode_scope'], 'Strict generation mode scope missing')
    # The unbound cohort key-set excludes runtime_mode and the exact three
    # mode field objects above exclude deterministic annotations. No defaults.
    mappings = {'hr_compute_construction_sha256': 'hr_compute',
                'lr_degradation_sha256': 'lr_degradation',
                'latent_normalization_sha256': 'normalization',
                'posterior_recipe_sha256': 'posterior',
                'teacher_adapter_sha256': 'teacher',
                'execution_precision_sha256': 'execution_precision'}
    for key, field in mappings.items():
        require(contract['generation_binding'][key] == b.canonical_sha(recipe[field]),
                'Recipe/child generation hash mismatch: ' + key)
    require(contract['generation_binding']['flow_generator_sha256'] ==
            recipe['flow']['original_generator_sha256'], 'Flow source identity differs')


def read_router(root, descriptor):
    """Metadata only; subgroup payload arrays are neither opened nor verified."""
    root = Path(root)
    b = frozen_binder()
    l = b.strict_loader()
    require(root.is_absolute() and root.resolve() == root and root.is_dir(),
            'Real absolute router root required')
    snapshot = {}

    def read(base, desc):
        p = l.safe_member(base, desc['path'])
        require(p.stat().st_size <= MAX_METADATA_BYTES, 'Metadata exceeds bounded read')
        value = b.checked_json(base, desc)
        now = {'bytes': p.stat().st_size, 'sha256': b.file_sha(p)}
        require(now == {'bytes': desc['bytes'], 'sha256': desc['sha256']},
                'Metadata changed during read')
        prior = snapshot.setdefault(str(p), now)
        require(prior == now, 'One file has conflicting descriptors')
        return value

    value = read(root, descriptor)
    require(set(value) == {'schema', 'status', 'training_ready', 'formal_training_allowed',
            'policy_sha256', 'source_selection_sha256', 'source_group_splits',
            'cohorts', 'sample_order'}, 'Unexpected router schema')
    require(value['schema'] == SCHEMA and
            value['status'] == 'frozen_candidate_not_training_ready' and
            value['training_ready'] is False and value['formal_training_allowed'] is False,
            'Router cannot upgrade data admission')
    require(value['policy_sha256'] == POLICY_SHA, 'Router/policy identity differs')
    l.sha_value(value['source_selection_sha256'])
    policy = load_policy()
    groups = value['source_group_splits']
    require(type(groups) is dict and groups and
            all(type(k) is str and v in ('train', 'dev', 'test') for k, v in groups.items()),
            'Global film-family split mapping required')
    require(type(value['cohorts']) is list and len(value['cohorts']) == 2,
            'Exactly two explicit cohorts required')
    cohorts, rows, route, recipes, identities = {}, {}, {}, {}, {}
    source_owner, windows = {}, {}
    for entry in value['cohorts']:
        require(type(entry) is dict and set(entry) == {'cohort_id', 'registry', 'recipes'},
                'Unexpected cohort descriptor schema')
        cid = entry['cohort_id']
        require(cid in SCIENCE_SHA and cid not in cohorts, 'Missing, duplicate or unknown cohort')
        local = read(root, entry['registry'])
        require(local['source_selection_sha256'] == value['source_selection_sha256'],
                'Cohorts must bind the same frozen global source selection')
        # Real original validator, not a projection or synthetic common science.
        child = b.read_registry(root, entry['registry'])
        require(child['common_science_sha256'] == SCIENCE_SHA[cid] and
                b.canonical_sha(child['common_science']) == SCIENCE_SHA[cid],
                'Cohort has an unapproved full science identity')
        require(type(entry['recipes']) is dict and
                set(entry['recipes']) == {r['sample_id'] for r in child['rows']},
                'Every child must retain its original recipe descriptor')
        for group, split in child['registry']['source_group_splits'].items():
            require(groups.get(group) == split, 'Cross-cohort film-family split conflict')
        for row in child['rows']:
            sid = row['sample_id']
            require(sid not in rows, 'Duplicate sample across cohorts')
            contract = read(root, row['contract'])
            pair_root = b.directory(root, row['pair_root'])
            pair = read(pair_root, row['pair'])
            read(root, row['native_acceptance'])
            recipe = read(root, entry['recipes'][sid])
            validate_recipe(recipe, pair, contract, cid, policy)
            if row['split'] != 'test':
                read(b.directory(root, row['teacher_root']), row['teacher'])
            identity = child['source_identities'][sid]
            lo, hi = identity['window_inclusive']
            for raw_key in identity['source_collision_keys']:
                key = tuple(raw_key)
                owner = (row['source_group'], row['split'], identity['source_frame_index_domain'])
                require(key not in source_owner or source_owner[key] == owner,
                        'Original source alias crosses family, split or frame domain')
                source_owner[key] = owner
                for a, z in windows.setdefault(key, []):
                    require(hi < a or lo > z, 'Overlapping windows across cohorts')
                windows[key].append((lo, hi))
            rows[sid], route[sid] = copy.deepcopy(row), cid
            recipes[sid] = copy.deepcopy(entry['recipes'][sid])
            identities[sid] = copy.deepcopy(identity)
        cohorts[cid] = child
    require(set(cohorts) == set(SCIENCE_SHA), 'Both named cohorts must be represented')
    require(set(groups) == {r['source_group'] for r in rows.values()},
            'Unrepresented global source family')
    order = value['sample_order']
    require(type(order) is list and all(type(x) is str for x in order) and
            len(order) == len(set(order)) and set(order) == set(rows),
            'Global sample order must enumerate each child exactly once')
    for p, expected in snapshot.items():
        require(Path(p).stat().st_size == expected['bytes'] and
                b.file_sha(p) == expected['sha256'], 'Metadata changed during validation')
    return {'root': str(root), 'descriptor': copy.deepcopy(descriptor), 'registry': value,
            'cohort_datasets': cohorts, 'cohort_for_sample': route,
            'rows': [rows[x] for x in order], 'recipe_descriptors': recipes,
            'source_identities': identities, 'metadata_snapshot': snapshot,
            'cohort_science_sha256': copy.deepcopy(SCIENCE_SHA),
            'generation_mode_semantics': {k: v['generation_mode_semantics']
                                         for k, v in policy['cohorts'].items()},
            'scope': 'metadata_validated_explicit_two_cohort_candidate_only',
            'actual_payloads_verified': False, 'training_ready': False,
            'formal_training_allowed': False}


def rows_for(dataset, split):
    require(split in ('train', 'dev'), 'Test is metadata-only in this router')
    return [copy.deepcopy(r) for r in dataset['rows'] if r['split'] == split]


def load_bundles(dataset, sample_id, *, split):
    """Explicit train/dev strict child load; no old/new contract substitution."""
    require(split in ('train', 'dev'), 'Test is metadata-only in this router')
    # Revalidate immutable small descriptors and routes before delegating an
    # actual payload read; caller mutation of the returned object is not trust.
    current = read_router(dataset['root'], dataset['descriptor'])
    matches = [r for r in rows_for(current, split) if r['sample_id'] == sample_id]
    require(len(matches) == 1, 'Missing sample or wrong requested split')
    cid = current['cohort_for_sample'][sample_id]
    return frozen_binder().load_bundles(current['cohort_datasets'][cid], sample_id, split=split)
