"""Read-only candidate data interface for explicit H3 67/226 geometry.

No writer, encoder, model, resize, normalization, download or training entry.
New cache format: separate allow_pickle=False NPY members, immutable JSON/SHA
bindings. Native visible HR is distinct from compute padding and teacher output.
This is NOT a READY validator: scientific source/generation protocols and actual
model/data audits remain external and must be bound before real cache use.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import math
import re
from fractions import Fraction
from pathlib import Path

GEOMETRY_SOURCE_SHA = '147765ef79549bc31fd2b9e5026d5983be2328ee12ee04743f9202543b4af383'
GEOMETRY_PATH = Path(__file__).with_name('h3_production67_geometry_v1.py')
SCHEMA = 'h3-production67-pair-cache-v1-candidate'
GENERATION_SCHEMA = 'h3-production67-generation-binding-v1-candidate'
PAIR_SCHEMA = 'h3-production67-native-pair-v1-candidate'
TEACHER_SCHEMA = 'h3-production67-teacher-cache-v1-candidate'
INDEX_SCHEMA = 'h3-production67-pair-index-v1-candidate'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def sha_value(value):
    require(type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None,
            'Expected exact lowercase SHA256')
    return value


def exact(value, expected, message):
    # Avoid accepting True as1 or float226.0 asinteger226 in contracts.
    require(json.dumps(value, sort_keys=True, allow_nan=False) ==
            json.dumps(expected, sort_keys=True, allow_nan=False), message)


def geometry_module():
    require(file_sha(GEOMETRY_PATH) == GEOMETRY_SOURCE_SHA, 'Frozen geometry source changed')
    spec = importlib.util.spec_from_file_location('h3_production67_geometry_v1', GEOMETRY_PATH)
    # dataclasses accesses the defining module through sys.modules.
    import sys
    mod = sys.modules.get(spec.name)
    if mod is None:
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    return mod


def cache_layout():
    return {'draft': {'shape': [24, 67, 44, 80], 'dtype': '<f4'},
            'reference_hr': {'shape': [24, 67, 68, 120], 'dtype': '<f4'},
            'native_hr_rgb': {'shape': [226, 1080, 1920, 3], 'dtype': '|u1'},
            'flow': {'shape': [66, 2, 68, 120], 'dtype': '<f4'},
            'latent_valid': {'shape': [67], 'dtype': '|b1'},
            'flow_valid': {'shape': [66], 'dtype': '|b1'}}


def temporal_contract():
    g = geometry_module()
    anchors = list(g.latent_anchors(67))
    require(anchors[-1] == 225 and len(anchors) == 67, 'Unexpected anchor helper')
    return {'source_frames': 226, 'rgb_frames': 226, 'fps': [24, 1],
            'latent_frames': 67, 'encoder_internal_pad_frames': 12,
            'encoder_internal_frame_count': 238, 'external_time_padding': False,
            'external_slowdown': False, 'decoder_raw_frame_count': 226,
            'latent_anchor_frames': anchors,
            'latent_valid': [True] * 67, 'flow_valid': [True] * 66,
            'anchor_pairs': [[a, b] for a, b in zip(anchors, anchors[1:])],
            'anchor_meaning': 'encoder_source_dependency_not_point_sampled_rgb'}


def flow_contract():
    return {'shape': [66, 2, 68, 120], 'direction': 'current_to_previous',
            'source_domain': 'hr_compute_rgb_raw1088', 'source_hw': [1088, 1920],
            'estimation_hw': [272, 480], 'target_grid_hw': [68, 120],
            'vector_units': 'target_latent_pixels_xy',
            'quarter_to_latent_vector_scale_xy': [[1, 4], [1, 4]],
            'spatial_mask': 'inbounds_only', 'occlusion_confidence_mask': False,
            'sampling': 'grid_sample_bilinear_border_align_corners_true',
            'generation_implemented_here': False}


def roi_contract():
    return {'source': 'detached_decoded_reference_hr',
            'coordinate_domain': 'visible_rgb_after_raw_crop',
            'visible_hw': [1080, 1920], 'raw_crop_y0_y1_x0_x1': [4, 1084, 0, 1920],
            'frame_count': 226, 'all_adjacent_pairs': 225,
            'criterion': 'mean_channel_abs_current_minus_previous_greater_equal',
            'threshold': 0.05, 'rgb_domain': 'decoder_unclipped_float32',
            'both_pair_endpoints': True, 'shared_between_training_arms': True,
            'teacher_derived_mask_allowed': False, 'cached_mask': False,
            'sparse_selection': 'only_actual_global_rgb_adjacency_no_packed_gap_bridging'}


def candidate_contract():
    """Unbound proposal, deliberately unusable by load_pair/load_teacher."""
    g = geometry_module().production67()
    return {'schema': SCHEMA, 'geometry': g.description(),
            'geometry_sha256': g.sha256, 'temporal': temporal_contract(),
            'layout': cache_layout(), 'flow': flow_contract(), 'roi': roi_contract(),
            'latent_storage_domain': 'open_h3_vae_affine_normalized_(raw-mean)/std',
            'teacher_wrapper_chain': 'native_vae_normalized_to_upscaler_node_normalized_to_native_vae_normalized',
            'storage_precision': 'FP32_no_implicit_cast',
            'generation_binding': None, 'generation_binding_sha256': None,
            'formal_training_allowed': False, 'private_production_equivalent': False}


BINDING_HASHES = ('source_split_contract_sha256', 'selected_windows_sha256',
    'native_rgb_reader_sha256', 'hr_compute_construction_sha256', 'lr_degradation_sha256',
    'latent_normalization_sha256', 'posterior_recipe_sha256', 'vae_checkpoint_sha256',
    'teacher_checkpoint_sha256', 'teacher_adapter_sha256', 'decoder_checkpoint_sha256',
    'execution_precision_sha256', 'flow_generator_sha256')


def validate_binding(binding):
    require(type(binding) is dict and set(binding) ==
            set(BINDING_HASHES) | {'schema', 'status', 'source_group_splits',
                                  'private_production_equivalent'}, 'Incomplete generation binding')
    require(binding['schema'] == GENERATION_SCHEMA and
            binding['status'] == 'frozen_candidate_not_training_ready' and
            binding['private_production_equivalent'] is False, 'Generation scope changed')
    for key in BINDING_HASHES:
        sha_value(binding[key])
    splits = binding['source_group_splits']
    require(type(splits) is dict and splits, 'Need frozen film/source-group splits')
    for group, split in splits.items():
        require(type(group) is str and re.fullmatch(r'[a-zA-Z0-9_-]+', group) is not None
                and split in ('train', 'dev', 'test'), 'Invalid film-level split')
    return binding


def bind_candidate(binding):
    """Bind caller-reviewed protocol hashes; does not certify their science/content."""
    validate_binding(binding)
    result = candidate_contract()
    result['generation_binding'] = copy.deepcopy(binding)
    result['generation_binding_sha256'] = canonical_sha(binding)
    return result


def validate_contract(contract):
    binding = validate_binding(contract.get('generation_binding'))
    expected = bind_candidate(binding)
    exact(contract, expected, 'Unbound, stale, old2x/37token or modified data contract')
    return contract


def safe_member(root, relative):
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'Real absolute cache root required')
    require(type(relative) is str and relative and not Path(relative).is_absolute()
            and all(p not in ('', '.', '..') for p in relative.split('/')), 'Unsafe member path')
    result = root / relative
    require(result.resolve() == result and result.is_file(), 'Missing or symlink cache member')
    require(result.is_relative_to(root), 'Member escaped cache root')
    return result


def checked_json(root, relative, expected_sha, *, cap=4 << 20):
    p = safe_member(root, relative)
    require(p.stat().st_size <= cap, 'JSON size cap exceeded')
    raw = p.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == sha_value(expected_sha), 'JSON SHA mismatch')
    return json.loads(raw)


def validate_array_record(record, expected):
    require(type(record) is dict and set(record) == {'path', 'bytes', 'sha256', 'shape', 'dtype'},
            'Array record schema changed')
    exact(record['shape'], expected['shape'], 'Wrong payload geometry; no resize permitted')
    require(record['dtype'] == expected['dtype'], 'Storage dtype changed')
    sha_value(record['sha256'])
    n = math.prod(expected['shape'])
    width = {'<f4': 4, '|u1': 1, '|b1': 1}[expected['dtype']]
    require(type(record['bytes']) is int and n * width < record['bytes'] <= n * width + 65536,
            'Unexpected NPY byte length')


def checked_array(root, record, expected):
    """Real finite validation using a read-only mmap; no fullRGB heap allocation."""
    import numpy as np
    validate_array_record(record, expected)
    p = safe_member(root, record['path']); before = p.stat()
    require(before.st_size == record['bytes'] and file_sha(p) == record['sha256'], 'Array bytes/SHA mismatch')
    value = np.load(p, mmap_mode='r', allow_pickle=False, max_header_size=16384)
    require(isinstance(value, np.memmap) and list(value.shape) == expected['shape']
            and value.dtype.str == expected['dtype'] and value.flags.c_contiguous
            and not value.flags.writeable, 'NPY actual shape/dtype/layout mismatch')
    require(value.offset + value.nbytes == before.st_size, 'Trailing or missing NPY payload')
    if value.dtype.kind == 'f':
        flat = value.reshape(-1)
        for start in range(0, flat.size, 1 << 20):
            require(bool(np.isfinite(flat[start:start + (1 << 20)]).all()), 'Nonfinite payload')
    after = p.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            and file_sha(p) == record['sha256'], 'Array changed while validating')
    return value


def validate_time(time):
    require(type(time) is dict and set(time) == {'source_frame_indices', 'source_pts'}, 'Missing actual source-time proof')
    ids, pts = time['source_frame_indices'], time['source_pts']
    require(type(ids) is list and len(ids) == 226 and all(type(i) is int and i >= 0 for i in ids), 'Need226 actual source indices')
    exact(ids, list(range(ids[0], ids[0] + 226)), 'Source frames must be consecutive, unique, not repeated')
    require(type(pts) is list and len(pts) == 226, 'Need226 actual source PTS')
    times = []
    for pair in pts:
        require(type(pair) is list and len(pair) == 2 and all(type(x) is int for x in pair)
                and pair[0] >= 0 and pair[1] > 0, 'Invalid rational source PTS')
        times.append(Fraction(*pair))
    require(all(t - times[0] == Fraction(i, 24) for i, t in enumerate(times)), 'Source PTS must be consecutive24fps')


def sample_identity(record, contract):
    require(type(record['sample_id']) is str and re.fullmatch(r'[A-Za-z0-9_-]+', record['sample_id']) is not None,
            'Invalid sample ID')
    splits = contract['generation_binding']['source_group_splits']
    require(record['source_group'] in splits and record['split'] == splits[record['source_group']],
            'Source group split leakage or identity changed')
    for k in ('source_video_sha256', 'source_window_receipt_sha256', 'native_hr_rgb_sha256',
              'hr_compute_rgb_sha256', 'lr_rgb_sha256', 'pair_generation_receipt_sha256'):
        sha_value(record[k])
    require(record['generation_binding_sha256'] == contract['generation_binding_sha256'] and
            record['data_contract_sha256'] == canonical_sha(contract), 'Pair generation contract mismatch')
    validate_time(record['source_time'])
    native = record['native_hr_identity']
    require(type(native) is dict, 'Native HR provenance required')
    exact(native.get('visible_hw'), [1080, 1920], 'Native visibleHR dimensions wrong')
    require(native.get('kind') == 'original_source_native_crop_or_identity' and
            native.get('is_upscaled') is False and native.get('is_model_output') is False,
            'Teacher/generated/upscaled images cannot be native HR')
    require(native.get('source_video_sha256') == record['source_video_sha256'] and
            native.get('native_hr_rgb_sha256') == record['native_hr_rgb_sha256'] and
            native.get('source_window_receipt_sha256') == record['source_window_receipt_sha256'],
            'Native HR/source/time identity mismatch')
    box, size = native.get('source_crop_y0_y1_x0_x1'), native.get('source_raster_hw')
    require(type(box) is list and len(box) == 4 and all(type(x) is int and x >= 0 for x in box)
            and type(size) is list and len(size) == 2 and all(type(x) is int and x > 0 for x in size),
            'Native crop provenance malformed')
    require(box[1]-box[0] == 1080 and box[3]-box[2] == 1920 and box[1] <= size[0]
            and box[3] <= size[1], 'Native HR must be a real1080x1920 crop; resizing disallowed')


def validate_pair_record(record, contract):
    validate_contract(contract)
    require(record.get('schema') == PAIR_SCHEMA and record.get('training_ready') is False,
            'Wrong pair schema or unsupported READY claim')
    sample_identity(record, contract)
    exact(record['temporal'], contract['temporal'], 'Actual encoder/decoder/time contract differs')
    exact(record['compute_output_hw'], [1088, 1920], 'Compute canvas wrong')
    exact(record['visible_crop_y0_y1_x0_x1'], [4, 1084, 0, 1920], 'Visible crop changed')
    exact(record['normalization_binding_sha256'], contract['generation_binding']['latent_normalization_sha256'],
          'Normalization identity changed')
    require(set(record['arrays']) == set(cache_layout()), 'Missing/extra pair payloads')
    for role, layout in cache_layout().items():
        validate_array_record(record['arrays'][role], layout)
    require(record['arrays']['native_hr_rgb']['sha256'] == record['native_hr_rgb_sha256'], 'Native source payload identity changed')
    flow = record['flow_metadata']
    exact(flow['contract'], contract['flow'], 'Flow domain/grid/direction changed')
    require(flow['draft_latent_sha256'] == record['arrays']['draft']['sha256']
            and flow['reference_hr_latent_sha256'] == record['arrays']['reference_hr']['sha256']
            and flow['hr_compute_rgb_sha256'] == record['hr_compute_rgb_sha256']
            and flow['generator_sha256'] == contract['generation_binding']['flow_generator_sha256'],
            'Flow source/payload identity mismatch')
    exact(flow['anchor_pairs'], contract['temporal']['anchor_pairs'], 'Flow time anchors changed')
    exact(record['roi_metadata'], dict(contract['roi'],
          reference_hr_latent_sha256=record['arrays']['reference_hr']['sha256'],
          decoder_checkpoint_sha256=contract['generation_binding']['decoder_checkpoint_sha256']),
          'ROI must use common decodedHR visible crop, not teacher or latent grid')
    expected_roles = {'source_window', 'pair_generation', 'flow_generation'}
    require(set(record['evidence']) == expected_roles, 'Missing actual data-generation receipts')
    for role, item in record['evidence'].items():
        require(type(item) is dict and set(item) == {'path', 'sha256'}, 'Receipt descriptor malformed')
        sha_value(item['sha256'])
    require(record['evidence']['source_window']['sha256'] == record['source_window_receipt_sha256']
            and record['evidence']['pair_generation']['sha256'] == record['pair_generation_receipt_sha256'],
            'Receipt SHA binding mismatch')
    return record


def expected_pair_evidence(record, contract):
    # Receipts avoid pair-record circular hashes: they bind immutable payloads,
    # source identities and scientific contract, then pair JSON binds receipts.
    common = {'sample_id': record['sample_id'], 'source_group': record['source_group'],
              'split': record['split'], 'data_contract_sha256': canonical_sha(contract),
              'generation_binding_sha256': contract['generation_binding_sha256']}
    return {
        'source_window': dict(common, schema='h3-production67-source-window-receipt-v1',
            source_video_sha256=record['source_video_sha256'], source_time=record['source_time'],
            native_hr_identity={k: v for k, v in record['native_hr_identity'].items()
                                if k != 'source_window_receipt_sha256'},
            native_hr_rgb_sha256=record['native_hr_rgb_sha256'],
            reader_sha256=contract['generation_binding']['native_rgb_reader_sha256']),
        'pair_generation': dict(common, schema='h3-production67-pair-generation-receipt-v1',
            temporal=record['temporal'], lr_rgb_sha256=record['lr_rgb_sha256'],
            hr_compute_rgb_sha256=record['hr_compute_rgb_sha256'],
            native_hr_rgb_sha256=record['native_hr_rgb_sha256'],
            actual_encoded_shapes={k: record['arrays'][k]['shape'] for k in ('draft', 'reference_hr')},
            payload_sha256={k: record['arrays'][k]['sha256'] for k in ('draft', 'reference_hr', 'latent_valid')},
            generation_binding=contract['generation_binding']),
        'flow_generation': dict(common, schema='h3-production67-flow-generation-receipt-v1',
            flow_metadata=record['flow_metadata'],
            flow_sha256=record['arrays']['flow']['sha256'],
            valid_pairs_sha256=record['arrays']['flow_valid']['sha256'])}


def load_pair(root, relative, expected_sha, contract):
    record = checked_json(root, relative, expected_sha)
    validate_pair_record(record, contract)
    for role, expected in expected_pair_evidence(record, contract).items():
        evidence = record['evidence'][role]
        exact(checked_json(root, evidence['path'], evidence['sha256']), expected,
              'Actual generation receipt mismatch: ' + role)
    values = {role: checked_array(root, desc, cache_layout()[role])
              for role, desc in record['arrays'].items()}
    require(bool(values['latent_valid'].all()) and bool(values['flow_valid'].all()),
            '67/226 requires67 real latent anchors and66 valid warp pairs')
    # All values are read-only CPU arrays, no casting/normalizing/reshaping of olddata.
    return {'record': record, 'pair_record_sha256': expected_sha, 'arrays': values,
            'training_ready': False}


def validate_teacher_record(record, pair, contract, pair_record_sha256):
    validate_pair_record(pair, contract)
    require(record.get('schema') == TEACHER_SCHEMA and record.get('training_ready') is False,
            'Teacher cache schema/READY changed')
    require(pair['split'] in ('train', 'dev'), 'Heldout teacher-cache generation requires separate gate/protocol')
    for k in ('sample_id', 'source_group', 'split', 'generation_binding_sha256', 'data_contract_sha256'):
        exact(record.get(k), pair[k], 'Teacher/pair identity mismatch: ' + k)
    require(record['pair_record_sha256'] == sha_value(pair_record_sha256), 'Teacher pair-record SHA mismatch')
    exact(record['label'], {'kind': 'frozen_bidirectional_upscaler', 'native_hr_ground_truth': False,
          'input_draft_sha256': pair['arrays']['draft']['sha256'],
          'checkpoint_sha256': contract['generation_binding']['teacher_checkpoint_sha256'],
          'adapter_sha256': contract['generation_binding']['teacher_adapter_sha256'],
          'normalization_binding_sha256': contract['generation_binding']['latent_normalization_sha256'],
          'execution_precision_sha256': contract['generation_binding']['execution_precision_sha256'],
          'explicit_target_thw': [67, 68, 120], 'enable_chunking': False},
          'Teacher same-input/normalization/precision/target identity changed')
    validate_array_record(record['latent'], {'shape': [24, 67, 68, 120], 'dtype': '<f4'})
    sha_value(record['generation_receipt_sha256'])
    require(type(record['generation_receipt_path']) is str, 'Need actual teacher-generation receipt')
    return record


def load_teacher(root, relative, expected_sha, pair_bundle, contract):
    record = checked_json(root, relative, expected_sha)
    validate_teacher_record(record, pair_bundle['record'], contract, pair_bundle['pair_record_sha256'])
    exact(checked_json(root, record['generation_receipt_path'], record['generation_receipt_sha256']),
          {'schema': 'h3-production67-teacher-generation-receipt-v1',
           'sample_id': record['sample_id'], 'pair_record_sha256': record['pair_record_sha256'],
           'data_contract_sha256': record['data_contract_sha256'], 'label': record['label'],
           'actual_output_shape': record['latent']['shape'],
           'teacher_latent_sha256': record['latent']['sha256']}, 'Actual teacher receipt mismatch')
    value = checked_array(root, record['latent'], {'shape': [24, 67, 68, 120], 'dtype': '<f4'})
    return {'record': record, 'teacher_record_sha256': expected_sha, 'latent': value,
            'training_ready': False}


def read_index(root, relative, expected_sha, contract):
    validate_contract(contract)
    index = checked_json(root, relative, expected_sha)
    require(index['schema'] == INDEX_SCHEMA and index['training_ready'] is False
            and index['data_contract_sha256'] == canonical_sha(contract), 'Wrong index identity/schema')
    require(type(index['samples']) is list and index['samples'], 'Empty index')
    ids = set()
    for row in index['samples']:
        require(row['sample_id'] not in ids, 'Duplicate sample ID'); ids.add(row['sample_id'])
        r = checked_json(root, row['pair_record_path'], row['pair_record_sha256'])
        validate_pair_record(r, contract)
        for key in ('sample_id', 'source_group', 'split'):
            exact(row[key], r[key], 'Index/pair binding mismatch')
    return copy.deepcopy(index)


def torch_arrays(bundle, *, include_native_hr_rgb=False):
    """Explicit copy into CPU Torch. No device move, scale change or trainer call.

    Native fullRGB is omitted unless requested; downstream code may read mmap
    source crops instead. FP32/bool/uint8 preserved; gradients remain disabled.
    """
    import numpy as np
    import torch
    require(bundle.get('training_ready') is False and 'arrays' in bundle, 'Expected checked pair bundle')
    return {role: torch.from_numpy(np.array(value, copy=True, order='C')).requires_grad_(False)
            for role, value in bundle['arrays'].items()
            if include_native_hr_rgb or role != 'native_hr_rgb'}


def torch_teacher(teacher_bundle):
    """Explicit detached CPU copy; no dtype/device/geometry change."""
    import numpy as np
    import torch
    require(teacher_bundle.get('training_ready') is False and 'latent' in teacher_bundle,
            'Expected checked teacher bundle')
    return torch.from_numpy(np.array(teacher_bundle['latent'], copy=True, order='C')).requires_grad_(False)


def checked_pixel_spec(spec, *, temporal_scope, min_spatial=64, lpips_count=4):
    """Reuse reviewed catalog; does not adopt an objective or perform decode."""
    g = geometry_module()
    path = Path(__file__).with_name('h3_production67_pixel_specs_v1.py')
    require(file_sha(path) == '50ea70c63a31ce14a3aec11b5072d369bcbd2d6b739a08b049e08cd40e2edfd0',
            'Frozen pixel-spec source changed')
    module_spec = importlib.util.spec_from_file_location('production67_pair_pixel_specs', path)
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    catalog = module.PixelSpecCatalog(g.production67(), temporal_scope=temporal_scope,
                                      min_spatial=min_spatial, lpips_count=lpips_count)
    catalog.validate_spec(spec)
    return copy.deepcopy(spec)
