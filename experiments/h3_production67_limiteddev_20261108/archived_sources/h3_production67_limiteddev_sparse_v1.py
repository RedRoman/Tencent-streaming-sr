"""Sparse nonblend dev selection and common original-v5 step0 candidates.

No model factory, launcher, data writer, optimizer, test access or full-video
quality gate. The caller owns actual device locks and separately admitted data.
Every covered frame/LPIPS position/decoder span is reported explicitly.
"""
from __future__ import annotations
import contextlib
import copy
import hashlib
import importlib.util
import inspect
from pathlib import Path
import torch
import h3_production67_train_step_v1 as core
import h3_production67_limiteddev_core_v1 as paired
import h3_production67_limiteddev_data_v1 as data

PAIRED_SHA = 'b07513005e30e2faec2f58f298ec774ee6143a9eac685d249809dec34c994af0'
DATA_SHA = 'fe2a6e20c3a4779a73a947001f792896576d9d43b22cf29eddcaea9d3a9fcc75'
V5_LOADER_SHA = '03d043154f9d920a3c63ebeb39d713a825ea32dd7f9592b4264ced8e935a4512'
METRICS = paired.METRICS
require, file_sha, canonical_sha = paired.require, paired.file_sha, paired.canonical_sha


def check_sources():
    sources = paired.check_sources()
    require(file_sha(inspect.getfile(paired)) == PAIRED_SHA and
            file_sha(inspect.getfile(data)) == DATA_SHA, 'Reviewed thin interface changed')
    return dict(sources, sparse_evaluator=file_sha(__file__),
                pixel_decoder=paired.PROVIDER_SHA, explicit_wrapper=core.WRAPPER_SHA)


def describe_coverage(selection_plan):
    """Exact per-clip coverage; a union across different clips is not full QA."""
    require(type(selection_plan) is dict and set(selection_plan) == {'scope', 'samples'}, 'Selection plan schema differs')
    ids = [r['sample_id'] for r in selection_plan['samples']]
    require(selection_plan == paired.build_selection_plan(ids), 'Selection region/LPIPS plan differs')
    loader = core.strict_loader(); result = []
    for row in selection_plan['samples']:
        spec = loader.checked_pixel_spec(row['region'], temporal_scope='nonblend_only')
        frames = spec['rgb_frame_indices']; positions = row['lpips_packed_positions']
        require(positions == list(range(len(frames))), 'Sparse dev LPIPS must cover every selected RGB frame')
        spans = sorted({(piece['latent_start'], piece['latent_stop'])
                        for item in spec['temporal_assembly'] for piece in item['contributions']})
        pairs = [[a, b] for a, b in zip(frames, frames[1:]) if b == a + 1]
        result.append({'sample_id': row['sample_id'], 'block_index': spec['block_index'],
            'spec_sha256': canonical_sha(spec), 'upscaler_forward_bcthw': [1, 24, 67, 68, 120],
            'decoder_latent_spans_half_open': [list(s) for s in spans],
            'decoder_target_latent_yx': spec['spatial']['target_latent_yx'],
            'raw_compute_hw': [1088, 1920], 'visible_global_box_yx': spec['spatial']['visible_global_box_yx'],
            'decoder_local_crop_yx': spec['spatial']['decoder_local_box_yx'],
            'rgb_frame_indices': frames, 'selected_rgb_frames': len(frames),
            'unobserved_rgb_frame_indices': sorted(set(range(226)) - set(frames)),
            'lpips_packed_positions': positions, 'lpips_global_frame_indices': [frames[i] for i in positions],
            'lpips_execution_batch_size': 1, 'motion_delta_global_adjacent_pairs': pairs,
            'blend_frames_observed': 0, 'all226_frames_observed': False,
            'all166_nonblend_frames_observed_for_this_clip': False,
            'full67_latent_loss_computed': False})
    return {'schema': 'h3-production67-sparse-dev-coverage-v1',
            'scope': selection_plan['scope'], 'sample_order': ids, 'samples': result,
            'clip_weighting': 'equal_weight_per_dev_clip_after_region_mean',
            'native_HR_RGB_measured': False, 'HR_guard_domain': 'frozen_HR_latent_decoded_visible_RGB',
            'teacher_is_native_HR_GT': False, 'full226_dev_quality_gate': False,
            'test_access': False}


def lpips_one_frame_at_a_time(module):
    """Same frozen per-image network and final mean, explicit bounded batch1.

    This is a newly declared execution batch size, not a claim of NPU bit-parity
    with historical batched calls. The same path is used for both arms/step0.
    """
    def apply(left, right):
        require(left.ndim == 4 and left.shape == right.shape and left.shape[0] > 0,
                'LPIPS images must have matching nonempty NCHW')
        values = []
        for i in range(left.shape[0]):
            value = module(left[i:i+1], right[i:i+1])
            require(torch.is_tensor(value) and value.numel() == 1 and bool(torch.isfinite(value).all()),
                    'Expected one finite LPIPS distance per image')
            values.append(value.reshape(1))
        return torch.cat(values, dim=0)
    return apply


def visible_metrics(prediction, reference, teacher, *, frames, lpips_positions, lpips_fn, recipe):
    """Tiny-testable metric kernel; production entry checks full geometry."""
    values = core.pixel_terms_visible(prediction, reference, teacher,
        rgb_frame_indices=frames, lpips_packed_positions=lpips_positions,
        lpips_fn=lpips_one_frame_at_a_time(lpips_fn), recipe=recipe)
    return {k: float(values[k].detach()) for k in METRICS}


def tensor_identity(value):
    cpu = value.detach().cpu().contiguous()
    return {'shape': list(cpu.shape), 'dtype': str(cpu.dtype),
            'value_sha256': hashlib.sha256(memoryview(cpu.numpy()).cast('B')).hexdigest()}


def module_snapshot(modules):
    """Bounded per-tensor copy; retain no full-model CPU duplicate in the result."""
    result = {}
    for role, module in modules.items():
        require(all(not child.training for child in module.modules()), 'Selection requires unchanged eval mode')
        tensors = []
        for kind, iterator in (('parameter', module.named_parameters()), ('buffer', module.named_buffers())):
            for name, value in iterator:
                grad = value.grad
                tensors.append({'kind': kind, 'name': name, 'id': id(value), 'version': value._version,
                    'device': str(value.device), 'requires_grad': bool(value.requires_grad), **tensor_identity(value),
                    'grad': None if grad is None else {'id': id(grad), 'version': grad._version, **tensor_identity(grad)}})
        result[role] = {'module_modes': [(name, child.training) for name, child in module.named_modules()],
                        'tensors': tensors}
    return result


@contextlib.contextmanager
def readonly_selection(modules, device):
    """Preserve RNG and existing student gradients, also on an exception."""
    rng = paired.capture_rng(device); before = module_snapshot(modules)
    proof = {'scope': 'read_only_selection', 'preserves_existing_student_gradients': True}
    try:
        with torch.no_grad():
            yield proof
    finally:
        paired.restore_rng(rng, device)
        after = module_snapshot(modules)
        require(before == after, 'Selection mutated model/frozen values, modes, identities or gradients')
        proof.update(RNG_restored_exact=True, before_after_modules_exact=True,
                     module_snapshot_sha256=canonical_sha(before),
                     tensor_value_fingerprints={role: canonical_sha([
                         {k: t[k] for k in ('kind', 'name', 'shape', 'dtype', 'value_sha256')}
                         for t in state['tensors']]) for role, state in before.items()})


def load_dev_example(dataset, sample_id, device):
    pair, teacher, contract = data.load_bundles(dataset, sample_id, split='dev')
    loader = core.strict_loader(); values = loader.torch_arrays(pair)
    example = {'draft': values['draft'].unsqueeze(0), 'reference': values['reference_hr'].unsqueeze(0),
               'teacher': loader.torch_teacher(teacher).unsqueeze(0), 'flow': values['flow'], 'valid': values['flow_valid'],
               'pair_bundle': pair, 'teacher_bundle': teacher, 'contract': contract,
               'scope': 'candidate_checked_cache_not_training_READY'}
    example = core.move_example(example, device=device); core.validate_example_tensors(example)
    return example


def evaluate_dev(*, student, provider, lpips_fn, dataset, selection_plan, step):
    """Exact callback signature consumed by pinned paired2000_core_v2.

    Caller owns actual device/locks/admission. No implicit2x, temporal truncation,
    scope fallback, mode changes, optimizer or test access occur here.
    """
    sources = check_sources()
    require(type(step) is int and 0 <= step <= paired.UPDATES and step % paired.VALIDATE_EVERY == 0,
            'Sparse selection runs only at frozen step0/200 cadence')
    dev_ids = [r['sample_id'] for r in data.rows_for(dataset, 'dev')]
    require(selection_plan == paired.build_selection_plan(dev_ids), 'Dev identity/order/specs changed')
    coverage = describe_coverage(selection_plan)
    require(file_sha(inspect.getfile(type(student))) == core.WRAPPER_SHA and
            file_sha(inspect.getfile(type(provider))) == paired.PROVIDER_SHA, 'Reviewed wrapper/provider source changed')
    device = provider.device
    precision = paired.precision(device); mode = paired.deterministic_mode()
    paired.check_frozen(provider, lpips_fn)
    require(all(p.device == device and p.dtype == torch.float32 for p in student.parameters()), 'Student device/dtype changed')
    recipe = core.frozen_recipe(Path(__file__).parent)
    result = {'schema': 'h3-production67-sparse-dev-observation-v1', 'step': step,
              'registry_sha256': dataset['descriptor']['sha256'],
              'selection_plan_sha256': canonical_sha(selection_plan),
              'selection_evaluator_sha256': file_sha(__file__), 'source_sha256': sources,
              'precision': precision, 'mode': mode, 'coverage': coverage, 'samples': [],
              'optimizer_updates': 0, 'full226_dev_quality_gate_passed': False, 'test_access': False}
    modules = {'student': student.model, 'decoder': provider.vae, 'LPIPS': lpips_fn}
    with readonly_selection(modules, device) as proof:
        for row in selection_plan['samples']:
            paired.deterministic_mode(); require(paired.precision(device) == precision, 'Selection precision drift')
            example = load_dev_example(dataset, row['sample_id'], device)
            prediction = core.explicit_forward67(student, example['draft'])
            require(bool(torch.isfinite(prediction).all()), 'Nonfinite selection prediction')
            spec = core.strict_loader().checked_pixel_spec(row['region'], temporal_scope='nonblend_only')
            decoded = [provider.decode_region(value, spec) for value in
                       (prediction, example['reference'], example['teacher'])]
            y0, y1, x0, x1 = spec['spatial']['visible_global_box_yx']
            expected = (1, 3, len(spec['rgb_frame_indices']), y1-y0, x1-x0)
            require(all(v.shape == expected and v.dtype == torch.float32 and v.device == device and
                    not v.requires_grad and bool(torch.isfinite(v).all()) for v in decoded), 'Sparse decode geometry/dtype/autograd differs')
            metrics = visible_metrics(*decoded, frames=spec['rgb_frame_indices'],
                lpips_positions=row['lpips_packed_positions'], lpips_fn=lpips_fn, recipe=recipe)
            result['samples'].append({'sample_id': row['sample_id'], 'metrics': metrics,
                'pair_record_sha256': example['pair_bundle']['pair_record_sha256'],
                'teacher_record_sha256': example['teacher_bundle']['teacher_record_sha256'],
                'child_contract_sha256': canonical_sha(example['contract']),
                'region_sha256': canonical_sha(spec)})
            del prediction, decoded, example
            paired.check_frozen(provider, lpips_fn)
        require(check_sources() == sources and paired.precision(device) == precision, 'Selection source/precision drift')
        paired.deterministic_mode()
    result['readonly_proof'] = proof
    return result


def fresh_v5_identity(student, tensor_manifest_path):
    path = Path(__file__).with_name('h3_production67_v5_loader_v1.py')
    require(file_sha(path) == V5_LOADER_SHA, 'Strict v5 identity reader changed')
    spec = importlib.util.spec_from_file_location('h3_sparse_v5_identity_v1', path)
    loader = importlib.util.module_from_spec(spec); spec.loader.exec_module(loader)
    schema = loader.load_tensor_manifest(tensor_manifest_path)
    require(student.source_checkpoint_sha256 == core.V5_SHA, 'Common step0 must use original v5')
    state = student.original_state_dict()
    require(set(state) == set(schema) and len(tuple(student.parameters())) == 322 and
            sum(p.numel() for p in student.parameters()) == 345280216, 'Common step0 model/schema differs')
    for name, value in state.items():
        actual = tensor_identity(value)
        require(actual['dtype'] == 'torch.float32' and actual['shape'] == schema[name]['shape'] and
                actual['value_sha256'] == schema[name]['tensor_sha256'], 'Common step0 is not fresh v5: ' + name)
    return {'initializer_checkpoint_sha256': core.V5_SHA,
            'tensor_manifest_sha256': loader.TENSOR_MANIFEST_SHA,
            'initial_tensor_schema_sha256': canonical_sha(schema),
            'actual_tensor_count': 322, 'actual_parameter_count': 345280216,
            'all_initial_values_match_frozen_v5': True}


def make_common_step0(*, student, provider, lpips_fn, dataset, selection_plan,
                     tensor_manifest_path, payload_audit):
    """Future real observation producer, not a formal-admission issuer.

    Requires all3 actual dev entries and an unchanged full payload audit. Source
    scientific acceptance and formal paired preflight remain external gates.
    Returns JSON data; caller writes a new exclusive receipt after completion.
    """
    audit = paired.checked_external(payload_audit)
    data.validate_audit(dataset,audit)
    require(len(data.rows_for(dataset,'dev'))==3,'Fixed limited-dev3 required')
    identity_before = fresh_v5_identity(student, tensor_manifest_path)
    observation = evaluate_dev(student=student, provider=provider, lpips_fn=lpips_fn,
                               dataset=dataset, selection_plan=selection_plan, step=0)
    identity_after = fresh_v5_identity(student, tensor_manifest_path)
    require(identity_before == identity_after, 'Fresh v5 changed during step0')
    recipe = core.frozen_recipe(Path(__file__).parent)
    metric_protocol = {'registry_sha256': dataset['descriptor']['sha256'],
                       'selection_plan_sha256': canonical_sha(selection_plan),
                       'dev_order': [r['sample_id'] for r in data.rows_for(dataset, 'dev')]}
    metrics = paired.selection_metrics(observation, protocol=metric_protocol, step=0, recipe=recipe)
    require(recipe.hr_guard(metrics, metrics) is True, 'Step0 HR guard must accept its valid baseline')
    return {'schema': 'h3-production67-limiteddev-common-step0-v1',
            'status': 'actual_sparse_step0_observed_not_formal_admission',
            'initializer_checkpoint_sha256': core.V5_SHA, **metric_protocol,
            'selection_evaluator_sha256': file_sha(__file__), 'metrics': metrics,
            'fresh_v5_identity': identity_before, 'observation': observation,
            'payload_audit': copy.deepcopy(payload_audit), 'best0_allowed': True,
            'formal_training_allowed': False, 'full226_dev_quality_gate_passed': False,
            'test_access': False, 'optimizer_updates': 0}
