"""Minimal production67 candidate training core; no launcher or READY claim.

The new cache loader and mandatory explicit target replace the historical
37-token/exact2x assumptions. Fixed mathematical loss/selection functions are
compiled, unchanged, from SHA-pinned historical sources; old data loaders and
PixelSupervisor are never imported. A separately validated frozen decoder
region provider is injected. This module does not encode, generate pairs,
choose windows, change precision, or begin a training run on import.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import math
import random
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


SOURCE_PINS = {
    'ultraflash_ewarp.py': ('a9cc9147b2ff46ef2b62d3d5c0f6bb0567ad2b9eaa433a90a9b402ebb03db33d',
                          ('warp_previous', 'losses')),
    'h3_v5_motion.py': ('855d262cd46e4e74e2336333599f980c640c4cc309c5c05bf4fffd33a3206faa',
                        ('_sequence_pair', '_time_indices', 'rgb_motion_losses', 'target_residual_ewarp')),
    'h3_teacher_distill.py': ('4c38c7b96ebb754b3987f8d8f30d373c6352ec87e0c45aa524500077a7cd58f1',
        ('paired_schedule', 'shared_roi_rgb_losses', 'training_objective', 'selection_fields', 'hr_guard')),
}
LOADER_SHA = '5d3ea3d6e8d59896e5c2b664addf3a5bba5d4700efa525a0335125973124602b'
WRAPPER_SHA = '8c5bcbd1702a252266c68cbdf131e40fa7b56366deaeef17c53a9b84dbd5b3c6'
V5_SHA = 'c7aaaa396d877a84c3fed2a5536189b7ed5e61226a82ef1a9e6be1e831f7f914'
LR_SHAPE = (1, 24, 67, 44, 80)
HR_SHAPE = (1, 24, 67, 68, 120)
TARGET_THW = (67, 68, 120)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frozen_recipe(source_root):
    """Reuse only named function ASTs from exact old files, without imports.

    Function bodies, defaults and signatures are unchanged. The historical
    losses() default36 is NEVER used: latent_terms_all explicitly supplies T.
    No source files are written. This is dependency isolation, not a new loss.
    """
    namespace = {'torch': torch, 'F': F, 'copy': copy, 'math': math, 'random': random}
    provenance = {}
    for filename, (expected, names) in SOURCE_PINS.items():
        path = Path(source_root) / filename
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected, 'Frozen loss source changed: ' + filename)
        tree = ast.parse(raw, filename=str(path))
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        require(tuple(n.name for n in selected) == names, 'Frozen function selection changed')
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
        provenance[filename] = {'sha256': expected, 'functions': list(names)}
    namespace['legacy_losses'] = namespace['losses']
    return SimpleNamespace(**namespace, source_provenance=provenance)


def strict_loader():
    path = Path(__file__).with_name('h3_production67_pair_loader_v1.py')
    require(file_sha(path) == LOADER_SHA, 'New strict pair loader changed')
    spec = importlib.util.spec_from_file_location('h3_prod67_step_pair_loader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Production67ScheduleSampler:
    """Same old rng.choice plus sorted rng.sample law, new exact geometry.

    Canonical region specs are never mutated to hide a geometry difference.
    Random LPIPS positions live beside the canonical region in the paired
    schedule. The historical paired_schedule() is reused without alteration.
    """
    def __init__(self):
        loader = strict_loader()
        geometry = loader.geometry_module()
        path = Path(__file__).with_name('h3_production67_pixel_specs_v1.py')
        require(file_sha(path) == '50ea70c63a31ce14a3aec11b5072d369bcbd2d6b739a08b049e08cd40e2edfd0',
                'Frozen pixel catalog changed')
        spec = importlib.util.spec_from_file_location('h3_prod67_step_specs', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.catalog = module.PixelSpecCatalog(geometry.production67(), temporal_scope='nonblend_only')
        self.full_specs = self.catalog.full_specs()
        self.specs = self.catalog.tile_specs()

    def sample_window(self, rng, *, full=False):
        region = copy.deepcopy(rng.choice(self.full_specs if full else self.specs))
        count = len(region['rgb_frame_indices'])
        positions = sorted(rng.sample(range(count), min(4, count)))
        return {'region': region, 'lpips_packed_positions': positions}


def load_training_example(root, *, pair_path, pair_sha256, teacher_path,
                          teacher_sha256, contract):
    """Read the new bound cache schema; never silently convert old .pt caches.

    NativeHR uint8 mmap remains available in pair_bundle for separate native
    RGB metrics, but is not copied into Torch or confused with decodedHR.
    Device transfer is explicit in move_example(). This is not data acceptance.
    """
    loader = strict_loader()
    loader.validate_contract(contract)
    pair = loader.load_pair(root, pair_path, pair_sha256, contract)
    teacher = loader.load_teacher(root, teacher_path, teacher_sha256, pair, contract)
    require(pair['record']['split'] in ('train', 'dev'), 'Heldout data requires later gated evaluation')
    values = loader.torch_arrays(pair)
    result = {
        'draft': values['draft'].unsqueeze(0),
        'reference': values['reference_hr'].unsqueeze(0),
        'teacher': loader.torch_teacher(teacher).unsqueeze(0),
        'flow': values['flow'], 'valid': values['flow_valid'],
        'pair_bundle': pair, 'teacher_bundle': teacher,
        'contract': copy.deepcopy(contract),
        'scope': 'candidate_checked_cache_not_training_READY',
    }
    validate_example_tensors(result)
    return result


def move_example(example, *, device):
    """No cast or normalization. CPU source caches and source mmp stay CPU."""
    validate_example_tensors(example)
    result = dict(example)
    for name in ('draft', 'reference', 'teacher', 'flow', 'valid'):
        result[name] = example[name].to(device=device).detach()
    validate_example_tensors(result)
    return result


def validate_example_tensors(example):
    shapes = {'draft': LR_SHAPE, 'reference': HR_SHAPE, 'teacher': HR_SHAPE,
              'flow': (66, 2, 68, 120), 'valid': (66,)}
    device = example['draft'].device
    for name, shape in shapes.items():
        value = example[name]
        require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape,
                'Wrong production67 shape: ' + name)
        require(value.dtype == (torch.bool if name == 'valid' else torch.float32),
                'Storage precision changed: ' + name)
        require(value.device == device and not value.requires_grad, 'Targets/input must be detached on one device')
        require(bool(torch.isfinite(value).all()), 'Nonfinite input: ' + name)
    require(bool(example['valid'].all()), 'All66 true latent-adjacent pairs are valid')


def explicit_forward67(student, draft):
    """No fallback to student(draft) or implicit spatial_scale=2."""
    require(tuple(draft.shape) == LR_SHAPE and draft.dtype == torch.float32,
            'Explicit67 forward requires exactFP32 LR shape')
    require(getattr(student, 'source_checkpoint_sha256', None) == V5_SHA,
            'Both arms must originate from the selected v5 step1800')
    require(getattr(student, 'model_config', {}).get('kind') == 'h3_causal_conv3d_explicit_target_v1',
            'Require reviewed explicit-target wrapper')
    result = student.forward_full(draft, target_size=TARGET_THW)
    require(tuple(result.shape) == HR_SHAPE and result.dtype == torch.float32 and
            result.device == draft.device, 'Explicit target output shape/dtype/device changed')
    return result


def latent_terms_all(pred, reference, teacher, flow, valid, recipe):
    """Shape-generic mathematical kernel; production entry enforces67/66.

    This tiny-testable kernel includes the final real token and every valid
    adjacent flow pair. It has no resizing, implicit2x, or historical slice36.
    """
    require(pred.ndim == 5 and tuple(reference.shape) == tuple(pred.shape)
            and tuple(teacher.shape) == tuple(pred.shape), 'Latent target shape mismatch')
    require(pred.shape[2] >= 2 and reference.device == teacher.device == pred.device,
            'Need at least2 latent tokens on one device')
    reference, teacher = reference.detach(), teacher.detach()
    old = recipe.legacy_losses(pred, reference, flow.detach(), valid.detach(), valid_frames=pred.shape[2])
    result = {'hr_latent_mse': old['latent_mse'], 'legacy_ewarp_l1': old['ewarp_l1'],
              'teacher_mse': (pred.float()-teacher.float()).square().mean()}
    for prefix, target in (('hr', reference), ('teacher', teacher)):
        values = recipe.target_residual_ewarp(pred, target, flow, valid)
        result[prefix+'_residual_ewarp_l1'] = values['residual_ewarp_l1']
        result[prefix+'_target_ewarp_l1'] = values['target_ewarp_l1']
    return result


def pixel_terms_visible(pred, reference, teacher, *, rgb_frame_indices,
                        lpips_packed_positions, lpips_fn, recipe):
    """Generic RGB loss kernel after explicit visible crop/temporal assembly.

    Decoded targets are detached; both arms use exactly the decodedHR mask.
    Native source RGB is a separate evaluation domain, never mislabeled here.
    LPIPS positions must come from the frozen paired step schedule.
    """
    require(pred.ndim == 5 and pred.shape[1] == 3 and
            tuple(reference.shape) == tuple(pred.shape) == tuple(teacher.shape), 'RGB region mismatch')
    require(pred.device == reference.device == teacher.device, 'RGB device mismatch')
    recipe._time_indices(rgb_frame_indices, pred.shape[2], 'rgb_frame_indices')
    require(type(lpips_packed_positions) is list and len(lpips_packed_positions) > 0 and
            all(type(x) is int and 0 <= x < pred.shape[2] for x in lpips_packed_positions) and
            lpips_packed_positions == sorted(set(lpips_packed_positions)), 'Bad LPIPS packed positions')
    targets = {'hr': reference.detach(), 'teacher': teacher.detach()}
    def images(rgb):
        selected = rgb[:, :, lpips_packed_positions]
        return (selected.clamp(0., 1.)*2.-1.).permute(0, 2, 1, 3, 4).flatten(0, 1)
    image_pred = images(pred)
    values = {}
    for prefix, target in targets.items():
        current = dict(pixel_l1=(pred-target).abs().mean(),
                       lpips=lpips_fn(image_pred, images(target)).mean(),
                       **recipe.shared_roi_rgb_losses(pred, target, targets['hr'], rgb_frame_indices))
        values.update({prefix+'_'+key: value for key, value in current.items()})
    require(all(bool(torch.isfinite(v)) for v in values.values()), 'Nonfinite RGB loss or metric')
    return values


def loss_for_example(student, example, spec, *, lpips_packed_positions, temporal_scope, decode_region,
                     lpips_fn, recipe, arm):
    """Production-geometry differentiable objective, without optimizer update.

    decode_region(latent,spec) -> unclipped floatRGB [1,3,K,H,W] in the
    canonical spec's visible1080 coordinates. The caller supplies a separately
    tested frozen VAE provider; a callback is not evidence of decoder parity.
    Existing objective is nonblend_only. Selecting assembled_all is explicitly
    rejected here until a separate training-objective protocol is adopted.
    """
    require(temporal_scope == 'nonblend_only', 'New blend supervision is not part of this candidate')
    require(arm in ('control', 'teacher'), 'Unknown arm')
    require(file_sha(inspect.getfile(type(student))) == WRAPPER_SHA,
            'Student wrapper implementation differs from the reviewed source')
    loader = strict_loader()
    loader.validate_contract(example['contract'])
    validate_example_tensors(example)
    checked = loader.checked_pixel_spec(spec, temporal_scope=temporal_scope)
    pred = explicit_forward67(student, example['draft'])
    terms = latent_terms_all(pred, example['reference'], example['teacher'],
                             example['flow'], example['valid'], recipe)
    rgb_pred = decode_region(pred, checked)
    with torch.no_grad():
        rgb_hr = decode_region(example['reference'].detach(), checked)
        rgb_teacher = decode_region(example['teacher'].detach(), checked)
    y0, y1, x0, x1 = checked['spatial']['visible_global_box_yx']
    expected = (1, 3, len(checked['rgb_frame_indices']), y1-y0, x1-x0)
    for value in (rgb_pred, rgb_hr, rgb_teacher):
        require(tuple(value.shape) == expected and value.device == pred.device and
                value.dtype == torch.float32, 'Decoder must return exact FP32 visible region')
    require(rgb_pred.requires_grad, 'Prediction decoder path has no gradient')
    pixels = pixel_terms_visible(rgb_pred, rgb_hr, rgb_teacher,
        rgb_frame_indices=checked['rgb_frame_indices'],
        lpips_packed_positions=lpips_packed_positions, lpips_fn=lpips_fn, recipe=recipe)
    loss = recipe.training_objective(terms, pixels, arm=arm, control_recipe='motion')
    require(loss.ndim == 0 and bool(torch.isfinite(loss)) and loss.requires_grad,
            'Training loss must be finite differentiable scalar')
    return loss, terms, pixels


def fresh_adamw(parameters, *, lr):
    require(type(lr) in (int, float) and math.isfinite(lr) and lr > 0, 'Invalid initial LR')
    parameters = tuple(parameters)
    require(parameters and all(p.requires_grad and p.dtype == torch.float32 for p in parameters),
            'Expected trainable FP32 student parameters')
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=.01, foreach=False)


def apply_candidate_update(loss, parameters, optimizer):
    """Small optimizer/gradient interface, also used by actual tinyCPU tests.

    Caller computes one loss then explicitly requests one update. There is no
    training loop, implicit resume, data acquisition, gate bypass, or launch.
    The candidate does not claim actual345M/NPU recovery validation.
    """
    parameters = tuple(parameters)
    require(parameters and all(p.requires_grad for p in parameters), 'Empty/frozen student parameters')
    require(isinstance(optimizer, torch.optim.AdamW), 'Expected fixed AdamW recipe')
    require({id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in parameters},
            'Optimizer must contain only the complete student parameter set')
    for group in optimizer.param_groups:
        require(group['weight_decay'] == .01 and group['betas'] == (.9, .999) and
                group['eps'] == 1e-8 and group['foreach'] is False, 'Fixed AdamW recipe changed')
    require(loss.ndim == 0 and bool(torch.isfinite(loss)) and loss.requires_grad,
            'Invalid optimizer objective')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    require(all(p.grad is not None for p in parameters), 'Some student parameters have no gradient')
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., foreach=False, error_if_nonfinite=True)
    require(float(norm) > 0, 'Zero student gradient')
    optimizer.step()
    return {'gradient_norm_before_clip': float(norm), 'optimizer_updates': 1,
            'formal_training_preflight_passed': False}


def description():
    return {'schema': 'h3-production67-train-step-candidate-v1',
        'input': list(LR_SHAPE), 'target': list(HR_SHAPE), 'target_size_explicit': list(TARGET_THW),
        'source_frames': 226, 'fps': 24, 'raw_hw': [1088, 1920], 'visible_crop': [4, 1084, 0, 1920],
        'latent_supervision_tokens': 67, 'flow_pairs': 66, 'temporal_scope': 'nonblend_only',
        'direct_rgb_frames': 166, 'excluded_blend_frames': 60,
        'teacher_hr_gradient_mixed': False, 'ROI_source': 'shared_detached_decodedHR',
        'nativeHR_target_label': 'native source RGB, separate evaluation domain',
        'data_loader_sha256': LOADER_SHA, 'explicit_wrapper_sha256': WRAPPER_SHA,
        'loss_source_pins': SOURCE_PINS,
        'LPIPS_schedule': 'old rng.choice then sorted rng.sample law; positions stored beside unmutated canonical region; paired_schedule reused',
        'HR_guard': 'unchanged selection_fields/hr_guard; not a replacement for full226 nativeHR/dev gates',
        'remaining': ['bound real generation contract/data acceptance', 'frozen decoder provider actual parity/gradient',
                      'new geometry paired schedule/protocol freeze', 'actual paired8 and345M optimizer/RNG restore',
                      'full226 evaluation and gate integration'],
        'formal_training_allowed': False, 'real_optimizer_updates': 0}
