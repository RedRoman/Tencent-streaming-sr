"""Teacher-target losses and contracts for a new, unstarted H3 experiment.

The student is unchanged. Teacher is the original bidirectional upscaler, not
the H3 diffusion refiner. Both targets are detached and decoded with the same
frozen H3 decoder. Native crop/blend exclusions remain unchanged so changing
the target does not silently change pixel coverage. No optical-flow confidence
or occlusion correctness is assumed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path

import torch

from data_utils import tensor_load
from h3_hybrid_pixel import PixelSupervisor
from h3_v5_motion import rgb_motion_losses, target_residual_ewarp
from train_upscaler_trial import sha256, to_cpu
from ultraflash_ewarp import FLOW_VERSION, losses as legacy_losses


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def capture_optimizer_rng(optimizer, device):
    """Shared production/test checkpoint mechanism; schedules are immutable."""
    device = torch.device(device)
    return dict(optimizer=to_cpu(optimizer.state_dict()), torch_rng=torch.get_rng_state(),
                device_rng=torch.npu.get_rng_state(device) if device.type == 'npu' else None)


def restore_optimizer_rng(optimizer, saved, device):
    device = torch.device(device)
    optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng'])
    if device.type == 'npu':
        if saved.get('device_rng') is None:
            raise ValueError('NPU resume requires the saved selected-device RNG state')
        torch.npu.set_rng_state(saved['device_rng'],device)


def paired_schedule(sample_ids, supervisor, *, steps, seed, lr, min_lr,
                    warmup, full_every):
    """Materialize the actual identical sample/region/LPIPS/LR schedule."""
    if (not sample_ids or len(set(sample_ids)) != len(sample_ids) or
            min(steps, full_every) < 1 or warmup < 0 or
            not 0 <= min_lr <= lr or not math.isfinite(lr)):
        raise ValueError('Invalid paired schedule')
    order_rng, window_rng = random.Random(seed), random.Random(seed+1)
    indices = []
    while len(indices) < steps:
        order = list(range(len(sample_ids)))
        order_rng.shuffle(order)
        indices.extend(order)
    result = []
    for step, index in enumerate(indices[:steps]):
        warm = min(1., (step+1)/max(1, warmup))
        ratio = max(0, step-warmup)/max(1, steps-warmup-1)
        rate = warm*(min_lr+(lr-min_lr)*(1+math.cos(math.pi*ratio))/2)
        result.append(dict(step=step+1, sample_id=sample_ids[index], lr=rate,
            spec=supervisor.sample_window(window_rng, full=step % full_every == 0)))
    return result


def fixed_split_identity(manifest):
    """Compare IDs/source identity and order, independent of resolution caches."""
    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    result = []
    for row in rows:
        source = row.get('source_video_sha256')
        if not source:
            raise ValueError('Split manifest must bind original source_video_sha256')
        result.append(dict(sample_id=row['sample_id'], split=row['split'],
                           source_video_sha256=source))
    if len({row['sample_id'] for row in result}) != len(result):
        raise ValueError('Duplicate split identity')
    return result


def load_spatially_configurable_pairs(manifest):
    """Same 120-frame/37-latent contract; spatial size comes from paired data.

    No resize, temporal sampling, normalization, VAE encode or cache mutation
    happens here. New sizes require finalized newly encoded pairs and flows.
    """
    manifest = Path(manifest)
    result, geometry = [], None
    for row in [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]:
        path = Path(row['path'])
        path = path if path.is_absolute() else manifest.parent/path
        flow_path = manifest.parent/'flow'/path.name
        pair_sha, flow_sha = sha256(path), sha256(flow_path)
        pair, flow = tensor_load(path), tensor_load(flow_path)
        draft, reference = pair['draft'], pair['reference']
        meta = pair['metadata']
        shape = tuple(reference.shape)
        if (draft.ndim != 4 or draft.shape[:2] != (24, 37) or
                shape != (24, 37, 2*draft.shape[-2], 2*draft.shape[-1]) or
                min(draft.shape[-2:]) < 1):
            raise ValueError('Expected native H3 24x37 latents and exact spatial 2x')
        if draft.dtype != torch.float32 or reference.dtype != torch.float32 or flow['flow'].dtype != torch.float32:
            raise ValueError('Pair and flow storage must be FP32 before any conversion')
        expected_metadata = dict(source_frames=120,source_frames_encoded=120,
            output_frames=120,decoded_frames=120,latent_frames=37,fps=24.,
            posterior='sample',posterior_seed=42,
            normalization='(raw-vae.latents_mean)/vae.latents_std',
            external_time_padding=False,external_slowdown=False,
            target_pixels=[shape[-1]*16,shape[-2]*16],
            draft_pixels=[draft.shape[-1]*16,draft.shape[-2]*16])
        if any(meta.get(key) != value for key,value in expected_metadata.items()):
            raise ValueError('Pair metadata changed: FP32 seed42,120 real frames at24fps,native normalization required')
        if geometry is not None and shape != geometry:
            raise ValueError('A run must use one predeclared geometry')
        geometry = shape
        if (row.get('sha256') != pair_sha or flow['pair_sha256'] != pair_sha or
                flow['metadata']['version'] != FLOW_VERSION or
                meta['split'] != row['split'] or
                meta.get('sample_id', meta.get('id')) != row['sample_id'] or
                meta['source_video_sha256'] != row['source_video_sha256']):
            raise ValueError('Pair/flow/source/split fingerprint mismatch')
        if (flow['flow'].shape != (36, 2, *shape[-2:]) or
                flow['valid_pairs'].dtype != torch.bool or
                flow['valid_pairs'].shape != (36,) or
                not bool(flow['valid_pairs'][:35].all()) or
                bool(flow['valid_pairs'][35])):
            raise ValueError('Wrong flow geometry or real-frame validity')
        if (pair['source_pixels'].shape != (120, shape[-2]*16, shape[-1]*16, 3) or
                pair['source_pixels'].dtype != torch.uint8):
            raise ValueError('Expected 120 full-resolution uint8 source frames')
        if not all(bool(torch.isfinite(v).all()) for v in (draft, reference, flow['flow'])):
            raise ValueError('Nonfinite pair/flow')
        result.append(dict(sample_id=row['sample_id'], split=row['split'],
            draft=draft.float(), reference=reference.float(), flow=flow['flow'],
            valid_pairs=flow['valid_pairs'], pair_sha256=pair_sha,
            source_sha256=meta['source_video_sha256'], pair_path=str(path),
            flow_path=str(flow_path), flow_sha256=flow_sha,
            encoding_metadata=expected_metadata))
    if not result:
        raise ValueError('Empty training/dev split')
    return result


class TeacherPixelSupervisor(PixelSupervisor):
    """Decode prediction once and two frozen targets, using identical specs.

    HR measurements are diagnostics/selection guards for the teacher arm.
    Neither decoded target receives gradients. Use a separate base supervisor
    for PixelSupervisor.audit, which expects only its two original losses.
    """
    def losses(self, pred, reference, teacher, spec):
        self._validate_decoder()
        pixels = self._decode_region(pred, spec)
        with torch.no_grad():
            targets = {'hr': self._decode_region(reference.detach(), spec),
                       'teacher': self._decode_region(teacher.detach(), spec)}

        def images(rgb):
            selected = rgb[:, :, spec['lpips_frame_indices']]
            return (selected.clamp(0., 1.)*2.-1.).permute(0, 2, 1, 3, 4).flatten(0, 1)

        image_pred = images(pixels)
        values = {}
        for prefix, target in targets.items():
            current = dict(pixel_l1=(pixels-target).abs().mean(),
                lpips=self.lpips_fn(image_pred, images(target)).mean(),
                **shared_roi_rgb_losses(pixels, target, targets['hr'], spec['rgb_frame_indices'],
                                        spec['raw_frame_indices']))
            values.update({prefix+'_'+key: value for key, value in current.items()})
        if not all(bool(torch.isfinite(value)) for value in values.values()):
            raise FloatingPointError('Nonfinite dual-target pixel metric')
        return values

    def description(self):
        return {**super().description(),
                'targets': ['original bidirectional upscaler decoded RGB', 'HR latent decoded RGB'],
                'teacher_target_detached': True, 'hr_target_detached': True,
                'motion_threshold': .05,
                'motion_mask': 'Both objectives use the same detached HR target change mask; mean RGB change >= .05',
                'new_blend_supervision': False}


def shared_roi_rgb_losses(pred, target, hr, rgb_frame_indices, raw_frame_indices=None):
    """Keep HR-defined ROI fixed while changing only the supervised target.

    Delta covers all real adjacent pairs. ROI covers both pair endpoints;
    teacher mask never substitutes for the predeclared HR motion regions.
    """
    values = rgb_motion_losses(pred, target, rgb_frame_indices, raw_frame_indices,
                               change_threshold=.05)
    if hr.shape != pred.shape or hr.device != pred.device or not hr.is_floating_point():
        raise ValueError('HR mask source must match floating prediction geometry/device')
    left = [i for i in range(len(rgb_frame_indices)-1)
            if rgb_frame_indices[i+1]-rgb_frame_indices[i] == 1]
    if not left:
        return values
    right = [i+1 for i in left]
    hr, target, pred = hr.detach().float(), target.detach().float(), pred.float()
    mask = (hr[:, :, right]-hr[:, :, left]).abs().mean(dim=1, keepdim=True) >= .05
    error = ((pred[:, :, left]-target[:, :, left]).abs()+
             (pred[:, :, right]-target[:, :, right]).abs())
    values['motion_l1'] = (error*mask).sum()/(2*pred.shape[1]*mask.sum()).clamp_min(1)
    values['motion_roi_fraction'] = mask.float().mean().detach()
    return values


def latent_terms(pred, reference, teacher, flow, valid):
    reference, teacher = reference.detach(), teacher.detach()
    old = legacy_losses(pred, reference, flow.detach(), valid.detach(), valid_frames=36)
    result = dict(hr_latent_mse=old['latent_mse'], legacy_ewarp_l1=old['ewarp_l1'],
                  teacher_mse=(pred[:, :, :36]-teacher[:, :, :36]).square().mean())
    for prefix, target in (('hr', reference), ('teacher', teacher)):
        values = target_residual_ewarp(pred, target, flow, valid)
        result[prefix+'_residual_ewarp_l1'] = values['residual_ewarp_l1']
        result[prefix+'_target_ewarp_l1'] = values['target_ewarp_l1']
    return result


def training_objective(terms, pixels, *, arm, control_recipe='motion'):
    """Explicit fixed coefficients; no HR gradient term in teacher objective.

    1.25 teacher MSE preserves the old total latent coefficient 1+.25.
    RGB/LPIPS coefficients and motion-family coefficients retain previous
    scale; these are experiment choices, not proven optimal values.
    """
    if arm not in ('control', 'teacher') or control_recipe not in ('v5', 'motion'):
        raise ValueError('Unknown predeclared arm/control recipe')
    if arm == 'teacher':
        return (1.25*terms['teacher_mse']+.1*terms['teacher_residual_ewarp_l1']+
                pixels['teacher_pixel_l1']+.2*pixels['teacher_lpips']+
                .5*pixels['teacher_motion_l1']+.2*pixels['teacher_delta_l1'])
    pixel = pixels['hr_pixel_l1']+.2*pixels['hr_lpips']
    if control_recipe == 'v5':
        return terms['hr_latent_mse']+.25*terms['teacher_mse']+.1*terms['legacy_ewarp_l1']+pixel
    return (terms['hr_latent_mse']+.25*terms['teacher_mse']+
            .1*terms['hr_residual_ewarp_l1']+pixel+
            .5*pixels['hr_motion_l1']+.2*pixels['hr_delta_l1'])


def selection_fields(mean):
    result = copy.deepcopy(mean)
    result['hr_global_score'] = result['hr_pixel_l1']+.2*result['hr_lpips']
    result['teacher_global_score'] = result['teacher_pixel_l1']+.2*result['teacher_lpips']
    result['selection_score'] = (result['teacher_global_score']+
        .5*result['teacher_motion_l1']+.2*result['teacher_delta_l1'])
    return result


def hr_guard(candidate, baseline):
    """Fixed step0 HR guard; sparse training-dev regions are not full120 QA."""
    keys = ('hr_global_score', 'hr_pixel_l1', 'hr_lpips')
    if not all(math.isfinite(float(value[key])) and float(value[key]) >= 0
               for value in (candidate, baseline) for key in keys):
        return False
    return all(candidate[key] <= baseline[key]*factor
               for key, factor in zip(keys, (1.01, 1.02, 1.02)))
