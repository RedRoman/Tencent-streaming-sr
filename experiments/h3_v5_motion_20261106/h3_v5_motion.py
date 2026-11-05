"""Additional H3 v5 motion losses; no decoder, model, or flow recipe changes.

RGB losses operate on the exact unclipped float RGB region selected by the
existing PixelSupervisor. Only pairs adjacent in actual RGB output time are
eligible. A detached target change mask supervises both
endpoints of each eligible pair; a frame occurring in two pairs is counted
twice, explicitly defining a pair-weighted ROI objective. The differential
loss covers all pixels of eligible pairs, without optical flow.

The optional latent target-residual eWarp term retains the target's genuine
warp residual instead of encouraging every predicted warp residual to zero.
It uses the existing backward-flow sampler and in-bounds/valid-pair masks.
There is no occlusion/confidence mask and no claim that the flow is reliable.
All functions return UNWEIGHTED losses. The trainer owns their weights.
"""
from __future__ import annotations

import math

import torch

from h3_hybrid_pixel import PixelSupervisor
from ultraflash_ewarp import warp_previous


def _sequence_pair(pred, target, *, rgb=False):
    if (pred.ndim != 5 or pred.shape != target.shape or
            min(pred.shape) < 1 or pred.device != target.device or
            not pred.is_floating_point() or not target.is_floating_point()):
        raise ValueError('Expected equal floating [B,C,T,H,W] on one device')
    if rgb and pred.shape[1] != 3:
        raise ValueError('RGB motion losses require three channels')


def _time_indices(values, count, label):
    if (not isinstance(values, (list, tuple)) or len(values) != count or
            any(type(v) is not int or v < 0 for v in values) or
            any(a >= b for a, b in zip(values, values[1:]))):
        raise ValueError(f'{label} must be strictly increasing nonnegative integer indices')
    return values


def rgb_motion_losses(pred, target, rgb_frame_indices, raw_frame_indices=None,
                      *, change_threshold=0.05):
    """Return motion_l1, delta_l1 and detached scalar coverage diagnostics.

    pred/target: unclipped float RGB [B,3,K,H,W]. Indices describe the K
    selected frames, not positions within the selected array. Raw indices,
    when supplied, are checked for consistent length/order, not adjacency:
    H3 raw19->23 maps to consecutive RGB118->119 in the last block after its
    prescribed token-drop. The production supervisor supplies both mappings.
    motion_roi_fraction counts high-change pixels among eligible pair pixels.
    Empty pair sets/ROIs return zero connected to pred (and never target).
    """
    _sequence_pair(pred, target, rgb=True)
    if not math.isfinite(change_threshold) or change_threshold <= 0:
        raise ValueError('change_threshold must be finite and positive')
    count = pred.shape[2]
    rgb_ids = _time_indices(rgb_frame_indices, count, 'rgb_frame_indices')
    if raw_frame_indices is not None:
        _time_indices(raw_frame_indices, count, 'raw_frame_indices')
    left = [i for i in range(count-1) if rgb_ids[i+1]-rgb_ids[i] == 1]
    zero = pred.float().sum()*0.
    stats = {
        'motion_roi_fraction': zero.detach(),
        'adjacent_pair_count': pred.new_tensor(len(left), dtype=torch.float32),
        'adjacent_pair_fraction': pred.new_tensor(
            len(left)/max(count-1, 1), dtype=torch.float32),
    }
    if not left:
        return {'motion_l1': zero, 'delta_l1': zero, **stats}
    right = [i+1 for i in left]
    pixels, truth = pred.float(), target.detach().float()
    previous, current = pixels[:, :, left], pixels[:, :, right]
    target_previous, target_current = truth[:, :, left], truth[:, :, right]
    target_delta = target_current-target_previous
    mask = target_delta.abs().mean(dim=1, keepdim=True) >= change_threshold
    # Both pair endpoints use the SAME target-change region. Normalize by
    # 2 endpoints * 3 channels * selected pixels; never dilute by static area.
    endpoint_error = ((previous-target_previous).abs() +
                      (current-target_current).abs())
    denominator = mask.sum().to(endpoint_error.dtype)*2*pred.shape[1]
    motion_l1 = (endpoint_error*mask).sum()/denominator.clamp_min(1)
    delta_l1 = ((current-previous)-target_delta).abs().mean()
    stats['motion_roi_fraction'] = mask.float().mean().detach()
    return {'motion_l1': motion_l1, 'delta_l1': delta_l1, **stats}


def target_residual_ewarp(pred, target, flow, valid_pairs):
    """Unweighted residual eWarp plus the unchanged legacy raw metric.

    pred/target are [B,C,T,H,W]; backward flow is [B,T-1,2,H,W] and valid_pairs
    is bool [B,T-1]. The batch dimension may be omitted for B=1. Target and
    flow are detached. Like ultraflash_ewarp.losses, reduction is a per-pair
    masked channel/pixel mean, then an equal mean over nonempty pairs.
    valid_pairs is caller-supplied real-frame validity, NOT occlusion masking.
    """
    _sequence_pair(pred, target)
    batch, channels, frames, height, width = pred.shape
    if flow.ndim == 4:
        flow = flow[None]
    if valid_pairs.ndim == 1:
        valid_pairs = valid_pairs[None]
    if (flow.shape != (batch, frames-1, 2, height, width) or
            valid_pairs.shape != (batch, frames-1) or
            flow.device != pred.device or valid_pairs.device != pred.device or
            not flow.is_floating_point() or valid_pairs.dtype != torch.bool):
        raise ValueError('Invalid backward flow or boolean valid-pair geometry/device')
    zero = pred.float().sum()*0.
    if frames == 1:
        return {'residual_ewarp_l1': zero, 'legacy_ewarp_l1': zero,
                'target_ewarp_l1': zero.detach(),
                'valid_warp_fraction': zero.detach()}

    def pair_tensor(value):
        return value.permute(0, 2, 1, 3, 4).reshape(
            batch*(frames-1), channels, height, width)

    pixels = pred.float()
    previous, current = pair_tensor(pixels[:, :, :-1]), pair_tensor(pixels[:, :, 1:])
    fixed_flow = flow.detach().reshape(batch*(frames-1), 2, height, width)
    warped, inbounds = warp_previous(previous, fixed_flow)
    with torch.no_grad():
        truth = target.detach().float()
        target_previous = pair_tensor(truth[:, :, :-1])
        target_current = pair_tensor(truth[:, :, 1:])
        target_warped, _ = warp_previous(target_previous, fixed_flow)
        target_residual = target_current-target_warped
        mask = inbounds & valid_pairs.detach().reshape(-1, 1, 1, 1)
        denominator = mask.float().sum((1, 2, 3))*channels
        has_pixels = denominator > 0

    def masked_mean(value):
        per_pair = (value.abs()*mask).sum((1, 2, 3))/denominator.clamp_min(1)
        return per_pair[has_pixels].mean() if bool(has_pixels.any()) else zero

    residual = current-warped
    return {
        'residual_ewarp_l1': masked_mean(residual-target_residual),
        'legacy_ewarp_l1': masked_mean(residual),
        'target_ewarp_l1': masked_mean(target_residual).detach(),
        'valid_warp_fraction': (mask.float().sum()/
            (valid_pairs.float().sum()*height*width).clamp_min(1)).detach(),
    }


class MotionPixelSupervisor(PixelSupervisor):
    """Same frozen decoder/LPIPS/crops with additional target-motion terms.

    Call PixelSupervisor.audit on a separate base supervisor using the same
    frozen modules: the inherited audit assumes exactly two gradient-bearing
    losses, whereas this losses() also returns detached coverage diagnostics.
    Additional-loss autograd behavior is covered by test_h3_v5_motion.py and
    must additionally be checked in the caller's actual-model preflight.
    """

    def __init__(self, *args, change_threshold=0.05, **kwargs):
        if not math.isfinite(change_threshold) or change_threshold <= 0:
            raise ValueError('change_threshold must be finite and positive')
        self.change_threshold = float(change_threshold)
        super().__init__(*args, **kwargs)

    def losses(self, pred, reference, spec):
        self._validate_decoder()
        pixels = self._decode_region(pred, spec)
        with torch.no_grad():
            target = self._decode_region(reference.detach(), spec)
        # Deliberately preserve PixelSupervisor's original L1 and LPIPS
        # operations, without decoding a second time or changing frame samples.
        pixel_l1 = (pixels-target).abs().mean()

        def images(rgb):
            selected = rgb[:, :, spec['lpips_frame_indices']]
            return (selected.clamp(0., 1.)*2.-1.).permute(0, 2, 1, 3, 4).flatten(0, 1)

        perceptual = self.lpips_fn(images(pixels), images(target)).mean()
        terms = {'pixel_l1': pixel_l1, 'lpips': perceptual,
                 **rgb_motion_losses(pixels, target, spec['rgb_frame_indices'],
                     spec['raw_frame_indices'], change_threshold=self.change_threshold)}
        if not all(bool(torch.isfinite(value)) for value in terms.values()):
            raise FloatingPointError('Nonfinite v5 pixel loss or diagnostic')
        return terms

    def description(self):
        return {**super().description(),
            'motion_change_threshold': self.change_threshold,
            'motion_mask': 'detached decoded target abs temporal change, mean RGB >= threshold',
            'motion_adjacency': 'Actual RGB index delta==1; raw indices validated for order only',
            'motion_raw_drop_transition': 'Raw19->23 / RGB118->119 is a real adjacent pair and is included',
            'motion_l1_domain': 'Unclipped float RGB; both endpoints in each pair ROI; pair-weighted',
            'delta_l1_domain': 'Unclipped RGB temporal-difference residual; all eligible pair pixels',
            'motion_roi_fraction_domain': 'High-change pixels / all eligible pair pixels',
            'motion_weights': 'Unweighted here; controlled by trainer',
            'motion_mask_uses_optical_flow': False}
