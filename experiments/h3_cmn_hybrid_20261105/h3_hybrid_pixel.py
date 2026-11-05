"""Differentiable, deployment-aligned H3 pixel supervision on sparse regions.

The released H3 decoder reads seven latent frames at starts 0,5,10,... . Its
28 raw output frames become 17 frames plus a five-frame overlap, after dropping
three leading frames in each group of twenty. Later blocks crossfade their
first five frames with the preceding block. We exclude those five frames.
For 37 latents / 120 real RGB frames, this covers 90 real temporal positions
(including frame119), not all120. No artificial temporal tail is supplied.

``full`` regions call the original spatially tiled ``_decode_clip``. ``tile``
regions call the very same decoder on one of its actual spatial tiles, and
exclude all spatial blend/crop boundaries. At384x672, LPIPS-compatible tile
interiors cover four corners; alternate with full regions to cover the center.
Uniformly sampled regions define a sparse-region objective, not an unbiased
estimate of the full-video pixel objective. The VAE remains noncausal/frozen.

L1 uses unclipped, ImageNet-denormalized floating RGB. Only the LPIPS branch
clips RGB to[0,1] and maps it to[-1,1]. No uint8 conversion or detach is applied
to the predicted pixels. Model/checkpoint loading belongs to the caller.
"""
from __future__ import annotations

import copy
import math
from contextlib import nullcontext

import torch
from torch.utils.checkpoint import checkpoint

from data_utils import latent_stats, native_attention, pixel_stats


class PixelSupervisor:
    """Frozen H3 VAE + frozen LPIPS, with explicit JSON-safe sampling specs.

    ``sample_window(rng, full=False)`` uses a caller-owned random.Random so its
    state can be checkpointed. ``specs`` / ``full_specs`` are deterministic
    validation choices. ``losses`` returns unweighted scalar tensors.
    """

    def __init__(self, vae, lpips_fn, device, *, latent_frames=37,
                 real_frames=120, latent_height=24, latent_width=42,
                 checkpoint_decode=True, min_spatial=64, lpips_frames=4):
        self.vae = vae.eval().requires_grad_(False)
        self.lpips_fn = lpips_fn.eval().requires_grad_(False)
        self.device = torch.device(device)
        self.latent_frames, self.real_frames = latent_frames, real_frames
        self.height, self.width = latent_height, latent_width
        self.checkpoint_decode = bool(checkpoint_decode)
        self.min_spatial, self.lpips_frames = min_spatial, lpips_frames
        if (any(type(v) is not int or v <= 0 for v in
                (latent_frames, real_frames, latent_height, latent_width,
                 min_spatial, lpips_frames)) or latent_frames < 7 or
                (latent_frames - 2) % 5):
            raise ValueError('Require positive dimensions and T=5*n+2, T>=7')
        self.blocks = (latent_frames - 2) // 5
        if real_frames > 17*self.blocks + 5:
            raise ValueError('Requested real RGB frames exceed the decoded length')
        self._validate_decoder()
        self.lmean, self.lstd = latent_stats(vae, self.device)
        self.pmean, self.pstd = pixel_stats(self.device)
        if not bool(torch.isfinite(self.lmean).all() and torch.isfinite(self.lstd).all()):
            raise ValueError('Nonfinite VAE latent normalization')
        self.full_specs, self.specs = [], []
        height, width = latent_height*16, latent_width*16
        ys, hs, yo = vae._split_tiles(height, 256, 64)
        xs, ws, xo = vae._split_tiles(width, 256, 64)
        for block in range(self.blocks):
            raw_frames = list(range(3 if block == 0 else 8, 20))
            global_frames = [17*block + n - 3 for n in raw_frames]
            if block == self.blocks - 1:
                raw_frames += list(range(23, 28))
                global_frames += list(range(17*self.blocks, 17*self.blocks+5))
            valid = [(r, g) for r, g in zip(raw_frames, global_frames) if g < real_frames]
            if not valid:
                continue
            raw_frames, global_frames = map(list, zip(*valid))
            common = {'version': 1, 'block_index': block, 'latent_start': 5*block,
                      'latent_stop': 5*block+7, 'raw_frame_indices': raw_frames,
                      'rgb_frame_indices': global_frames,
                      'lpips_frame_indices': self._even_indices(len(valid)),
                      'real_frames': real_frames, 'latent_frames': latent_frames}
            self.full_specs.append({**common, 'spatial_mode': 'full',
                'tile_row': -1, 'tile_column': -1,
                'latent_y': [0, latent_height], 'latent_x': [0, latent_width],
                'raw_y': [0, height], 'raw_x': [0, width],
                'rgb_y': [0, height], 'rgb_x': [0, width]})
            for i, (y, h) in enumerate(zip(ys, hs)):
                y0, y1 = (yo[i-1] if i else 0), h-(yo[i] if i < len(ys)-1 else 0)
                for j, (x, w) in enumerate(zip(xs, ws)):
                    x0, x1 = (xo[j-1] if j else 0), w-(xo[j] if j < len(xs)-1 else 0)
                    if min(y1-y0, x1-x0) < min_spatial:
                        continue
                    if any(v % 16 for v in (y, x, h, w)):
                        raise ValueError('Original VAE tile is not latent-grid aligned')
                    self.specs.append({**copy.deepcopy(common), 'spatial_mode': 'tile',
                        'tile_row': i, 'tile_column': j,
                        'latent_y': [y//16, (y+h)//16], 'latent_x': [x//16, (x+w)//16],
                        'raw_y': [y0, y1], 'raw_x': [x0, x1],
                        'rgb_y': [y+y0, y+y1], 'rgb_x': [x+x0, x+x1]})
        if not self.specs or not self.full_specs:
            raise ValueError('No valid pixel regions; check geometry/min_spatial')
        self._spec_lookup = {self._key(s): copy.deepcopy(s) for s in self.specs+self.full_specs}

    def _validate_decoder(self):
        vae = self.vae
        actual = (vae.config.clip_length, vae.config.token_drop,
                  vae.temporal_compression_ratio, vae.spatial_compression_ratio,
                  vae.tokens_chunk_size, vae.token_overlap,
                  vae.frame_pre_padding, vae.frame_overlap)
        if actual != (17, 3, 4, 16, 5, 2, 3, 5):
            raise ValueError(f'Unsupported H3 decoder geometry: {actual}')
        if not vae.use_tiling or (vae.tile_sample_min_height, vae.tile_sample_min_width,
                vae.tile_sample_min_overlap_height, vae.tile_sample_min_overlap_width) != (256, 256, 64, 64):
            raise ValueError('Pixel supervision requires deployment tile256/overlap64')
        for module in (vae, self.lpips_fn):
            if module.training or any(p.requires_grad for p in module.parameters()):
                raise ValueError('Decoder and LPIPS must remain frozen/eval')
            for p in module.parameters():
                if p.device != self.device or (p.is_floating_point() and p.dtype != torch.float32):
                    raise ValueError('Decoder and LPIPS must be FP32 on the selected device')

    @staticmethod
    def _key(spec):
        return (spec['spatial_mode'], spec['block_index'], spec['tile_row'], spec['tile_column'])

    def _even_indices(self, count):
        n = min(self.lpips_frames, count)
        return [0] if n == 1 else [round(i*(count-1)/(n-1)) for i in range(n)]

    def sample_window(self, rng, *, full=False):
        spec = copy.deepcopy(rng.choice(self.full_specs if full else self.specs))
        count = len(spec['raw_frame_indices'])
        spec['lpips_frame_indices'] = sorted(rng.sample(range(count), min(self.lpips_frames, count)))
        return spec

    def _validate_spec(self, spec):
        base = self._spec_lookup.get(self._key(spec))
        if base is None or set(spec) != set(base):
            raise ValueError('Unknown or incomplete pixel-region spec')
        if any(spec[k] != v for k, v in base.items() if k != 'lpips_frame_indices'):
            raise ValueError('Spec geometry changed; use sample_window or fixed specs')
        ids = spec['lpips_frame_indices']
        if (not isinstance(ids, list) or len(ids) != min(self.lpips_frames, len(base['raw_frame_indices']))
                or any(type(i) is not int for i in ids) or ids != sorted(set(ids))
                or min(ids) < 0 or max(ids) >= len(base['raw_frame_indices'])):
            raise ValueError('Invalid LPIPS frame indices')

    def _validate_latent(self, latent):
        if (latent.ndim != 5 or tuple(latent.shape) !=
                (1, 24, self.latent_frames, self.height, self.width)
                or latent.device != self.device or latent.dtype != torch.float32):
            raise ValueError('Expected FP32 [1,24,T,H,W] normalized latent on selected device')

    def _decode_region(self, latent, spec):
        """Return unclipped float RGB, preserving autograd on predicted latent."""
        self._validate_spec(spec)
        self._validate_latent(latent)
        y0, y1 = spec['latent_y']; x0, x1 = spec['latent_x']
        z = latent[:, :, spec['latent_start']:spec['latent_stop'], y0:y1, x0:x1]
        raw = z*self.lstd + self.lmean

        def run_tile(value):
            with native_attention():
                return self.vae.decoder(self.vae.post_quant_conv(value))

        needs_checkpoint = self.checkpoint_decode and torch.is_grad_enabled() and raw.requires_grad
        if spec['spatial_mode'] == 'full' and needs_checkpoint:
            # Checkpoint each real tile separately. An outer _decode_clip
            # checkpoint re-materializes all eight decoder graphs together
            # during backward and exceeded64GiB on the production H3 decoder.
            # Keep the released split coordinates and stitch implementation;
            # only activation storage/recomputation changes.
            ys, hs, yo = self.vae._split_tiles(raw.shape[-2]*16, 256, 64)
            xs, ws, xo = self.vae._split_tiles(raw.shape[-1]*16, 256, 64)
            rows = []
            for y, h in zip(ys, hs):
                row = []
                for x, w in zip(xs, ws):
                    tile = raw[..., y//16:(y+h)//16, x//16:(x+w)//16]
                    row.append(checkpoint(run_tile, tile, use_reentrant=False,
                                          preserve_rng_state=False))
                rows.append(row)
            decoded = self.vae._stitch_tiles(rows, yo, xo)
        elif spec['spatial_mode'] == 'full':
            # Independent reference route for no-grad parity and optional
            # checkpoint-off diagnostics: use the released method unchanged.
            with native_attention():
                decoded = self.vae._decode_clip(raw)
        elif needs_checkpoint:
            decoded = checkpoint(run_tile, raw, use_reentrant=False, preserve_rng_state=False)
        else:
            decoded = run_tile(raw)
        expected = (1, 3, 28, (y1-y0)*16, (x1-x0)*16)
        if tuple(decoded.shape) != expected:
            raise ValueError(f'Unexpected raw decoder shape: {tuple(decoded.shape)} != {expected}')
        y0, y1 = spec['raw_y']; x0, x1 = spec['raw_x']
        decoded = decoded[:, :, spec['raw_frame_indices'], y0:y1, x0:x1].float()
        return decoded*self.pstd + self.pmean

    def losses(self, pred, reference, spec):
        self._validate_decoder()
        pixels = self._decode_region(pred, spec)
        with torch.no_grad():
            target = self._decode_region(reference.detach(), spec)
        pixel_l1 = (pixels-target).abs().mean()

        def images(rgb):
            selected = rgb[:, :, spec['lpips_frame_indices']]
            return (selected.clamp(0., 1.)*2.-1.).permute(0, 2, 1, 3, 4).flatten(0, 1)

        perceptual = self.lpips_fn(images(pixels), images(target)).mean()
        if not bool(torch.isfinite(pixel_l1) & torch.isfinite(perceptual)):
            raise FloatingPointError('Nonfinite pixel loss')
        return {'pixel_l1': pixel_l1, 'lpips': perceptual}

    def description(self):
        frames = sorted({i for s in self.full_specs for i in s['rgb_frame_indices']})
        return {'latent_frames': self.latent_frames, 'real_frames': self.real_frames,
                'temporal_frames_eligible': frames,
                'temporal_frames_excluded': sorted(set(range(self.real_frames))-set(frames)),
                'full_spec_count': len(self.full_specs), 'tile_spec_count': len(self.specs),
                'spatial_tile_recipe': [256, 256, 64, 64], 'lpips_frames_per_spec': self.lpips_frames,
                'l1_domain': 'Unclipped ImageNet-denormalized floating RGB; all valid region frames',
                'lpips_domain': 'RGB clamped0..1 then mapped-1..1; explicit spec frame subset',
                'sampling': 'Caller-selected uniform regions; not full-video pixel-unbiased',
                'checkpoint_decode': self.checkpoint_decode,
                'decoder_checkpoint_scope': 'Each original spatial tile separately, then original _stitch_tiles',
                'decoder_trained': False,
                'end_to_end_causal': False}

    @staticmethod
    def _comparison(a, b, atol, rtol):
        a, b = a.detach().float(), b.detach().float()
        finite = bool(torch.isfinite(a).all() & torch.isfinite(b).all())
        delta = (a-b).abs()
        return {'passed': finite and bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
                'finite': finite, 'max_abs': float(delta.max()) if finite else None,
                'mean_abs': float(delta.mean()) if finite else None, 'atol': atol, 'rtol': rtol}

    def audit(self, pred, reference, specs=None, *, atol=1e-5, rtol=1e-5):
        """Explicit expensive preflight: full-decode parity and pixel-only grads.

        Defaults cover first/middle/last temporal blocks and full/tile modes.
        Caller can pass every spec for exhaustive regional parity. No optimizer
        is created; gradient checks use a detached perturbed input copy.
        """
        if not all(math.isfinite(v) and v >= 0 for v in (atol, rtol)):
            raise ValueError('Finite nonnegative audit tolerances required')
        self._validate_decoder()
        if specs is None:
            blocks = sorted({0, self.blocks//2, self.blocks-1})
            specs = [copy.deepcopy(next(s for s in collection if s['block_index'] == block))
                     for block in blocks for collection in (self.full_specs, self.specs)]
        if not specs:
            raise ValueError('Audit requires at least one region')
        records = []
        for label, value in (('pred', pred), ('reference', reference)):
            self._validate_latent(value)
            with torch.no_grad(), native_attention():
                full = self.vae.decode(value.detach()*self.lstd+self.lmean, return_dict=False)[0].float()
                if tuple(full.shape) != (1, 3, 17*self.blocks+5, self.height*16, self.width*16):
                    raise ValueError('Full decoder geometry changed')
                full = full[:, :, :self.real_frames]*self.pstd+self.pmean
                for spec in specs:
                    local = self._decode_region(value.detach(), spec)
                    y0, y1 = spec['rgb_y']; x0, x1 = spec['rgb_x']
                    expected = full[:, :, spec['rgb_frame_indices'], y0:y1, x0:x1]
                    records.append({'source': label, 'spec': copy.deepcopy(spec),
                                    **self._comparison(local, expected, atol, rtol)})
                del full
        grad_checks = []
        for mode in sorted({s['spatial_mode'] for s in specs}):
            spec = next(s for s in specs if s['spatial_mode'] == mode)
            # Keep probes independent of an accidentally identical pred/reference.
            offset = torch.randn(pred.shape, generator=torch.Generator().manual_seed(20261105),
                                 dtype=torch.float32).to(self.device)*.01
            probe = (pred.detach()+offset).requires_grad_(True)
            losses = self.losses(probe, reference.detach(), spec)
            for i, (key, loss) in enumerate(losses.items()):
                gradient, = torch.autograd.grad(loss, probe, retain_graph=i == 0)
                finite = bool(torch.isfinite(gradient).all())
                permitted = torch.zeros_like(gradient, dtype=torch.bool)
                y0, y1 = spec['latent_y']; x0, x1 = spec['latent_x']
                permitted[:, :, spec['latent_start']:spec['latent_stop'], y0:y1, x0:x1] = True
                outside = gradient.masked_select(~permitted)
                max_abs = float(gradient.abs().max()) if finite else None
                outside_max = float(outside.abs().max()) if outside.numel() else 0.
                grad_checks.append({'mode': mode, 'loss': key, 'finite': finite,
                    'max_abs': max_abs, 'outside_region_max_abs': outside_max,
                    'passed': finite and max_abs > 0. and outside_max == 0.})
            del probe, losses, offset
        frozen = all(p.grad is None and not p.requires_grad
                     for module in (self.vae, self.lpips_fn) for p in module.parameters())
        return {'passed': all(r['passed'] for r in records+grad_checks) and frozen,
                'parity': records, 'pixel_only_input_gradient': grad_checks,
                'frozen_parameter_grads_none': frozen,
                'gradient_probe': 'Detached pred plus deterministic0.01 Gaussian perturbation; no optimizer',
                'scope': self.description()}
