"""Frozen H3 region decoder for production67 canonical pixel specs.

Candidate only. Synthetic CPU checks do not establish real VAE/NPU gradients.
No weights, device selection, optimizer, loss recipe, or training loop is loaded.
The caller must inject the authorized frozen FP32 VAE and an explicit scope.
"""
from __future__ import annotations

import copy
import torch
from torch.utils.checkpoint import checkpoint

from data_utils import latent_stats, native_attention, pixel_stats
from h3_production67_geometry_v1 import canonical_sha, production67
from h3_production67_pixel_specs_v1 import PixelSpecCatalog, split_tiles

SCHEMA = 'h3-production67-pixel-decoder-v1-candidate'


class Production67PixelDecoder:
    def __init__(self, vae, device, *, temporal_scope, checkpoint_decode=True,
                 allow_full_spatial_grad=False, min_spatial=64, lpips_count=4):
        if type(checkpoint_decode) is not bool or type(allow_full_spatial_grad) is not bool:
            raise ValueError('Checkpoint and full-gradient options must be explicit booleans')
        self.vae = vae
        self.device = torch.device(device)
        self.checkpoint_decode = checkpoint_decode
        self.allow_full_spatial_grad = allow_full_spatial_grad
        self.catalog = PixelSpecCatalog(production67(), temporal_scope=temporal_scope,
                                       min_spatial=min_spatial, lpips_count=lpips_count)
        self._parameter_identity = tuple((n, id(p), p._version) for n, p in vae.named_parameters())
        self._buffer_identity = tuple((n, id(p), p._version) for n, p in vae.named_buffers())
        self._normalization = copy.deepcopy([list(vae.config.latents_mean), list(vae.config.latents_std)])
        self.validate_frozen()
        self.lmean, self.lstd = latent_stats(vae, self.device)
        self.pmean, self.pstd = pixel_stats(self.device)
        if not bool(torch.isfinite(self.lmean).all() & torch.isfinite(self.lstd).all()):
            raise ValueError('Nonfinite latent normalization')

    def validate_frozen(self):
        vae = self.vae
        actual = (vae.config.clip_length, vae.config.token_drop,
                  vae.temporal_compression_ratio, vae.spatial_compression_ratio,
                  vae.tokens_chunk_size, vae.token_overlap, vae.frame_pre_padding, vae.frame_overlap)
        if actual != (17, 3, 4, 16, 5, 2, 3, 5):
            raise ValueError('VAE temporal/spatial geometry differs from the frozen recipe')
        if vae.use_tiling is not True or (vae.tile_sample_min_height, vae.tile_sample_min_width,
                vae.tile_sample_min_overlap_height, vae.tile_sample_min_overlap_width) != (256, 256, 64, 64):
            raise ValueError('VAE must preserve tile256/overlap64')
        if any(m.training for m in vae.modules()):
            raise ValueError('VAE and all its modules must already be eval')
        if tuple((n, id(p), p._version) for n, p in vae.named_parameters()) != self._parameter_identity:
            raise ValueError('Frozen VAE parameter identity/version changed')
        if tuple((n, id(p), p._version) for n, p in vae.named_buffers()) != self._buffer_identity:
            raise ValueError('Frozen VAE buffer identity/version changed')
        if [list(vae.config.latents_mean), list(vae.config.latents_std)] != self._normalization:
            raise ValueError('Latent normalization changed')
        for p in list(vae.parameters()) + list(vae.buffers()):
            if p.device != self.device or (p.is_floating_point() and p.dtype != torch.float32):
                raise ValueError('Frozen VAE must be FP32 on the explicit device')
            if p.requires_grad or p.grad is not None:
                raise ValueError('VAE must remain frozen with no parameter/buffer gradients')
        for size in (1088, 1920):
            if vae._split_tiles(size, 256, 64) != split_tiles(size):
                raise ValueError('Actual VAE tile split differs from the canonical specs')
        return True

    def _validate_latent(self, latent):
        if (not torch.is_tensor(latent) or tuple(latent.shape) != (1, 24, 67, 68, 120)
                or latent.dtype != torch.float32 or latent.device != self.device):
            raise ValueError('Require normalized FP32 [1,24,67,68,120] on the explicit device')
        if not bool(torch.isfinite(latent).all()):
            raise ValueError('Nonfinite prediction/target latent')

    def _decode_block(self, latent, spatial, block):
        ly0, ly1, lx0, lx1 = spatial['target_latent_yx']
        raw = latent[:, :, block*5:block*5+7, ly0:ly1, lx0:lx1]*self.lstd + self.lmean

        def run_tile(value):
            with native_attention():
                return self.vae.decoder(self.vae.post_quant_conv(value))

        use_cp = self.checkpoint_decode and torch.is_grad_enabled() and raw.requires_grad
        full = spatial['spatial_mode'] == 'full_spatial'
        if full and use_cp:
            # Recompute each actual spatial tile separately; never checkpoint a
            # single outer _decode_clip that reconstructs all60 decoder graphs.
            ys, hs, yo = self.vae._split_tiles(1088, 256, 64)
            xs, ws, xo = self.vae._split_tiles(1920, 256, 64)
            rows = []
            for y, h in zip(ys, hs):
                row = []
                for x, w in zip(xs, ws):
                    tile = raw[..., y//16:(y+h)//16, x//16:(x+w)//16]
                    row.append(checkpoint(run_tile, tile, use_reentrant=False, preserve_rng_state=False))
                rows.append(row)
            decoded = self.vae._stitch_tiles(rows, yo, xo)
        elif full:
            with native_attention():
                decoded = self.vae._decode_clip(raw)
        elif use_cp:
            decoded = checkpoint(run_tile, raw, use_reentrant=False, preserve_rng_state=False)
        else:
            decoded = run_tile(raw)
        expected = (1, 3, 28, (ly1-ly0)*16, (lx1-lx0)*16)
        if tuple(decoded.shape) != expected or decoded.dtype != torch.float32 or decoded.device != self.device:
            raise ValueError('Raw decoder must return exact FP32 28-frame spatial geometry')
        y0, y1, x0, x1 = spatial['decoder_local_box_yx']
        return decoded[:, :, :, y0:y1, x0:x1]

    @staticmethod
    def _assemble(decoded_blocks, recipes):
        """Use released FP32 subtraction/division order, not rational->float a.

        In particular 1 - FP32(3/5) is not bit-equal to FP32(2/5).
        Keep zero-weight current contributions at each blend's first frame.
        """
        frames = []
        for recipe in recipes:
            pieces = recipe['contributions']
            first = pieces[0]
            a = decoded_blocks[first['block_index']][:, :, first['raw_frame_index']:first['raw_frame_index']+1]
            if len(pieces) == 1:
                frames.append(a)
            else:
                second = pieces[1]
                b = decoded_blocks[second['block_index']][:, :, second['raw_frame_index']:second['raw_frame_index']+1]
                position = recipe['rgb_frame'] % 17
                wb = torch.tensor(position, device=b.device, dtype=b.dtype) / 5
                wa = 1 - wb
                frames.append(a*wa + b*wb)
        return torch.cat(frames, dim=2)

    def decode_region(self, latent, spec):
        """Return unclipped FP32 [1,3,K,H,W], preserving predicted autograd."""
        self.catalog.validate_spec(spec)
        self.validate_frozen()
        self._validate_latent(latent)
        spatial = spec['spatial']
        if (spatial['spatial_mode'] == 'full_spatial' and torch.is_grad_enabled()
                and latent.requires_grad and not self.allow_full_spatial_grad):
            raise ValueError('Full-spatial backward has no default memory authorization; explicitly opt in after its gate')
        blocks = sorted({c['block_index'] for recipe in spec['temporal_assembly'] for c in recipe['contributions']})
        decoded = {block: self._decode_block(latent, spatial, block) for block in blocks}
        rgb = self._assemble(decoded, spec['temporal_assembly'])*self.pstd + self.pmean
        y0, y1, x0, x1 = spatial['visible_global_box_yx']
        if tuple(rgb.shape) != (1, 3, len(spec['rgb_frame_indices']), y1-y0, x1-x0):
            raise ValueError('Assembled visible-region shape differs from canonical spec')
        if not bool(torch.isfinite(rgb).all()):
            raise FloatingPointError('Nonfinite unclipped decoder RGB')
        return rgb

    def description(self):
        return {'schema': SCHEMA, 'catalog': self.catalog.description(),
                'input_shape': [1,24,67,68,120], 'raw_hw': [1088,1920],
                'visible_crop_yx': [4,1084,0,1920], 'output_range': 'unclipped floating RGB',
                'checkpoint_decode': self.checkpoint_decode,
                'checkpoint_scope': 'each actual256x256 decoder spatial tile separately',
                'allow_full_spatial_grad': self.allow_full_spatial_grad,
                'temporal_blend_weights': 'released FP32 position/5 and 1-position/5, before RGB denormalization',
                'normalization_sha256': canonical_sha(self._normalization),
                'decoder_trained': False, 'end_to_end_causal': False,
                'new_objective_adopted': False, 'formal_training_allowed': False,
                'real_decoder_gradient_validated': False}


def decoder_l1_gradient_probe(provider, prediction, teacher, spec):
    """One region, one leaf-latent backward; no optimizer or upscaler backward.

    Caller owns exact VAE/source/latent SHA, physical device and three locks.
    This does not authorize training or certify full-video/model gradients.
    """
    if spec['spatial']['spatial_mode'] != 'stable_tile_interior':
        raise ValueError('Minimum decoder probe is one actual spatial tile only')
    with torch.no_grad():
        target = provider.decode_region(teacher.detach(), spec)
    probe = prediction.detach().clone().requires_grad_(True)
    pixels = provider.decode_region(probe, spec)
    loss = (pixels-target).abs().mean()
    grad, = torch.autograd.grad(loss, probe)
    if not bool(torch.isfinite(loss) & torch.isfinite(grad).all()) or not bool((grad != 0).any()):
        raise FloatingPointError('Decoder leaf-latent L1 gradient must be finite and nonzero')
    outside = grad.detach().clone()
    y0,y1,x0,x1 = spec['spatial']['target_latent_yx']
    spans = sorted({(c['latent_start'],c['latent_stop']) for r in spec['temporal_assembly'] for c in r['contributions']})
    for t0,t1 in spans:
        outside[:,:,t0:t1,y0:y1,x0:x1] = 0
    if bool((outside != 0).any()):
        raise AssertionError('Input gradient escaped the actual selected decoder latent windows')
    provider.validate_frozen()
    return {'schema': 'h3-production67-decoder-leaf-gradient-probe-v1', 'passed': True,
            'spec_sha256': canonical_sha(spec), 'pixel_l1': float(loss.detach()),
            'gradient_l2': float(grad.detach().double().norm()),
            'gradient_maxabs': float(grad.detach().abs().max()),
            'outside_selected_latent_window_maxabs': float(outside.abs().max()),
            'latent_spans': [list(s) for s in spans], 'decoder_parameter_gradients': False,
            'upscaler_gradient_tested': False, 'optimizer_updates': 0,
            'formal_training_allowed': False}
