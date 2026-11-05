"""CPU-only geometry/autograd tests; no model weights, LPIPS download or NPU."""
from __future__ import annotations

import copy
import json
import math
import random
import unittest
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from h3_hybrid_pixel import PixelSupervisor


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(24, 3, 1)

    def forward(self, z):
        # Deliberately mix all7 time positions and all tile pixels: a wrong
        # context, phase or tile must not accidentally pass the parity test.
        x = self.conv(z)
        x = x + .17*x.mean(dim=(2, 3, 4), keepdim=True)
        x = x.repeat_interleave(4, 2)
        return x.repeat_interleave(16, 3).repeat_interleave(16, 4)


class FakeVAE(nn.Module):
    """Independent minimal mirror of released temporal/spatial assembly."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(clip_length=17, token_drop=3,
            latents_mean=[.11]*24, latents_std=[1.3]*24)
        self.tokens_chunk_size, self.token_overlap = 5, 2
        self.frame_pre_padding, self.frame_overlap = 3, 5
        self.temporal_compression_ratio, self.spatial_compression_ratio = 4, 16
        self.use_tiling = True
        self.tile_sample_min_height = self.tile_sample_min_width = 256
        self.tile_sample_min_overlap_height = self.tile_sample_min_overlap_width = 64
        self.post_quant_conv = nn.Conv3d(24, 24, 1)
        self.decoder = FakeDecoder()

    def _split_tiles(self, length, tile_size, min_overlap):
        if tile_size >= length:
            return [0], [length], []
        count = math.ceil(length/tile_size)
        while tile_size*count-min_overlap*(count-1)-length < 0:
            count += 1
        overlaps = [min_overlap]*(count-1)
        remaining = tile_size*count-sum(overlaps)-length
        for i in range(remaining//16):
            overlaps[i % (count-1)] += 16
        starts = [0]
        for overlap in overlaps:
            starts.append(starts[-1]+tile_size-overlap)
        return starts, [tile_size]*count, overlaps

    @staticmethod
    def blend(a, b, extent, dim):
        sa, sb = [slice(None)]*5, [slice(None)]*5
        sa[dim], sb[dim] = slice(-extent, None), slice(0, extent)
        shape = [1]*5; shape[dim] = extent
        w = (torch.arange(extent, dtype=b.dtype)/extent).reshape(shape)
        out = a[tuple(sa)]*(1-w)+b[tuple(sb)]*w
        sb[dim] = slice(extent, None)
        return torch.cat((out, b[tuple(sb)]), dim=dim)

    def _decode_clip(self, z):
        ys, hs, yo = self._split_tiles(z.shape[-2]*16, 256, 64)
        xs, ws, xo = self._split_tiles(z.shape[-1]*16, 256, 64)
        rows = [[self.decoder(self.post_quant_conv(z[..., y//16:(y+h)//16, x//16:(x+w)//16]))
                 for x, w in zip(xs, ws)] for y, h in zip(ys, hs)]
        return self._stitch_tiles(rows, yo, xo)

    def _stitch_tiles(self, rows, yo, xo):
        result = []
        for i, row in enumerate(rows):
            columns = []
            for j, tile in enumerate(row):
                if i: tile = self.blend(rows[i-1][j], tile, yo[i-1], -2)
                if j: tile = self.blend(row[j-1], tile, xo[j-1], -1)
                if i < len(rows)-1: tile = tile[..., :-yo[i], :]
                if j < len(row)-1: tile = tile[..., :-xo[j]]
                columns.append(tile)
            result.append(torch.cat(columns, -1))
        return torch.cat(result, -2)

    def decode(self, z, return_dict=False):
        assert (z.shape[2]-2) % 5 == 0
        chunks, overlap = [], None
        for start in range(0, z.shape[2]-2, 5):
            raw = self._decode_clip(z[:, :, start:start+7])
            current = raw[:, :, 3:20]
            if overlap is not None:
                current = self.blend(overlap, current, 5, 2)
            chunks.append(current)
            overlap = raw[:, :, 23:28]
        return (torch.cat(chunks+[overlap], dim=2),)


class FakeLPIPS(nn.Module):
    def forward(self, a, b):
        return (a-b).square().mean(dim=(1, 2, 3), keepdim=True)


class PixelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.vae = FakeVAE()
        self.sup = PixelSupervisor(self.vae, FakeLPIPS(), 'cpu', latent_frames=12,
            real_frames=35, latent_height=8, latent_width=20, checkpoint_decode=False)
        self.pred = torch.randn(1, 24, 12, 8, 20)*.3
        self.ref = torch.randn_like(self.pred)*.3

    def test_every_region_full_decode_parity_and_gradients(self):
        report = self.sup.audit(self.pred, self.ref, self.sup.specs+self.sup.full_specs,
                                atol=1e-6, rtol=1e-6)
        self.assertTrue(report['passed'], report)

    def test_production_geometry_and_json_sampling(self):
        sup = PixelSupervisor(FakeVAE(), FakeLPIPS(), 'cpu')
        self.assertEqual(len(sup.full_specs), 7)
        self.assertEqual(len(sup.specs), 28)
        desc = sup.description()
        self.assertEqual(len(desc['temporal_frames_eligible']), 90)
        self.assertEqual(len(desc['temporal_frames_excluded']), 30)
        self.assertEqual(desc['temporal_frames_eligible'][-1], 119)
        a, b = random.Random(9), random.Random(9)
        for i in range(40):
            spec = sup.sample_window(a, full=i % 4 == 3)
            self.assertEqual(spec, sup.sample_window(b, full=i % 4 == 3))
            sup._validate_spec(json.loads(json.dumps(spec)))
            self.assertEqual(spec['latent_stop']-spec['latent_start'], 7)
            self.assertEqual(spec['latent_start'] % 5, 0)

    def test_checkpoint_values_and_input_gradients(self):
        for spec in (self.sup.specs[-1], self.sup.full_specs[-1]):
            with self.subTest(mode=spec['spatial_mode']):
                grads, values, pixels = [], [], []
                for enabled in (False, True):
                    self.sup.checkpoint_decode = enabled
                    x = self.pred.clone().requires_grad_(True)
                    pixels.append(self.sup._decode_region(x, spec).detach())
                    terms = self.sup.losses(x, self.ref, spec)
                    loss = terms['pixel_l1']+.2*terms['lpips']
                    values.append(loss.detach())
                    grads.append(torch.autograd.grad(loss, x)[0])
                torch.testing.assert_close(pixels[0], pixels[1], atol=1e-7, rtol=1e-6)
                torch.testing.assert_close(values[0], values[1], atol=1e-7, rtol=1e-6)
                torch.testing.assert_close(grads[0], grads[1], atol=1e-7, rtol=1e-6)

    def test_full_checkpoint_two_dimensional_stitch_gradient(self):
        # Exercise both vertical and horizontal blends. The routine small
        # geometry has only horizontal tiling, which is insufficient here.
        sup = PixelSupervisor(FakeVAE(), FakeLPIPS(), 'cpu', latent_frames=7,
            real_frames=18, latent_height=20, latent_width=20)
        x0 = torch.randn(1, 24, 7, 20, 20)*.3
        target = torch.randn_like(x0)*.3
        grads, terms = [], []
        for enabled in (False, True):
            sup.checkpoint_decode = enabled
            x = x0.clone().requires_grad_(True)
            loss = sup.losses(x, target, sup.full_specs[0])
            combined = loss['pixel_l1']+.2*loss['lpips']
            terms.append(combined.detach())
            grads.append(torch.autograd.grad(combined, x)[0])
        torch.testing.assert_close(terms[0], terms[1], atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(grads[0], grads[1], atol=1e-7, rtol=1e-6)

    def test_tampered_geometry_rejected(self):
        bad = copy.deepcopy(self.sup.specs[0]); bad['latent_start'] += 1
        with self.assertRaises(ValueError): self.sup.losses(self.pred, self.ref, bad)
        self.vae.use_tiling = False
        with self.assertRaises(ValueError): self.sup.losses(self.pred, self.ref, self.sup.specs[0])

    def test_reference_detached_and_l1_unclipped_rgb(self):
        spec = self.sup.full_specs[0]
        x, target = (self.pred+8).requires_grad_(True), self.ref.clone().requires_grad_(True)
        terms = self.sup.losses(x, target, spec)
        terms['pixel_l1'].backward()
        self.assertIsNone(target.grad)
        self.assertGreater(float(x.grad.abs().max()), 0)
        self.assertTrue(all(p.grad is None for p in self.vae.parameters()))
        with torch.no_grad():
            a, b = self.sup._decode_region(x, spec), self.sup._decode_region(target, spec)
            torch.testing.assert_close(terms['pixel_l1'], (a-b).abs().mean())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
