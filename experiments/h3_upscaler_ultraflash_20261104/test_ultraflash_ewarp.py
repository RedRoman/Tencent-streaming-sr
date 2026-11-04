"""Independent, small tests of H3's Ultra Flash adaptation; no model weights.

    python -B experiment/test_ultraflash_ewarp.py --device cpu
    python -B experiment/test_ultraflash_ewarp.py --device npu:0

The optical-flow estimator always runs on CPU. Its exact interface is tested
with a deterministic spy, and a real Farneback translation test runs when cv2
is installed. Causality checks exercise the actual operator adapter on a tiny
network; they do not establish full-model quality or VAE/refiner causality.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from causal_upscaler import CausalUpscalerAdapter
from ultraflash_ewarp import anchors, flow_from_rgb, losses, warp_previous


DEVICE = torch.device('cpu')
EXPECTED_ANCHORS = [
    0, 4, 8, 12, 16, 17, 21, 25, 29, 33, 34, 38, 42, 46, 50,
    51, 55, 59, 63, 67, 68, 72, 76, 80, 84, 85, 89, 93, 97, 101,
    102, 106, 110, 114, 118, 119, 123,
]


class FlowSpy:
    """Constant known quarter-RGB motion, with observable frame ordering."""
    INTER_AREA = 1
    INTER_LINEAR = 2
    COLOR_RGB2GRAY = 3

    def __init__(self):
        self.calls = []

    @staticmethod
    def resize(image, size, interpolation):
        width, height = size
        yi = np.arange(height) * image.shape[0] // height
        xi = np.arange(width) * image.shape[1] // width
        return image[yi[:, None], xi[None, :]].copy()

    @staticmethod
    def cvtColor(image, convention):
        return image[..., 0].copy()

    def calcOpticalFlowFarneback(self, current, previous, initial, **kwargs):
        self.calls.append((int(current[0, 0]), int(previous[0, 0])))
        result = np.empty((*current.shape, 2), dtype=np.float32)
        result[..., 0], result[..., 1] = 6., -2.
        return result


class TinyUpscaler(nn.Module):
    """Author-compatible call signature; nontrivial temporal Conv and GN."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(2, 2, 3, padding=1)
        self.norm = nn.GroupNorm(1, 2)
        self.conv2 = nn.Conv3d(2, 2, 3, padding=1)

    def forward(self, value, *, scale, target_size, enable_chunking):
        value = self.conv2(F.silu(self.norm(self.conv1(value))))
        return F.interpolate(value, size=target_size, mode='nearest')


class UltraFlashEWarpTests(unittest.TestCase):
    def assert_close(self, actual, expected, *, atol=3e-5, rtol=1e-5):
        if not isinstance(expected, torch.Tensor):
            expected = torch.as_tensor(expected, dtype=actual.dtype, device=actual.device)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def plane(self, height=5, width=7):
        yy, xx = torch.meshgrid(torch.arange(height, device=DEVICE),
                                torch.arange(width, device=DEVICE), indexing='ij')
        return (2 * xx + 3 * yy).float()[None, None]

    def test_h3_temporal_anchors_and_padding(self):
        self.assertEqual(anchors(37), EXPECTED_ANCHORS)
        self.assertEqual(sum(t < 120 for t in anchors()), 36)
        deltas = [b-a for a, b in zip(EXPECTED_ANCHORS, EXPECTED_ANCHORS[1:])]
        self.assertEqual([j+1 for j, d in enumerate(deltas) if d == 1],
                         [5, 10, 15, 20, 25, 30, 35])
        self.assertEqual(deltas[-1], 4)

    def test_flow_estimator_direction_scale_and_real_pairs(self):
        rgb = np.broadcast_to(np.arange(120, dtype=np.uint8)[:, None, None, None],
                              (120, 64, 96, 3)).copy()
        spy = FlowSpy()
        with patch.dict(sys.modules, {'cv2': spy}):
            flow, valid, metadata = flow_from_rgb(rgb, (8, 8), 37)
        # Source flow is on 16x24, target on 8x8: x /3, y /2.
        self.assertEqual(spy.calls, list(zip(EXPECTED_ANCHORS[1:36], EXPECTED_ANCHORS[:35])))
        self.assertEqual(flow.shape, (36, 2, 8, 8))
        self.assert_close(flow[:35, 0], torch.full((35, 8, 8), 2.))
        self.assert_close(flow[:35, 1], torch.full((35, 8, 8), -1.))
        self.assertEqual(int(torch.count_nonzero(flow[-1])), 0)
        self.assertEqual(valid.tolist(), [True] * 35 + [False])
        self.assertFalse(flow.requires_grad)
        self.assertEqual(metadata['valid_latent_frames'], list(range(36)))
        self.assertEqual(metadata['source_flow_hw'], [16, 24])
        self.assertEqual(metadata['latent_flow_hw'], [8, 8])
        self.assertEqual(metadata['vector_scale_xy'], [1/3, 1/2])

    def test_known_integer_backward_translation_and_inbounds(self):
        previous = self.plane().repeat(2, 2, 1, 1)
        flow = torch.zeros(2, 2, 5, 7, device=DEVICE)
        flow[:, 0], flow[:, 1] = 1., -1.
        warped, valid = warp_previous(previous, flow)
        # current[y,x] samples previous[y-1,x+1], not the opposite direction.
        self.assert_close(warped[:, :, 1:, :-1], previous[:, :, :-1, 1:])
        expected_mask = torch.zeros(2, 1, 5, 7, dtype=torch.bool, device=DEVICE)
        expected_mask[:, :, 1:, :-1] = True
        self.assertTrue(torch.equal(valid, expected_mask))
        wrong, _ = warp_previous(previous, -flow)
        self.assertGreater(float((wrong[:, :, 1:-1, 1:-1] - warped[:, :, 1:-1, 1:-1]).abs().mean()), 1.)

    def test_fractional_bilinear_translation_has_analytic_value(self):
        previous = self.plane()
        flow = torch.empty(1, 2, 5, 7, device=DEVICE)
        flow[:, 0], flow[:, 1] = .25, .5
        warped, valid = warp_previous(previous, flow)
        # For f(x,y)=2x+3y, f(x+.25,y+.5)=f(x,y)+2 exactly.
        self.assert_close(warped[:, :, :-1, :-1], previous[:, :, :-1, :-1] + 2.)
        self.assertEqual(int(valid.sum()), 24)

    def test_invalid_spatial_samples_do_not_enter_loss(self):
        previous = self.plane()
        flow = torch.empty(1, 2, 5, 7, device=DEVICE)
        flow[:, 0], flow[:, 1] = 1., -1.
        current, valid = warp_previous(previous, flow)
        current = current + (~valid).float() * 1000.
        pred = torch.stack((previous, current), dim=2).requires_grad_(True)
        result = losses(pred, pred.detach().clone(), flow,
                        torch.ones(1, dtype=torch.bool, device=DEVICE), valid_frames=2)
        self.assert_close(result['ewarp_l1'], 0.)
        self.assert_close(result['valid_warp_fraction'], 24/35)
        result['loss'].backward()
        self.assertTrue(bool(torch.isfinite(pred.grad).all()))
        self.assert_close(pred.grad, torch.zeros_like(pred.grad))

    def test_nonzero_motion_loss_has_past_and_current_gradients(self):
        previous = self.plane()
        flow = torch.empty(1, 2, 5, 7, device=DEVICE)
        flow[:, 0], flow[:, 1] = .25, .5
        current = warp_previous(previous, flow)[0] + .75
        pred = torch.stack((previous, current), dim=2).detach().requires_grad_(True)
        result = losses(pred, pred.detach().clone(), flow,
                        torch.ones(1, dtype=torch.bool, device=DEVICE),
                        warp_weight=.2, valid_frames=2)
        self.assert_close(result['latent_mse'], 0.)
        self.assert_close(result['ewarp_l1'], .75)
        self.assert_close(result['loss'], .15)
        result['loss'].backward()
        self.assertTrue(bool(torch.isfinite(pred.grad).all()))
        self.assertGreater(float(pred.grad[:, :, 0].abs().sum()), 0.)
        self.assertGreater(float(pred.grad[:, :, 1].abs().sum()), 0.)
        self.assert_close(pred.grad[:, :, 0].sum(), -.2)
        self.assert_close(pred.grad[:, :, 1].sum(), .2)

    def test_no_valid_pixels_is_finite_differentiable_zero(self):
        pred = torch.arange(70, device=DEVICE, dtype=torch.float32).reshape(1, 1, 2, 5, 7).requires_grad_(True)
        flow = torch.full((1, 2, 5, 7), 100., device=DEVICE)
        result = losses(pred, pred.detach().clone(), flow,
                        torch.ones(1, dtype=torch.bool, device=DEVICE), valid_frames=2)
        self.assert_close(result['ewarp_l1'], 0.)
        self.assert_close(result['valid_warp_fraction'], 0.)
        result['loss'].backward()
        self.assertTrue(bool(torch.isfinite(pred.grad).all()))
        self.assert_close(pred.grad, torch.zeros_like(pred.grad))

    def test_padding_latent_and_its_pair_have_no_loss_or_gradient(self):
        pred = torch.arange(37 * 3 * 4, device=DEVICE, dtype=torch.float32).reshape(1, 1, 37, 3, 4) / 100.
        target = torch.zeros_like(pred)
        flow = torch.zeros(36, 2, 3, 4, device=DEVICE)
        valid = torch.tensor([True] * 35 + [False], device=DEVICE)
        baseline = losses(pred, target, flow, valid, valid_frames=36)
        poisoned = pred.clone()
        poisoned[:, :, 36] = 1.e6
        poisoned.requires_grad_(True)
        target[:, :, 36] = -1.e6
        flow[-1] = 1000.
        changed = losses(poisoned, target, flow, valid, valid_frames=36)
        for name in baseline:
            self.assert_close(changed[name], baseline[name])
        changed['loss'].backward()
        self.assertTrue(bool(torch.isfinite(poisoned.grad).all()))
        self.assertEqual(float(poisoned.grad[:, :, 36].abs().sum()), 0.)
        self.assertGreater(float(poisoned.grad[:, :, :36].abs().sum()), 0.)

    def test_prefix_warp_loss_cannot_reach_future_model_inputs(self):
        torch.manual_seed(11)
        model = TinyUpscaler().eval().to(DEVICE)
        node = SimpleNamespace(_make_norm_tensors=lambda device, dtype: (
            torch.zeros(1, 2, 1, 1, 1, device=device, dtype=dtype),
            torch.ones(1, 2, 1, 1, 1, device=device, dtype=dtype)))
        source = torch.arange(1 * 2 * 6 * 4 * 5, dtype=torch.float32).reshape(1, 2, 6, 4, 5).to(DEVICE) / 50.
        poisoned = source.clone()
        poisoned[:, :, 3:] = 10. - 2. * poisoned[:, :, 3:]
        with torch.no_grad():
            original = model(source, scale=1., target_size=(6, 4, 5), enable_chunking=False)
            changed = model(poisoned, scale=1., target_size=(6, 4, 5), enable_chunking=False)
        self.assertGreater(float((original[:, :, :3] - changed[:, :, :3]).abs().max()), .001)
        adapter = CausalUpscalerAdapter(node, model)
        try:
            source = source.detach().requires_grad_(True)
            output = adapter.native_forward(source, (6, 4, 5))
            altered = adapter.native_forward(poisoned, (6, 4, 5))
            prefix = adapter.native_forward(source[:, :, :3], (3, 4, 5))
            self.assert_close(output[:, :, :3], altered[:, :, :3])
            self.assert_close(output[:, :, :3], prefix)
            flow = torch.zeros(2, 2, 4, 5, device=DEVICE)
            flow[:, 0] = .5
            result = losses(output[:, :, :3], output[:, :, :3].detach(), flow,
                            torch.tensor([False, True], device=DEVICE), valid_frames=3)
            self.assertGreater(float(result['ewarp_l1']), 0.)
            result['ewarp_l1'].backward()
            self.assertTrue(bool(torch.isfinite(source.grad).all()))
            self.assertEqual(float(source.grad[:, :, 3:].abs().max()), 0.)
            self.assertGreater(float(source.grad[:, :, 2].abs().sum()), 0.)
            self.assertGreater(float(source.grad[:, :, :2].abs().sum()), 0.)
        finally:
            adapter.remove()

    @unittest.skipUnless(importlib.util.find_spec('cv2') is not None, 'OpenCV not installed; estimator interface is covered by deterministic spy')
    def test_real_farneback_on_known_rgb_translation(self):
        rng = np.random.default_rng(23)
        quarter = rng.integers(0, 256, (40, 48, 3), dtype=np.uint8)
        first = np.repeat(np.repeat(quarter, 4, axis=0), 4, axis=1)
        rgb = np.stack([first] * 5)
        # Content moves right 8 RGB pixels and down 4: backward displacement
        # is (-8,-4), or (-.5,-.25) on this 10x12 latent grid.
        rgb[4] = np.roll(first, shift=(4, 8), axis=(0, 1))
        flow, valid, _ = flow_from_rgb(rgb, (10, 12), 2)
        self.assertEqual(valid.tolist(), [True])
        median = flow[0, :, 2:-2, 2:-2].flatten(1).median(dim=1).values
        self.assert_close(median, torch.tensor([-.5, -.25]), atol=.1, rtol=0.)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args, unittest_args = parser.parse_known_args()
    if args.device.split(':')[0] == 'npu':
        import torch_npu  # noqa: F401
    DEVICE = torch.device(args.device)
    if DEVICE.type == 'npu':
        torch.npu.set_device(DEVICE)
    torch.set_num_threads(2)
    unittest.main(argv=[sys.argv[0], *unittest_args], verbosity=2)
