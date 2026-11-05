"""CPU tensor/autograd and decoder-region regression tests; no NPU or weights."""
from __future__ import annotations

import unittest

import torch

from h3_hybrid_pixel import PixelSupervisor
from h3_v5_motion import MotionPixelSupervisor, rgb_motion_losses, target_residual_ewarp
from test_h3_hybrid_pixel import FakeLPIPS, FakeVAE
from ultraflash_ewarp import losses as legacy_losses


class RGBMotionTests(unittest.TestCase):
    @staticmethod
    def clip(values):
        return torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1, 1, 1).repeat(1, 3, 1, 2, 2)

    def test_static_target_empty_roi_and_perfect_prediction(self):
        truth = self.clip([.2, .2, .2])
        pred = truth.clone().requires_grad_()
        result = rgb_motion_losses(pred, truth, [0, 1, 2], [3, 4, 5])
        self.assertEqual(float(result['motion_l1']), 0.)
        self.assertEqual(float(result['delta_l1']), 0.)
        self.assertEqual(float(result['motion_roi_fraction']), 0.)
        (result['motion_l1']+result['delta_l1']).backward()
        self.assertIsNotNone(pred.grad)
        self.assertEqual(float(pred.grad.abs().sum()), 0.)

    def test_real_motion_perfect_prediction_is_zero(self):
        truth = torch.zeros(1, 3, 3, 4, 4)
        for frame in range(3):
            truth[:, :, frame, 1:3, frame:frame+2] = 1.
        pred = truth.clone().requires_grad_()
        result = rgb_motion_losses(pred, truth, [5, 6, 7], [8, 9, 10])
        self.assertEqual(float(result['motion_l1']), 0.)
        self.assertEqual(float(result['delta_l1']), 0.)
        self.assertGreater(float(result['motion_roi_fraction']), 0.)
        (result['motion_l1']+result['delta_l1']).backward()
        self.assertEqual(float(pred.grad.abs().sum()), 0.)

    def test_ghost_motion_produces_gradients_for_both_losses(self):
        truth = torch.zeros(1, 3, 2, 3, 4)
        truth[:, :, 0, :, 0] = 1.
        truth[:, :, 1, :, 2] = 1.
        pred = truth.clone()
        pred[:, :, 1] = .5*truth[:, :, 0]+.5*truth[:, :, 1]
        pred.requires_grad_()
        result = rgb_motion_losses(pred, truth, [0, 1], [3, 4])
        for name in ('motion_l1', 'delta_l1'):
            self.assertGreater(float(result[name]), 0.)
            grad, = torch.autograd.grad(result[name], pred, retain_graph=True)
            self.assertTrue(bool(torch.isfinite(grad).all()))
            self.assertGreater(float(grad.abs().sum()), 0.)

    def test_mask_channel_mean_threshold_and_both_endpoints(self):
        truth = torch.zeros(1, 3, 2, 1, 2)
        truth[:, 0, 1, 0, 0] = .15  # RGB mean change .05: inclusive boundary.
        truth[:, 0, 1, 0, 1] = .12  # RGB mean change .04: excluded.
        pred = truth.clone()
        pred[:, :, 0, 0, 0] += .2
        pred[:, :, 1, 0, 0] += .4
        pred[:, :, :, 0, 1] += 9.
        pred.requires_grad_()
        result = rgb_motion_losses(pred, truth, [0, 1], [0, 1])
        torch.testing.assert_close(result['motion_l1'], torch.tensor(.3))
        self.assertEqual(float(result['motion_roi_fraction']), .5)
        result['motion_l1'].backward()
        self.assertGreater(float(pred.grad[:, :, 0, :, 0].abs().sum()), 0.)
        self.assertGreater(float(pred.grad[:, :, 1, :, 0].abs().sum()), 0.)
        self.assertEqual(float(pred.grad[:, :, :, :, 1].abs().sum()), 0.)

    def test_empty_adjacent_pairs_are_graph_connected_zero(self):
        for rgb, raw in (([0, 2], [0, 1]), ([0, 2], [0, 3]), ([0], [3])):
            pred = self.clip(list(range(len(rgb)))).requires_grad_()
            truth = (pred.detach()+1.).requires_grad_()
            result = rgb_motion_losses(pred, truth, rgb, raw)
            self.assertEqual(float(result['adjacent_pair_count']), 0.)
            (result['motion_l1']+result['delta_l1']).backward()
            self.assertEqual(float(pred.grad.abs().sum()), 0.)
            self.assertIsNone(truth.grad)

    def test_only_true_rgb_adjacency_including_raw_token_drop(self):
        # Raw19->23 is adjacent in final RGB after the prescribed token drop.
        truth = self.clip([0., 0., 0., 0.])
        pred = self.clip([0., 0., 2., 2.]).requires_grad_()
        result = rgb_motion_losses(pred, truth, [0, 1, 4, 5], [3, 4, 5, 6])
        self.assertEqual(float(result['adjacent_pair_count']), 2.)
        self.assertEqual(float(result['delta_l1']), 0.)
        self.assertEqual(float(result['motion_l1']), 0.)
        truth = self.clip([0., 1.])
        pred = self.clip([0., .5]).requires_grad_()
        result = rgb_motion_losses(pred, truth, [118, 119], [19, 23])
        self.assertEqual(float(result['adjacent_pair_count']), 1.)
        self.assertEqual(float(result['delta_l1']), .5)
        self.assertEqual(float(result['motion_l1']), .25)

    def test_target_detached_and_diagnostics_detached(self):
        truth = self.clip([0., .5, 0.]).requires_grad_()
        pred = self.clip([0., .2, .1]).requires_grad_()
        result = rgb_motion_losses(pred, truth, [0, 1, 2], [3, 4, 5])
        (result['motion_l1']+result['delta_l1']).backward()
        self.assertIsNone(truth.grad)
        self.assertGreater(float(pred.grad.abs().sum()), 0.)
        for name in ('motion_roi_fraction', 'adjacent_pair_count', 'adjacent_pair_fraction'):
            self.assertFalse(result[name].requires_grad)


class ResidualWarpTests(unittest.TestCase):
    def test_perfect_dynamic_target_preserves_real_change_and_legacy_metric(self):
        truth = RGBMotionTests.clip([0., .3, .8]).requires_grad_()
        pred = truth.detach().clone().requires_grad_()
        flow = torch.zeros(2, 2, 2, 2, requires_grad=True)
        valid = torch.tensor([True, True])
        result = target_residual_ewarp(pred, truth, flow, valid)
        self.assertEqual(float(result['residual_ewarp_l1']), 0.)
        self.assertGreater(float(result['legacy_ewarp_l1']), 0.)
        old = legacy_losses(pred, truth, flow.detach(), valid, valid_frames=3)
        torch.testing.assert_close(result['legacy_ewarp_l1'], old['ewarp_l1'], atol=0, rtol=0)
        result['residual_ewarp_l1'].backward()
        self.assertIsNone(truth.grad)
        self.assertIsNone(flow.grad)
        self.assertEqual(float(pred.grad.abs().sum()), 0.)

    def test_nonzero_flow_and_ghost_gradient(self):
        truth = torch.zeros(1, 3, 2, 3, 4)
        truth[:, :, 0, :, 0] = 1.
        truth[:, :, 1, :, 1] = 1.
        pred = truth.clone()
        pred[:, :, 1] *= .5
        pred.requires_grad_()
        flow = torch.zeros(1, 2, 3, 4, requires_grad=True)
        with torch.no_grad():
            flow[:, 0] = -1.
        result = target_residual_ewarp(pred, truth.requires_grad_(), flow, torch.tensor([True]))
        self.assertGreater(float(result['residual_ewarp_l1']), 0.)
        self.assertEqual(float(result['valid_warp_fraction']), .75)
        result['residual_ewarp_l1'].backward()
        self.assertGreater(float(pred.grad.abs().sum()), 0.)
        self.assertIsNone(truth.grad)
        self.assertIsNone(flow.grad)

    def test_invalid_pairs_and_out_of_bounds_are_excluded(self):
        truth = RGBMotionTests.clip([0., 1., 0.])
        pred = RGBMotionTests.clip([0., .5, .8]).requires_grad_()
        flow = torch.zeros(2, 2, 2, 2)
        flow[1] = 100.
        result = target_residual_ewarp(pred, truth, flow, torch.tensor([False, True]))
        self.assertEqual(float(result['valid_warp_fraction']), 0.)
        self.assertEqual(float(result['residual_ewarp_l1']), 0.)
        self.assertEqual(float(result['legacy_ewarp_l1']), 0.)
        result['residual_ewarp_l1'].backward()
        self.assertEqual(float(pred.grad.abs().sum()), 0.)

    def test_legacy_parity_with_partial_masks_and_multiple_batch(self):
        torch.manual_seed(5)
        pred = torch.randn(2, 4, 4, 3, 4, requires_grad=True)
        truth = torch.randn_like(pred)
        flow = torch.randn(2, 3, 2, 3, 4)*.4
        valid = torch.tensor([[True, False, True], [False, True, True]])
        result = target_residual_ewarp(pred, truth, flow, valid)
        old = legacy_losses(pred, truth, flow, valid, valid_frames=4)
        for new_key, old_key in (('legacy_ewarp_l1', 'ewarp_l1'),
                                 ('valid_warp_fraction', 'valid_warp_fraction')):
            torch.testing.assert_close(result[new_key], old[old_key], atol=0, rtol=0)
        old_grad, = torch.autograd.grad(old['ewarp_l1'], pred, retain_graph=True)
        new_grad, = torch.autograd.grad(result['legacy_ewarp_l1'], pred)
        torch.testing.assert_close(new_grad, old_grad, atol=0, rtol=0)


class PixelRegressionTests(unittest.TestCase):
    def test_same_decoder_regions_baseline_values_and_gradients(self):
        torch.manual_seed(7)
        vae, lpips = FakeVAE(), FakeLPIPS()
        options = dict(latent_frames=12, real_frames=35,
                       latent_height=8, latent_width=20, checkpoint_decode=False)
        base = PixelSupervisor(vae, lpips, 'cpu', **options)
        motion = MotionPixelSupervisor(vae, lpips, 'cpu', **options)
        self.assertEqual(base.specs, motion.specs)
        self.assertEqual(base.full_specs, motion.full_specs)
        original = torch.randn(1, 24, 12, 8, 20)*.3
        truth = (torch.randn_like(original)*.3).requires_grad_()
        # First/last blocks and full/tile modes include the raw19->23 gap.
        for spec in (base.full_specs[0], base.full_specs[-1], base.specs[-1]):
            expected_pairs = sum(a+1 == b for a, b in
                zip(spec['rgb_frame_indices'], spec['rgb_frame_indices'][1:]))
            x0, x1 = original.clone().requires_grad_(), original.clone().requires_grad_()
            old, new = base.losses(x0, truth, spec), motion.losses(x1, truth, spec)
            for name in ('pixel_l1', 'lpips'):
                torch.testing.assert_close(old[name], new[name], atol=0, rtol=0)
            g0, = torch.autograd.grad(old['pixel_l1']+.2*old['lpips'], x0)
            g1, = torch.autograd.grad(new['pixel_l1']+.2*new['lpips'], x1, retain_graph=True)
            torch.testing.assert_close(g0, g1, atol=0, rtol=0)
            self.assertEqual(float(new['adjacent_pair_count']), expected_pairs)
            grad, = torch.autograd.grad(new['motion_l1']+new['delta_l1'], x1)
            self.assertGreater(float(grad.abs().sum()), 0.)
            self.assertTrue(bool(torch.isfinite(grad).all()))
        self.assertIsNone(truth.grad)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in vae.parameters()))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
