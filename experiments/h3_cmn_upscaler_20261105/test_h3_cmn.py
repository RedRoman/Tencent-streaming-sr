"""CPU dependency/state/gradient tests for the independent CMN upsampler.

No model assets, decoder, optical flow, source dataset, or NPU are loaded.
The learned residual is activated for dependency tests so zero initialization
cannot make a missing-memory implementation pass accidentally.
"""
from __future__ import annotations

import unittest

import torch
from torch import nn
import torch.nn.functional as F

from h3_cmn import H3LatentUpsampler, detach_state, state_nbytes


class CMNUpsamplerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(719)

    def model(self, active=True):
        model = H3LatentUpsampler(width=8, blocks=2)
        if active:
            with torch.no_grad():
                model.output.weight.normal_(0., .04)
                model.output.bias.normal_(0., .01)
        return model

    def close(self, actual, expected):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def test_default_structure_and_no_temporal_normalization(self):
        model = H3LatentUpsampler()
        self.assertEqual(model.config['spatial_factors'], [1, 2, 1])
        self.assertEqual(model.config['temporal_factors'], [1, 1, 1])
        self.assertEqual(model.config['memory_init'], 'zero')
        self.assertEqual(len(model.stages), 3)
        self.assertEqual(sum(len(stage.blocks) for stage in model.stages), 6)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1061336)
        forbidden = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                     nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d,
                     nn.InstanceNorm3d, nn.Conv3d, nn.Dropout)
        self.assertFalse(any(isinstance(module, forbidden) for module in model.modules()))

    def test_initial_output_is_exact_bilinear_and_37_shape(self):
        model = self.model(active=False)
        x = torch.randn(2, 24, 37, 3, 5)
        output = model(x)
        b, c, t, h, w = x.shape
        expected = F.interpolate(x.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w),
                                 scale_factor=2., mode='bilinear', align_corners=False)
        expected = expected.reshape(b, t, c, 2*h, 2*w).permute(0, 2, 1, 3, 4)
        self.assertEqual(output.shape, (2, 24, 37, 6, 10))
        torch.testing.assert_close(output, expected, atol=0, rtol=0)

    def test_full_step_and_arbitrary_chunk_equivalence(self):
        model = self.model().eval()
        x = torch.randn(2, 24, 37, 3, 5)
        with torch.no_grad():
            full = model(x)
            state, frames = None, []
            for index in range(37):
                out, state = model.forward_step(x[:, :, index], state, index)
                frames.append(out)
            self.close(torch.cat(frames, dim=2), full)
            self.assertEqual(state.next_latent_index, 37)
            for chunks in ([37], [1]*37, [5]*7+[2], [7, 13, 17], [2, 3, 1, 31]):
                state, outputs, offset = None, [], 0
                for size in chunks:
                    out, state = model.forward_chunk(x[:, :, offset:offset+size], state, offset)
                    outputs.append(out)
                    offset += size
                self.close(torch.cat(outputs, dim=2), full)
                self.assertEqual(state.next_latent_index, 37)

    def test_future_perturbation_and_prefix_in_train_mode(self):
        model = self.model().train()
        x = torch.randn(1, 24, 13, 3, 5)
        full = model(x)
        for cut in (1, 5, 10):
            future = x.clone()
            future[:, :, cut:] = torch.randn_like(future[:, :, cut:])*10
            self.close(model(future)[:, :, :cut], full[:, :, :cut])
            self.close(model(x[:, :, :cut]), full[:, :, :cut])

    def test_loss_has_past_current_but_no_future_input_gradient(self):
        model = self.model().train()
        x = torch.randn(1, 24, 13, 3, 5, requires_grad=True)
        model(x)[:, :, 8].square().mean().backward()
        self.assertGreater(float(x.grad[:, :, 7].abs().max()), 0)
        self.assertGreater(float(x.grad[:, :, 8].abs().max()), 0)
        self.assertEqual(float(x.grad[:, :, 9:].abs().max()), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_sequence_vs_step_input_and_parameter_gradients(self):
        first = self.model().train()
        second = self.model().train()
        second.load_state_dict(first.state_dict())
        x = torch.randn(1, 24, 7, 3, 5, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        a = first(x)
        state, pieces = None, []
        for index in range(7):
            value, state = second.forward_step(y[:, :, index], state)
            pieces.append(value)
        b = torch.cat(pieces, dim=2)
        self.close(a, b)
        a.square().mean().backward()
        b.square().mean().backward()
        self.close(x.grad, y.grad)
        for p, q in zip(first.parameters(), second.parameters()):
            self.close(p.grad, q.grad)

    def test_zero_head_first_update_then_internal_gradients(self):
        model = self.model(active=False).train()
        opt = torch.optim.SGD(model.parameters(), lr=.1)
        x = torch.randn(1, 24, 5, 3, 5)
        target = torch.randn(1, 24, 5, 6, 10)
        first = model(x)
        (first-target).square().mean().backward()
        self.assertGreater(float(model.output.weight.grad.abs().max()), 0)
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        self.assertTrue(all(torch.count_nonzero(p.grad) == 0 for name, p in model.named_parameters() if not name.startswith('output.')))
        opt.step()
        opt.zero_grad(set_to_none=True)
        (model(x)-target).square().mean().backward()
        self.assertGreater(float(model.input.weight.grad.abs().max()), 0)
        for stage in model.stages:
            for block in stage.blocks:
                self.assertGreater(float(block.conv1.weight.grad.abs().max()), 0)

    def test_tbptt_detaches_history_without_changing_forward(self):
        model = self.model().train()
        past = torch.randn(1, 24, 5, 3, 5, requires_grad=True)
        current = torch.randn(1, 24, 2, 3, 5, requires_grad=True)
        _, state = model.forward_chunk(past)
        attached, _ = model.forward_chunk(current, state)
        attached.square().mean().backward(retain_graph=True)
        self.assertGreater(float(past.grad.abs().max()), 0)
        past.grad = None
        detached = model.detach_state(state)
        self.assertEqual(detached.next_latent_index, 5)
        self.assertTrue(all(not item.requires_grad and item.grad_fn is None for item in detached.memories))
        output, _ = model.forward_chunk(current, detached)
        self.close(output, attached)
        output.square().mean().backward()
        self.assertIsNone(past.grad)
        self.assertIsNone(detach_state(None))

    def test_fifth_token_boundary_does_not_reset_memory(self):
        model = self.model().eval()
        x = torch.randn(1, 24, 7, 3, 5)
        with torch.no_grad():
            full = model(x)
            _, state = model.forward_chunk(x[:, :, :5])
            continued, _ = model.forward_chunk(x[:, :, 5:], state)
            reset, _ = model.forward_chunk(x[:, :, 5:], None, latent_index=5)
        self.close(continued, full[:, :, 5:])
        self.assertGreater(float((continued[:, :, 0]-reset[:, :, 0]).abs().max()), 1e-6)

    def test_state_batch_geometry_dtype_owner_and_index_validation(self):
        model = self.model()
        x = torch.randn(2, 24, 3, 5)
        _, state = model.forward_step(x, latent_index=10)
        self.assertEqual(state.next_latent_index, 11)
        invalid = [(x[:1], state, 11), (x[:, :, :2], state, 11),
                   (x.double(), state, 11), (x, state, 10)]
        for value, cache, index in invalid:
            with self.assertRaises(ValueError):
                model.forward_step(value, cache, index)
        with self.assertRaises(ValueError):
            self.model().forward_step(x, state)
        with self.assertRaises(ValueError):
            model.forward_step(x, latent_index=-1)
        with self.assertRaises(ValueError):
            model.forward_chunk(torch.empty(2, 24, 0, 3, 5))

    def test_reset_batch_independence_and_state_immutability(self):
        model = self.model().eval()
        x = torch.randn(2, 24, 7, 3, 5)
        with torch.no_grad():
            batched = model(x)
            self.close(batched, torch.cat([model(x[i:i+1]) for i in range(2)]))
            _, state = model.forward_chunk(x[:, :, :5])
            saved = tuple(value.clone() for value in state.memories)
            model.forward_chunk(x[:, :, 5:], state)
            self.assertEqual(state.next_latent_index, 5)
            for before, after in zip(saved, state.memories):
                torch.testing.assert_close(before, after, atol=0, rtol=0)
            self.close(model(x), batched)

    def test_inference_cache_owns_only_last_frame_storage(self):
        model = H3LatentUpsampler().eval()
        with torch.no_grad():
            _, state = model.forward_chunk(torch.randn(1, 24, 3, 12, 21))
        expected_bytes = (4*64*12*21 + 2*64*24*42)*4
        self.assertEqual(state_nbytes(state), expected_bytes)
        self.assertTrue(all(value.untyped_storage().nbytes() == value.numel()*value.element_size() for value in state.memories))
        self.assertEqual(state_nbytes(None), 0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
