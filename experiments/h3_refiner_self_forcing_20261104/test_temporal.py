"""Small CPU-compatible checks; run on the authorized remote PyTorch host.

    python -m unittest discover -s experiment -p test_temporal.py -v
"""
import unittest
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from temporal import H3BlockCausalProcessor, build_temporal_plan, self_forcing_rollout


def layout(frames=3, padding=0):
    # One text row, one fixed audio row, then one video token per frame.
    return SimpleNamespace(token_tags=torch.tensor([1, 2] + [0] * (frames + padding)),
                           text_indices=torch.tensor([0]), audio_indices=torch.tensor([1]),
                           video_indices=torch.arange(2, 2 + frames))


def plan(frames=3, padding=0, sigma=0.7):
    return build_temporal_plan(torch.zeros(1, frames, 1, 1), layout(frames, padding),
                               block_frames=1, current_block=frames - 1, sigma=sigma,
                               patch_size=(1, 1, 1))


class ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 2
        self.fused_projections = False
        self.to_q, self.to_k, self.to_v = (nn.Linear(4, 4, bias=False) for _ in range(3))
        self.norm_q = self.norm_k = nn.Identity()
        self.to_out = nn.ModuleList([nn.Linear(4, 4, bias=True), nn.Identity()])
        with torch.no_grad():
            for layer in (self.to_q, self.to_k, self.to_v, self.to_out[0]):
                layer.weight.copy_(torch.eye(4))
            self.to_out[0].bias.fill_(0.1)


class TemporalTests(unittest.TestCase):
    def test_row_times_and_padding(self):
        p = plan(padding=3)
        actual = p.timesteps[p.timestep_indices]
        torch.testing.assert_close(actual, torch.tensor([1., 1., 1., 1., .3, 1., 1., 1.]))
        self.assertEqual(p.row_blocks.tolist(), [-1, -1, 0, 1, 2, -2, -2, -2])
        self.assertEqual(p.valid_rows, 5)

    def test_no_future_leak_including_static_paths(self):
        torch.manual_seed(8)
        attn, p = ToyAttention(), plan()
        processor = H3BlockCausalProcessor(p, detach_history=False, query_chunk_size=1)
        value = torch.randn(1, 5, 4)
        poison = value.clone()
        poison[:, 4] = torch.tensor([40., -30., 20., -10.])
        # Two layers expose an indirect future -> static -> history leak.
        def two_layers(x):
            x = x + processor(attn, x)
            return x + processor(attn, x)
        result, altered = two_layers(value), two_layers(poison)
        torch.testing.assert_close(result[:, :4], altered[:, :4], atol=0, rtol=0)
        self.assertGreater(float((result[:, 4] - altered[:, 4]).abs().max()), 1.)
        processor.mode = "dense"
        self.assertGreater(float((two_layers(value)[:, :4] - two_layers(poison)[:, :4]).abs().max()), .1)

    def test_history_kv_gradient_is_detached(self):
        torch.manual_seed(2)
        attn, p = ToyAttention(), plan()
        source = torch.randn(1, 5, 4)
        for detach in (True, False):
            x = source.clone().requires_grad_(True)
            proc = H3BlockCausalProcessor(p, detach_history=detach)
            y = x + proc(attn, x)
            y = y + proc(attn, y)
            y[:, 4].square().sum().backward()
            history_gradient = float(x.grad[:, 2:4].abs().sum())
            self.assertEqual(history_gradient, 0.) if detach else self.assertGreater(history_gradient, 0.)
            self.assertGreater(float(x.grad[:, 4].abs().sum()), 0.)
            self.assertGreater(float(x.grad[:, :2].abs().sum()), 0.)

    def test_dense_sdpa_parity_and_padding_exclusion(self):
        torch.manual_seed(4)
        p, attn = plan(padding=3), ToyAttention()
        x = torch.randn(1, 8, 4)
        x[:, 5:] = 1000.
        proc = H3BlockCausalProcessor(p, mode="dense", detach_history=False, query_chunk_size=2)
        result = proc(attn, x)
        q, k, v = [fn(x[:, :5]).unflatten(-1, (2, 2)).transpose(1, 2)
                   for fn in (attn.to_q, attn.to_k, attn.to_v)]
        expected = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2, 3)
        expected = attn.to_out[0](expected)
        torch.testing.assert_close(result[:, :5], expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(float(result[:, 5:].abs().sum()), 0.)

    def test_rollout_uses_own_history_with_truncated_gradient(self):
        parameter = nn.Parameter(torch.tensor(2.))
        seen = []
        def callback(prefix, *, block_index, current_start, current_end, sigma):
            seen.append(prefix.detach().clone())
            return torch.ones_like(prefix) * (parameter + prefix[:, :current_start].sum())
        source = torch.zeros(1, 2, 1, 1)
        result = self_forcing_rollout(source, 1., 1, callback, noise=torch.ones_like(source))
        torch.testing.assert_close(result.latent.flatten(), torch.tensor([3., 6.]))
        self.assertEqual(seen[1][:, :1].item(), 3.)  # generated output, not draft 0 or noise 1
        result.latent.sum().backward()
        self.assertEqual(parameter.grad.item(), 1.)  # no derivative through history
        self.assertEqual(result.audit["model_forwards"], 2)
        self.assertEqual(result.audit["gradient_blocks"], [1])
        self.assertTrue(result.audit["history_detached"])
        self.assertEqual(result.audit["future_frames_exposed"], 0)

    def test_all_blocks_keep_local_gradient_but_not_history_graph(self):
        parameter = nn.Parameter(torch.tensor(2.))
        def callback(prefix, *, current_start, **kwargs):
            return torch.ones_like(prefix) * (parameter + prefix[:, :current_start].sum())
        source = torch.zeros(1, 2, 1, 1)
        result = self_forcing_rollout(source, 1., 1, callback, noise=torch.ones_like(source), grad_blocks="all")
        result.latent.sum().backward()
        self.assertEqual(parameter.grad.item(), 2.)

    def test_partial_last_block_and_invalid_geometry(self):
        source = torch.zeros(1, 5, 1, 1)
        result = self_forcing_rollout(source, .5, 2, lambda x, **kw: torch.ones_like(x),
                                     noise=torch.zeros_like(source), grad_blocks="none")
        torch.testing.assert_close(result.latent, torch.full_like(source, .5))
        self.assertEqual([r["prefix_frames"] for r in result.audit["block_records"]], [2, 4, 5])
        with self.assertRaises(ValueError):
            build_temporal_plan(source, layout(4), block_frames=1, current_block=0,
                                sigma=.7, patch_size=(1, 1, 1))


if __name__ == "__main__":
    unittest.main()
