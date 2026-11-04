"""Tiny two-NPU test of the actual runtime's mixed-dtype FSDP2/role modules.

No H3 weights are loaded. Run under locked_run.py and torchrun on idle devices.
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.utils.checkpoint import checkpoint

from model_runtime import RoleLinear, shard_homogeneous


class TinyBlock(nn.Module):
    def __init__(self, state):
        super().__init__()
        base = nn.Linear(16, 16, bias=False, dtype=torch.bfloat16).requires_grad_(False)
        self.projection = RoleLinear(base, 4, state)
        self.register_buffer('curve_a', torch.randn(4, 8, dtype=torch.float32) * .01)
        self.register_buffer('curve_b', torch.randn(16, 4, dtype=torch.float32) * .01)

    def forward(self, x, time_coords):
        # Mimics Sharp's FP32 curve buffers inside a BF16 block.
        curve = F.linear(F.linear(time_coords.float(), self.curve_a), self.curve_b)
        return x + torch.tanh(self.projection(x) + curve.to(x.dtype))


class TinyModel(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.input = nn.Linear(16, 16, dtype=torch.float32).requires_grad_(False)
        self.blocks = nn.ModuleList([TinyBlock(state), TinyBlock(state)])
        self.output = nn.Linear(16, 16, dtype=torch.float32).requires_grad_(False)

    def forward(self, x, time_coords):
        x = self.input(x.float()).bfloat16()
        for block in self.blocks:
            if torch.is_grad_enabled():
                x = checkpoint(block, x, time_coords, use_reentrant=False)
            else:
                x = block(x, time_coords)
        return self.output(x.float())


def materialized(model, role=None, frozen=False):
    result = {}
    for name, parameter in model.named_parameters():
        if (frozen and not parameter.requires_grad) or (role and f'.{role}.' in name):
            value = parameter.detach()
            if hasattr(value, 'full_tensor'):
                value = value.full_tensor()
            result[name] = value.float().cpu().clone()
    return result


def max_change(before, after):
    return max(float((after[k] - v).abs().max()) for k, v in before.items())


def main():
    started = time.monotonic()
    rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    assert world == 2
    torch.npu.set_device(rank)
    dist.init_process_group('hccl')
    torch.manual_seed(901)
    state = SimpleNamespace(role='real')
    model = TinyModel(state)
    mesh = init_device_mesh('npu', (world,))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                  cast_forward_inputs=False)
    for block in model.blocks:
        shard_homogeneous(block, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    root_policy = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32,
                                       cast_forward_inputs=False)
    shard_homogeneous(model, mesh=mesh, mp_policy=root_policy, reshard_after_forward=True)
    model.eval()
    parameters = {role: [p for name, p in model.named_parameters()
                         if p.requires_grad and f'.{role}.' in name] for role in ('generator', 'fake')}
    assert all(parameters.values())
    optimizers = {role: torch.optim.AdamW(ps, lr=.02, weight_decay=0., foreach=False)
                  for role, ps in parameters.items()}
    device = torch.device(f'npu:{rank}')
    x = torch.randn(3, 16, generator=torch.Generator().manual_seed(902 + rank)).to(device)
    coords = torch.randn(3, 8, generator=torch.Generator().manual_seed(903 + rank)).to(device)
    frozen_before = materialized(model, frozen=True)
    g_before, f_before = materialized(model, 'generator'), materialized(model, 'fake')
    with torch.no_grad():
        real_before = model(x, coords)
        state.role = 'generator'
        generator_initial = model(x, coords)
        state.role = 'fake'
        fake_initial = model(x, coords)
    torch.testing.assert_close(real_before, generator_initial, rtol=0, atol=0)
    torch.testing.assert_close(real_before, fake_initial, rtol=0, atol=0)
    records = []
    for role in ('fake', 'generator'):
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        # Scores before graph-building forward, then no role/coordinate changes
        # until checkpointed backward has completed.
        with torch.no_grad():
            state.role = 'real'
            teacher = model(x, coords)
            state.role = 'fake'
            fake = model(x, coords)
        state.role = role
        prediction = model(x, coords)
        if role == 'fake':
            loss = (prediction - (teacher + .7)).square().mean()
        else:
            gap = fake - teacher
            assert bool((gap.abs().max() > 0).item()), 'fake never moved from teacher'
            gradient = gap / (generator_initial - teacher).abs().mean().clamp_min(.1)
            loss = .5 * (prediction - (prediction - gradient).detach()).square().mean()
        loss.backward()
        other = 'generator' if role == 'fake' else 'fake'
        assert all(p.grad is None for p in parameters[other]), f'{role} touched {other}'
        grad = torch.nn.utils.clip_grad_norm_(parameters[role], 1.)
        if hasattr(grad, 'full_tensor'):
            grad = grad.full_tensor()
        grad_value = float(grad.item())
        assert 0 < grad_value < float('inf'), grad_value
        optimizers[role].step()
        optimizers[role].zero_grad(set_to_none=True)
        records.append({'role': role, 'loss': float(loss.detach()), 'grad_norm': grad_value})
        if role == 'fake':
            assert max_change(g_before, materialized(model, 'generator')) == 0.
    g_delta, f_delta = max_change(g_before, materialized(model, 'generator')), max_change(f_before, materialized(model, 'fake'))
    frozen_delta = max_change(frozen_before, materialized(model, frozen=True))
    assert g_delta > 0 and f_delta > 0 and frozen_delta == 0.
    with torch.no_grad():
        state.role = 'real'
        real_after = model(x, coords)
    torch.testing.assert_close(real_before, real_after, rtol=0, atol=0)
    torch.npu.synchronize()
    result = {'passed': True, 'rank': rank, 'world': world, 'seconds': time.monotonic() - started,
              'completed_utc': datetime.now(timezone.utc).isoformat(), 'updates': records,
              'generator_parameter_max_delta': g_delta, 'fake_parameter_max_delta': f_delta,
              'frozen_parameter_max_delta': frozen_delta, 'real_forward_max_delta': float((real_after-real_before).abs().max()),
              'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
              'peak_npu_bytes': torch.npu.max_memory_allocated()}
    dist.barrier()
    print(json.dumps(result), flush=True)
    if rank == 0:
        root = Path(os.environ.get('EXP_ROOT', '.'))
        destination = root / 'outputs/fsdp_roles_preflight.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
