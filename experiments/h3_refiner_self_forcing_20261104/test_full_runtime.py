"""Full-size Ascend compatibility check using the existing H3 base checkpoint.

This is NOT a refiner quality result and does not write trained model weights.
It checks actual long-sequence causal forward/backward while official refiner
assets are downloading. Test outputs are not reused by the real experiment.
"""
import json
import os
import time
from pathlib import Path

import torch
import torch_npu
import torch.distributed as dist

from model_runtime import Runtime
from temporal import self_forcing_rollout


def main():
    rank, world = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    torch.npu.set_device(rank)
    torch.set_num_threads(4)
    dist.init_process_group('hccl')
    torch.manual_seed(20261104)
    root = Path(os.environ['EXP_ROOT'])
    started = time.time()
    def log(event, **values):
        if rank == 0:
            print(json.dumps({'event':event, 'seconds':time.time()-started, **values}), flush=True)
    log('base_preflight_start', world=world, released_refiner_assets_used=False)
    runtime = Runtime(Path(os.environ['H3_BASE_MODEL']) / 'transformer',
                      root/'assets', rank, world, base_preflight=True)
    runtime.model.eval()
    log('base_model_loaded', allocated_gb=torch.npu.memory_allocated()/1e9)
    sample = torch.load(root/'data/samples/VIDGEN5S_0380588.pt',map_location='cpu',weights_only=False)
    source = sample['upscaled'].to(runtime.device).float()
    noise = torch.randn(source.shape,generator=torch.Generator().manual_seed(42)).to(runtime.device)
    runtime.set_role('generator')
    def forward(prefix, **kw):
        return runtime.forward(prefix,sample,causal=True,**kw)
    with torch.no_grad():
        first = self_forcing_rollout(source,0.909375,5,forward,noise=noise,grad_blocks='none')
    assert torch.isfinite(first.latent).all()
    log('full_causal_forward', audit=first.audit, rms=float(first.latent.square().mean().sqrt()),
        peak_allocated_gb=torch.npu.max_memory_allocated()/1e9)
    second = self_forcing_rollout(source,0.909375,5,forward,noise=noise,grad_blocks='last')
    replay = float((first.latent-second.latent.detach()).abs().max())
    loss = second.latent[:,10:].square().mean()
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(runtime.params['generator'], 1.0)
    if hasattr(grad,'full_tensor'):
        grad = grad.full_tensor()
    assert torch.isfinite(grad) and grad > 0
    assert all(p.grad is None for p in runtime.params['fake'])
    evidence = {'purpose':'base model compatibility preflight only; not the released refiner',
                'world':world,'shape':list(source.shape),'gradient_norm':float(grad),
                'loss':float(loss.detach()),'replay_max_abs':replay,
                'peak_allocated_gb':torch.npu.max_memory_allocated()/1e9,
                'elapsed':time.time()-started, 'audit':second.audit}
    if rank==0:
        (root/'outputs').mkdir(exist_ok=True)
        (root/'outputs/full_runtime_preflight.json').write_text(json.dumps(evidence,indent=2))
    log('base_preflight_passed', **evidence)
    dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':
    main()
