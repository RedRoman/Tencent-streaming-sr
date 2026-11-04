"""Short, auditable temporal Self Forcing + DMD smoke experiment for H3 refiner.

This is a new causal student adapter, not a released one-step refiner checkpoint.
The score teacher is the released dense refiner, and the fake score is trained
with flow matching on detached on-policy student samples.
"""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch_npu  # registers NPU backend
import torch.distributed as dist

from model_runtime import Runtime
from temporal import self_forcing_rollout
from open_h3.sigmas import SCHEDULES


def read_samples(path):
    return [torch.load(row['path'], map_location='cpu', weights_only=False)
            for row in (json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())]


def norm(params):
    result = torch.nn.utils.clip_grad_norm_(params, 1.0)
    if hasattr(result, 'full_tensor'):
        result = result.full_tensor()
    value = float(result.item())
    if not math.isfinite(value):
        raise RuntimeError(f'Nonfinite gradient norm: {value}')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--adapter-rank', type=int, default=8)
    parser.add_argument('--block-frames', type=int, default=5)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--fake-warmup', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--sigma', type=float, default=0.909375)
    parser.add_argument('--seed', type=int, default=20261104)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()
    root = Path(args.root)
    rank, world = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    torch.npu.set_device(rank)
    dist.init_process_group('hccl')
    device = torch.device(f'npu:{rank}')
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    started = time.time()
    out = root / 'outputs'
    out.mkdir(parents=True, exist_ok=True)
    def log(event, **values):
        record = dict(event=event, seconds=round(time.time()-started, 3), **values)
        if rank == 0:
            text = json.dumps(record, ensure_ascii=False)
            print(text, flush=True)
            with (out / 'train.jsonl').open('a') as stream:
                stream.write(text+'\n')
    log('start', **vars(args), world=world, hostname=os.environ.get("H3_RUN_LABEL", "unspecified"),
        physical_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'))
    runtime = Runtime(root/'merged_transformer', root/'assets', rank, world,
                      adapter_rank=args.adapter_rank, block_frames=args.block_frames)
    runtime.model.eval()  # no dropout; gradient checkpointing remains enabled
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        runtime.load_adapter_state(checkpoint['adapters'])
        del checkpoint
        log('checkpoint_loaded', path=args.checkpoint)
    optimizers = {role: torch.optim.AdamW(params, lr=args.learning_rate, betas=(0.9, 0.99),
                                        eps=1e-8, weight_decay=0.0, foreach=False)
                  for role, params in runtime.params.items()}
    train = read_samples(root/'data/train.jsonl')
    holdout = read_samples(root/'data/holdout.jsonl')[0]
    log('model_ready', trainable={role: sum(p.numel() for p in params) for role, params in runtime.params.items()},
        train_ids=[s['metadata']['id'] for s in train], holdout=holdout['metadata']['id'],
        npu_allocated_gb=torch.npu.memory_allocated()/1e9)

    def frozen_fingerprint():
        checks = {}
        for name, value in list(runtime.model.named_parameters()) + list(runtime.model.named_buffers()):
            if not (name.endswith('attn.to_q.base.weight') or name.endswith('curve_b')):
                continue
            local = value.detach().to_local() if hasattr(value, 'to_local') else value.detach()
            # Bounded sampling is an invariance check, not a whole-checkpoint hash.
            raw = local.flatten()[:256].cpu().contiguous().view(torch.uint8).numpy().tobytes()
            checks[name] = hashlib.sha256(raw).hexdigest()
        assert checks
        return checks
    frozen_before = frozen_fingerprint()

    def rollout(sample, noise, *, grad=False, frames=None):
        runtime.set_role('generator')
        source = sample['upscaled'].to(device).float()
        if frames is not None:
            source, noise = source[:, :frames], noise[:, :frames]
        def velocity(prefix, **kw):
            return runtime.forward(prefix, sample, causal=True, **kw)
        return self_forcing_rollout(source, args.sigma, args.block_frames, velocity,
                                    noise=noise, grad_blocks='last' if grad else 'none')

    def noise_for(sample, seed):
        # CPU seeded stream makes exact replay independent of prior NPU operators.
        return torch.randn(sample['upscaled'].shape, generator=torch.Generator().manual_seed(seed)).to(device)

    def write_latent(name, latent, sample, metadata=None):
        if rank == 0:
            torch.save({'latent': latent.detach().cpu(), 'metadata': sample['metadata'],
                        'experiment': metadata or {}}, out/f'{name}.pt')

    if args.eval_only:
        if not args.checkpoint:
            raise ValueError('--eval-only requires --checkpoint')
        with torch.no_grad():
            result = rollout(holdout, noise_for(holdout, args.seed+991))
        write_latent('causal_one_reloaded', result.latent, holdout, result.audit)
        log('reloaded_inference_completed', audit=result.audit)
        dist.barrier()
        dist.destroy_process_group()
        return

    @torch.no_grad()
    def baselines():
        sample = holdout
        source = sample['upscaled'].to(device).float()
        noise = noise_for(sample, args.seed+991)
        write_latent('reference', sample['reference'], sample)
        write_latent('upscaled', source, sample)
        runtime.set_role('real')
        state = (1-args.sigma)*source+args.sigma*noise
        dense_one = state + args.sigma*runtime.forward(state, sample, args.sigma)
        assert bool(torch.isfinite(dense_one).all()), 'Nonfinite dense one-step baseline'
        write_latent('dense_one_untrained', dense_one, sample, {'sigmas':[args.sigma,0], 'causal':False})
        log('baseline_dense_one', rms=float(dense_one.square().mean().sqrt()))
        schedule = SCHEDULES[3]
        state = (1-schedule[0])*source+schedule[0]*noise
        for sigma, next_sigma in zip(schedule, schedule[1:]):
            state = state + (sigma-next_sigma)*runtime.forward(state, sample, sigma)
            assert bool(torch.isfinite(state).all()), 'Nonfinite dense three-step baseline'
            log('baseline_dense_three_step', sigma=sigma, next_sigma=next_sigma,
                finite=bool(torch.isfinite(state).all()))
        write_latent('dense_three', state, sample, {'sigmas':schedule, 'causal':False})
        result = rollout(sample, noise)
        write_latent('causal_one_untrained', result.latent, sample, result.audit)
        log('baseline_causal_one', audit=result.audit, rms=float(result.latent.square().mean().sqrt()))

    baselines()
    log('baselines_complete')

    def fake_step(sample, generated, seed):
        optimizers['fake'].zero_grad(set_to_none=True)
        optimizers['generator'].zero_grad(set_to_none=True)
        generator = torch.Generator().manual_seed(seed)
        sigma = 0.1 + 0.8*float(torch.rand((), generator=generator))
        noise = torch.randn(generated.shape, generator=generator).to(device)
        target = generated.detach().float()-noise
        noisy = (1-sigma)*generated.detach().float()+sigma*noise
        runtime.set_role('fake')
        velocity = runtime.forward(noisy, sample, sigma, causal=False)
        loss = F_mse(velocity, target)
        loss.backward()
        gradient = norm(runtime.params['fake'])
        optimizers['fake'].step()
        optimizers['fake'].zero_grad(set_to_none=True)
        return dict(fake_loss=float(loss.detach()), fake_grad_norm=gradient, fake_sigma=sigma)

    for step in range(args.fake_warmup):
        sample = train[step % len(train)]
        noise = noise_for(sample, args.seed+rank*100000+step)
        generated = rollout(sample, noise).latent
        values = fake_step(sample, generated, args.seed+rank*100000+1000+step)
        log('fake_warmup', step=step, **values)
        del generated, noise

    for step in range(args.steps):
        sample = train[step % len(train)]
        blocks = math.ceil(sample['upscaled'].shape[1]/args.block_frames)
        trained_block = step % blocks
        frames = min((trained_block+1)*args.block_frames, sample['upscaled'].shape[1])
        noise = noise_for(sample, args.seed+rank*100000+10000+step)
        generated_result = rollout(sample, noise, frames=frames)
        generated = generated_result.latent.detach()
        values = fake_step(sample, generated, args.seed+rank*100000+20000+step)
        rng = torch.Generator().manual_seed(args.seed+rank*100000+30000+step)
        sigma = 0.1+0.8*float(torch.rand((), generator=rng))
        score_noise = torch.randn(generated.shape, generator=rng).to(device)
        noisy = (1-sigma)*generated+sigma*score_noise
        with torch.no_grad():
            runtime.set_role('real')
            real_x0 = noisy+sigma*runtime.forward(noisy, sample, sigma)
            runtime.set_role('fake')
            fake_x0 = noisy+sigma*runtime.forward(noisy, sample, sigma)
            denominator = (generated-real_x0).abs().mean().clamp_min(1e-5)
            dmd = (fake_x0-real_x0)/denominator
            if not bool(torch.isfinite(dmd).all()):
                raise RuntimeError('Nonfinite DMD gradient')
            score_gap = float((fake_x0-real_x0).square().mean().sqrt())
            assert score_gap > 0, 'Fake and real scores have not diverged after warmup'
        optimizers['generator'].zero_grad(set_to_none=True)
        optimizers['fake'].zero_grad(set_to_none=True)
        recomputed = rollout(sample, noise, grad=True, frames=frames)
        replay_error = float((recomputed.latent.detach()-generated).abs().max())
        assert replay_error < 1e-4, f'Non-deterministic rollout replay: {replay_error}'
        last_start = trained_block*args.block_frames
        prediction = recomputed.latent[:, last_start:]
        target = (prediction-dmd[:, last_start:]).detach()
        loss = 0.5*F_mse(prediction, target)
        # Nothing changes role, time plan, or curve coordinates before backward.
        loss.backward()
        gradient = norm(runtime.params['generator'])
        assert gradient > 0, 'Generator receives no gradient'
        assert all(p.grad is None for p in runtime.params['fake']), 'G update touched fake parameters'
        optimizers['generator'].step()
        optimizers['generator'].zero_grad(set_to_none=True)
        log('generator_update', step=step+1, sample=sample['metadata']['id'], trained_block=trained_block,
            prefix_frames=frames, dmd_loss=float(loss.detach()), generator_grad_norm=gradient,
            score_gap_rms=score_gap, dmd_denominator=float(denominator), score_sigma=sigma,
            replay_max_abs=replay_error, audit=generated_result.audit, **values)
        del recomputed, generated, generated_result, prediction, target, loss, real_x0, fake_x0, dmd

    state = runtime.adapter_state()
    assert frozen_fingerprint() == frozen_before, 'Frozen base/Sharp samples changed during training'
    log('frozen_invariance_passed', sampled_tensors=len(frozen_before), sampled_values_per_tensor=256)
    if rank == 0:
        torch.save({'adapters':state, 'args':vars(args)}, out/'adapters.pt')
        log('checkpoint_saved', path=str(out/'adapters.pt'),
            generator_b_rms=float(torch.cat([v.flatten() for k,v in state.items() if '.generator.b.' in k]).square().mean().sqrt()),
            fake_b_rms=float(torch.cat([v.flatten() for k,v in state.items() if '.fake.b.' in k]).square().mean().sqrt()))
    del state
    with torch.no_grad():
        result = rollout(holdout, noise_for(holdout, args.seed+991))
        assert bool(torch.isfinite(result.latent).all())
        write_latent('causal_one_trained', result.latent, holdout, result.audit)
        log('final_inference', audit=result.audit, rms=float(result.latent.square().mean().sqrt()))
    dist.barrier()
    log('completed', generator_steps=args.steps, fake_updates=args.fake_warmup+args.steps,
        rank0_peak_npu_memory_gb=torch.npu.max_memory_allocated()/1e9)
    dist.destroy_process_group()


def F_mse(a, b):
    return (a.float()-b.float()).square().mean()


if __name__ == '__main__':
    main()
