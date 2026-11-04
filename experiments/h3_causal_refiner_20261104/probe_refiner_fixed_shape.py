"""Investigate prefix disagreement and verify a fixed-shape causal execution path.

Retains the original variable-prefix failure as evidence. No optimization.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from fixed_horizon_refiner import FixedHorizonRefiner
from model_runtime import Runtime
from probe_refiner_causality import difference, SEED, NOISE_SEED
from temporal import self_forcing_rollout
from refiner_trace import attach_trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root, output = args.root, args.root / 'refiner_fixed_shape'
    rank, world = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    torch.npu.set_device(rank)
    device = torch.device(f'npu:{rank}')
    dist.init_process_group('hccl')
    started = time.perf_counter()
    try:
        exists = torch.tensor([int(output.exists()) if rank == 0 else 0], device=device)
        dist.broadcast(exists, src=0)
        if exists.item():
            raise FileExistsError(output)
        if rank == 0:
            output.mkdir(parents=True)
        dist.barrier()
        torch.manual_seed(SEED)
        torch.npu.manual_seed_all(SEED)
        runtime = Runtime(root / 'merged_transformer', root / 'assets', rank, world)
        runtime.model.eval()
        runtime.set_role('real')
        row = json.loads((root / 'data/holdout.jsonl').read_text().splitlines()[0])
        sample = torch.load(row['path'], map_location='cpu', weights_only=False)
        source = sample['upscaled'].to(device).float()
        if source.shape != (24, 13, 24, 42):
            raise ValueError('This controlled probe uses the existing 13-frame held-out input')
        noise = torch.randn(source.shape, generator=torch.Generator().manual_seed(NOISE_SEED)).to(device)
        fixed = FixedHorizonRefiner(runtime, 13)
        static_rows = sample['prompt_embeds'].shape[-2] + 2 * sample['audio_latents'].shape[-1]
        captures = {}
        handles = attach_trace(runtime.model, static_rows, captures)
        traces = {}
        checks = {}

        def save(label, value):
            if rank == 0:
                torch.save({'latent': value.cpu(), 'metadata': {**sample['metadata'],
                    'decoded_frames': 43, 'fps': 24.}}, output / f'{label}.pt')

        with torch.no_grad():
            sigma = .909375
            state = (1-sigma) * source + sigma * noise
            runtime.block_frames = 5
            full = state + sigma * runtime.forward(state, sample, sigma, causal=True, block_index=0)
            traces['full13'] = dict(captures)
            captures.clear()
            short = state[:, :5].contiguous()
            prefix = short + sigma * runtime.forward(short, sample, sigma, causal=True, block_index=0)
            traces['prefix5'] = dict(captures)
            for handle in handles:
                handle.remove()
            raw_prefix_difference = difference(full[:, :5], prefix)
            fixed_prefix = short + sigma * fixed.forward(short, sample, sigma, block_index=0)
            checks['block5_full13_vs_zero_future_prefix5'] = difference(full[:, :5], fixed_prefix)
            trace_records = []
            for name, long_value in traces['full13'].items():
                short_value = traces['prefix5'][name]
                value = difference(long_value['sample'], short_value['sample'])
                trace_records.append({'module': name, 'full_shape': long_value['shape'],
                    'prefix_shape': short_value['shape'], 'dtype': long_value['dtype'], **value})
            del traces, captures
            sigma = .3
            state = (1-sigma) * source + sigma * noise
            runtime.block_frames = 1
            full = state + sigma * runtime.forward(state, sample, sigma, causal=True, block_index=0)
            short = state[:, :1].contiguous()
            strict_first = short + sigma * fixed.forward(short, sample, sigma, block_index=0)
            checks['frame1_full13_vs_zero_future_prefix1'] = difference(full[:, :1], strict_first)
            rollouts = {}
            for block_frames in (5, 1):
                runtime.block_frames = block_frames
                before = time.perf_counter()
                result = self_forcing_rollout(source, sigma, block_frames,
                    lambda value, **kw: fixed.forward(value, sample, **kw),
                    noise=noise, grad_blocks='none')
                torch.npu.synchronize()
                label = f'fixed_block{block_frames}_sigma03'
                save(label, result.latent)
                rollouts[label] = {'seconds': time.perf_counter()-before,
                    'audit': result.audit, 'execution_horizon': 13,
                    'future_slots': 'zeros, masked from earlier video and static queries'}
                if block_frames == 1:
                    checks['fixed_frame1_first_rollout_replay'] = difference(result.latent[:, :1], strict_first)
            save('upscaled', source)
            save('reference', sample['reference'])
        for value in checks.values():
            value['passed'] = value['max_abs'] <= 1e-4
        report = {'completed': True, 'fixed_shape_checks_passed': all(v['passed'] for v in checks.values()),
            'sample_id': sample['metadata']['id'], 'noise_seed': NOISE_SEED,
            'raw_variable_prefix_difference': raw_prefix_difference,
            'shared_row_traces': trace_records,
            'first_observed_divergence': next((v for v in trace_records if not v['bit_exact']), None),
            'checks': checks, 'rollouts': rollouts, 'seconds': time.perf_counter()-started,
            'limitations': ['Variable-prefix mismatch is preserved, not waived.',
                'Fixed-shape execution is a controlled fixed-window prototype, not a KV cache or unbounded stream.',
                'Future slots contain only zeros and cannot be attended by earlier blocks.',
                'Full text/silence audio are fixed; upscaler and VAE remain noncausal.',
                'One clip and seed; no optimization or generalization evidence.']}
        if rank == 0:
            (output / 'summary.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({'fixed_shape_completed': report['fixed_shape_checks_passed'],
                             'checks': checks, 'first_divergence': report['first_observed_divergence']}), flush=True)
        dist.barrier()
        if not report['fixed_shape_checks_passed']:
            raise AssertionError('Fixed-shape equivalence checks failed')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
