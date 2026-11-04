"""Validate the completed smoke run and describe latent changes without quality claims."""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    out = Path(args.root) / 'outputs'
    records = [json.loads(line) for line in (out / 'train.jsonl').read_text().splitlines() if line.strip()]
    completed = [i for i, row in enumerate(records) if row['event'] == 'completed']
    if not completed:
        raise RuntimeError('No completed training run in train.jsonl')
    end = completed[-1]
    start = max(i for i in range(end + 1) if records[i]['event'] == 'start')
    run = records[start:end + 1]
    updates = [row for row in run if row['event'] == 'generator_update']
    assert len(updates) == run[-1]['generator_steps'] > 0
    assert all(row['generator_grad_norm'] > 0 and row['fake_grad_norm'] > 0 and
               row['score_gap_rms'] > 0 and row['replay_max_abs'] < 1e-4 for row in updates)
    assert any(row['event'] == 'frozen_invariance_passed' for row in run)
    names = ['reference', 'upscaled', 'dense_one_untrained', 'dense_three',
             'causal_one_untrained', 'causal_one_trained']
    latents = {name: torch.load(out / f'{name}.pt', map_location='cpu', weights_only=False)['latent'].float()
               for name in names}
    shape = tuple(latents['reference'].shape)
    assert all(tuple(value.shape) == shape and torch.isfinite(value).all() for value in latents.values())
    before, after = latents['causal_one_untrained'], latents['causal_one_trained']
    difference = after - before
    assert difference.abs().max() > 0, 'Training did not change holdout inference'
    result = {
        'completed': True,
        'host_world_size': records[start]['world'],
        'generator_updates': len(updates),
        'fake_updates': run[-1]['fake_updates'],
        'elapsed_seconds': run[-1]['seconds'],
        'rank0_peak_npu_memory_gb': run[-1]['rank0_peak_npu_memory_gb'],
        'latent_shape': list(shape),
        'replay_max_abs': max(row['replay_max_abs'] for row in updates),
        'trained_blocks': [row['trained_block'] for row in updates],
        'generator_grad_norms': [row['generator_grad_norm'] for row in updates],
        'fake_grad_norms': [row['fake_grad_norm'] for row in updates],
        'score_gap_rms': [row['score_gap_rms'] for row in updates],
        'holdout_before_after': {
            'rms_difference': float(difference.square().mean().sqrt()),
            'max_abs_difference': float(difference.abs().max()),
            'relative_rms_difference': float(difference.square().mean().sqrt() / before.square().mean().sqrt()),
        },
        'latent_diagnostics': {name: {'rms': float(value.square().mean().sqrt()),
                                    'mse_to_reference': float((value-latents['reference']).square().mean()),
                                    'adjacent_frame_rms': float(value.diff(dim=1).square().mean().sqrt())}
                               for name, value in latents.items()},
        'interpretation_limits': [
            'Six generator updates on two clips test the pipeline, not convergence.',
            'Latent reference distance and adjacent-frame difference are diagnostics, not perceptual quality scores.',
            'One denoising step per causal block; three blocks require three model evaluations plus prefix recomputation.',
            'Official latent upscaler and VAE are temporally noncausal; only the refiner is block causal.',
            'Adapters checkpoint supports inference reload; optimizer and RNG states are not saved.',
        ],
    }
    reload_path = out / 'causal_one_reloaded.pt'
    if reload_path.exists():
        reloaded = torch.load(reload_path, map_location='cpu', weights_only=False)['latent'].float()
        assert reloaded.shape == after.shape and torch.isfinite(reloaded).all()
        error = float((after-reloaded).abs().max())
        assert error < 1e-4, f'Checkpoint reload changed inference: {error}'
        result['checkpoint_reload_max_abs'] = error
    (out / 'result_summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
