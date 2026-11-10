"""Summarize immutable diagnostic receipts without accelerator or weight access."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(paths):
    samples = []
    seen = set()
    checkpoints = set()
    for path in paths:
        raw = path.read_bytes()
        data = json.loads(raw)
        sid = data['sample_id']
        if sid in seen:
            raise ValueError('Duplicate sample')
        seen.add(sid)
        checkpoints.add(data['checkpoint_sha256'])
        assert data['new_optimizer_updates'] == 0
        assert len(data['rows']) == 36 and len(data['history_differences']) == 24
        assert len(data['four_latent_full_prefix_oracle_parity']) == 1
        assert data['four_latent_full_prefix_oracle_parity'][0]['max_abs'] == 0
        assert all(r['difference']['max_abs'] == 0 for r in data['history_differences'] if r['position'] == 0)
        rows = { (r['mode'], r['position']): r for r in data['rows'] if 'kind' in r }
        replay = [r['saved_self_history_replay']['max_abs'] for r in rows.values() if 'saved_self_history_replay' in r]
        assert len(replay) == 3 and max(replay) == 0
        for mode in ['gt_incremental', 'self_incremental']:
            assert [r['position'] for r in data['cache_trace'] if r['mode'] == mode] == list(range(12))
        assert max(r['video_frames'] for r in data['cache_trace']) <= 2
        positions = []
        for pos in [0, 3, 11]:
            gt, own, truncated = [rows[(m, pos)]['rollout_vs_HR']['mse'] for m in
                                  ['gt_incremental', 'self_incremental', 'gt_truncated_offline']]
            baseline = rows[('gt_incremental', pos)]['no_refiner_vs_HR']['mse']
            positions.append({'position_zero_based': pos, 'no_refiner_MSE': baseline,
                              'GT_history_MSE': gt, 'self_history_MSE': own, 'truncated_GT_MSE': truncated,
                              'self_over_GT': own/gt, 'self_over_no_refiner': own/baseline,
                              'truncated_over_GT': truncated/gt, 'GT_over_no_refiner': gt/baseline})
        actual = positions[0]['self_history_MSE']
        conditions = [{'condition': r['condition'], 'MSE': r['rollout_vs_HR']['mse'],
                       'ratio_to_actual_draft_condition': r['rollout_vs_HR']['mse']/actual}
                      for r in data['first_latent_condition_ablation']['rows']]
        fm = []
        for pos in [3, 11]:
            for row in data['rows']:
                if row['mode'] == 'self_incremental' and row['position'] == pos and 'sigma' in row:
                    gt = next(r for r in data['rows'] if r['mode'] == 'gt_incremental'
                              and r['position'] == pos and r.get('sigma') == row['sigma'])
                    fm.append({'position': pos, 'sigma': row['sigma'], 'self_over_GT': row['fm_mse']/gt['fm_mse']})
        samples.append({'sample_id': sid, 'split': data['split'], 'receipt_sha256': hashlib.sha256(raw).hexdigest(),
                        'positions': positions, 'first_latent_condition': conditions, 'point_FM_ratios': fm,
                        'four_latent_parity_max_abs': 0, 'saved_self_replay_max_abs': 0,
                        'max_cache_bytes': max(r['bytes'] for r in data['cache_trace'])})
    assert len(checkpoints) == 1
    return {'schema': 'matched-history-analysis-v1', 'status': 'partial_cohort_interpretation',
            'checkpoint_sha256': next(iter(checkpoints)), 'completed_samples': len(samples),
            'planned_samples': 5, 'new_optimizer_updates': 0, 'metric': 'latent MSE against current nativeHR latent',
            'samples': samples,
            'controlled_intervention': 'Same checkpoint/current noisy input/draft/noise/static conditioning; replace only history content or current condition. GT history and currentHR condition are diagnostic oracles, unavailable in deployment.',
            'confirmed_scope': ['Full current spatial68x120, three sampled positions,12-latent history.',
                                'Projection-aligned full-prefix oracle and saved self-history replay exact0 for the completed samples.',
                                'Condition ablation keeps original draft/noise start fixed; it changes only the current conditioning input.'],
            'interpretation': ['Self-generated history worsens the sampled later-position outputs substantially; exposure to deployment histories warrants priority investigation.',
                               'First-latent degradation precedes any video history; history error alone cannot explain the failure.',
                               'Replacing current draft condition with HR does not rescue first-latent reconstruction on these completed dev samples; this supports investigating fidelity conditioning and denoising design.',
                               'Truncated-versus-incremental GT context changes outputs, but the direction differs across these two samples and its effect is smaller than the self-versus-GT history intervention.'],
            'limits': ['Only completed samples included; train/dev comparison pending until both training clips finish.',
                       'Counterfactual GT histories are not a deployable solution and may supply strongly informative past content.',
                       'These latent probes do not establish full226 pixel improvement, visual acceptance, convergence, or an exclusive root cause.',
                       'No training restart or production-runtime change is implied.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('receipts', nargs='+', type=Path)
    args = parser.parse_args()
    result = summarize(args.receipts)
    with args.output.open('x') as f:
        f.write(json.dumps(result, indent=2)+'\n')
