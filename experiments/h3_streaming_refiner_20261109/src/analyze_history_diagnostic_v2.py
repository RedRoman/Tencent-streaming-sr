"""Complete-cohort interpretation, retaining the earlier two-dev analysis."""
import argparse
import json
from pathlib import Path
from analyze_history_diagnostic_v1 import summarize


def complete(paths):
    report = summarize(paths)
    samples = report['samples']
    assert len(samples) == 5
    assert sum(s['split'] == 'train' for s in samples) == 2
    assert sum(s['split'] == 'dev' for s in samples) == 3
    grouped = {}
    for split in ['train', 'dev']:
        rows = [s for s in samples if s['split'] == split]
        history = [p['self_over_GT'] for s in rows for p in s['positions'][1:]]
        first = [s['positions'][0]['self_over_no_refiner'] for s in rows]
        oracle = [1-c['ratio_to_actual_draft_condition'] for s in rows
                  for c in s['first_latent_condition'] if 'hr' in c['condition'].lower()]
        assert len(oracle) == len(rows)
        grouped[split] = {'sample_count': len(rows),
                          'later_self_over_GT_range': [min(history), max(history)],
                          'first_self_over_no_refiner_range': [min(first), max(first)],
                          'first_HR_condition_fractional_reduction_range': [min(oracle), max(oracle)]}
    report.update(schema='matched-history-analysis-v2', status='complete_five_sample_diagnostic',
                  split_summary=grouped)
    report['interpretation'] = [
        'Self-history degradation and first-latent fidelity failure occur in both training and development samples; held-out generalization alone cannot explain them.',
        'GT-to-self history intervention changes later outputs under fixed current input. GT histories are informative oracles, not a deployment fix or proof of a unique cause.',
        'First-latent failure precedes video history. CurrentHR conditioning alone does not rescue it while draft/noise initialization stays fixed.',
        'Projection alignment removes the measured numerical parity gap, but does not establish that it fixes image quality.',
        'A separately authorized candidate anchors its output to the draft and trains on draft-derived histories. Its effect requires new training and full-video validation.']
    report['limits'] = [
        'Only two training and three development engineering proxy clips; test remains locked.',
        'Three sampled positions and12-latent histories are not full67-latent/226-frame acceptance.',
        'Pointwise FM loss and three-step rollout use different input distributions and cannot be interpreted as interchangeable quality scores.',
        'This read-only diagnostic performs zero optimizer updates and does not prove convergence, visual quality, or an exclusive root cause.']
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('receipts', type=Path, nargs='+')
    a = p.parse_args()
    with a.output.open('x') as f:
        f.write(json.dumps(complete(a.receipts), indent=2)+'\n')
