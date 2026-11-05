"""Plot measured causal Conv3d hybrid training and fixed-region dev metrics on CPU.

Requires a completed run and its original contiguous steps.jsonl. Training
curves are trailing 20-update averages of changing clips/regions, not estimates
of the full-training-set mean. Dev curves use the logged fixed full-spatial
native temporal regions; they are not full-video test measurements.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


WINDOW = 20
TRAIN_KEYS = ('loss', 'latent_mse', 'teacher_mse', 'ewarp_l1', 'pixel_l1', 'lpips')
DEV_KEYS = ('pixel_l1', 'lpips', 'selection_score', 'latent_mse')
COLORS = {'loss': '#253858', 'latent_mse': '#176D64', 'teacher_mse': '#8563A6',
          'ewarp_l1': '#CA8433', 'pixel_l1': '#3076B7', 'lpips': '#C65E72',
          'selection_score': '#253858'}
LABELS = {'loss': 'Combined training loss', 'latent_mse': 'HR latent MSE',
          'teacher_mse': 'Teacher KD MSE', 'ewarp_l1': 'eWarp L1',
          'pixel_l1': 'RGB L1', 'lpips': 'LPIPS (Alex v0.1)',
          'selection_score': 'Selection score'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def invalid(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def finite_metrics(record, keys, context):
    for key in keys:
        require(key in record and isinstance(record[key], (int, float))
                and math.isfinite(float(record[key])), f'{context}: missing/nonfinite {key}')


def read_history(run):
    summary = read_json(run / 'summary.json')
    require(summary['status'] == 'completed' and not summary['config'].get('preflight'),
            'Plot requires a completed non-preflight run')
    count = summary['completed_updates']
    require(type(count) is int and count >= WINDOW and count == summary['config']['steps'],
            'Completed update count must match config and provide at least 20 updates')
    records = []
    for index, line in enumerate((run / 'steps.jsonl').read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        finite_metrics(record, TRAIN_KEYS, f'Training log line {index}')
        records.append(record)
    require([row['step'] for row in records] == list(range(1, count + 1)),
            'Training log has missing, duplicate, out-of-order or extra steps')
    validation = summary['validation']
    require(bool(validation), 'Missing dev history')
    steps = [row['step'] for row in validation]
    require(steps[0] == 0 and steps[-1] == count and steps == sorted(set(steps)),
            'Dev history must include unique ordered step 0 through the final update')
    require(all(type(step) is int and 0 <= step <= count for step in steps), 'Invalid dev step')
    weight = float(summary['config']['lpips_weight'])
    require(math.isfinite(weight) and weight >= 0, 'Invalid dev LPIPS weight')
    require(type(summary['dev_count']) is int and summary['dev_count'] > 0, 'Missing dev count')
    # Verify that the curves really describe unchanged full-spatial regions,
    # rather than silently plotting changing patches as a fixed dev signal.
    baseline_samples = validation[0]['samples']
    require(len(baseline_samples) == summary['dev_count'], 'Dev sample count differs')
    baseline_specs = {ident: row['spec'] for ident, row in baseline_samples.items()}
    require(all(spec['spatial_mode'] == 'full' for spec in baseline_specs.values()),
            'Dev history is not full-spatial region evaluation')
    lpips_counts = sorted({len(spec['lpips_frame_indices']) for spec in baseline_specs.values()})
    require(lpips_counts and min(lpips_counts) > 0, 'Missing dev LPIPS frame sampling')
    for row in validation:
        finite_metrics(row['mean'], DEV_KEYS, f'Dev step {row["step"]}')
        require({ident: value['spec'] for ident, value in row['samples'].items()} == baseline_specs,
                'Dev identities or fixed temporal/spatial regions changed across validation')
        expected_score = float(row['mean']['pixel_l1']) + weight * float(row['mean']['lpips'])
        require(math.isclose(expected_score, float(row['mean']['selection_score']), rel_tol=1e-6, abs_tol=1e-7),
                'Logged dev score differs from RGB L1 + configured LPIPS weight')
    # Trainer updates on strict improvement, so the first exact minimum wins.
    best = min(validation, key=lambda row: row['mean']['selection_score'])
    require(summary['best_step'] == best['step'], 'Best checkpoint does not match logged dev selection')
    score = summary.get('best_validation_score', summary.get('best_score'))
    require(score is not None and math.isclose(float(score), float(best['mean']['selection_score']),
                                               rel_tol=1e-6, abs_tol=1e-7),
            'Summary selected dev score differs from history')
    return summary, records, validation, best, lpips_counts


def trailing_mean(records, key):
    """Only full 20-update windows; do not invent smoothed values before step 20."""
    values = [float(row[key]) for row in records]
    return [row['step'] for row in records[WINDOW - 1:]], [
        math.fsum(values[index - WINDOW + 1:index + 1]) / WINDOW
        for index in range(WINDOW - 1, len(values))]


def train_panel(ax, records, keys, title):
    for key in keys:
        x, y = trailing_mean(records, key)
        ax.plot(x, y, color=COLORS[key], linewidth=1.8, label=LABELS[key])
    ax.set_title(title, fontsize=11, loc='left')
    ax.set_ylabel('Logged value (unweighted components)'
                  if keys != ('loss',) else 'Weighted combined objective')
    ax.legend(frameon=False, fontsize=8, loc='best')


def dev_panel(ax, validation, best, keys, title):
    steps = [row['step'] for row in validation]
    for key in keys:
        values = [row['mean'][key] for row in validation]
        ax.plot(steps, values, color=COLORS[key], marker='o', markersize=3.5,
                linewidth=1.5, label=LABELS[key])
        ax.axhline(values[0], color=COLORS[key], linewidth=1., linestyle=':',
                   alpha=.65, label=f'{LABELS[key]} at step 0')
        ax.scatter([best['step']], [best['mean'][key]], color=COLORS[key],
                   edgecolor='white', linewidth=.6, marker='*', s=130, zorder=5)
    ax.set_title(title, fontsize=11, loc='left')
    ax.set_ylabel('Mean across fixed dev clips')
    ax.legend(frameon=False, fontsize=7.5, loc='best')


def plot(run, output):
    summary, records, validation, best, lpips_counts = read_history(run)
    require(output.suffix.lower() == '.png', '--output must be a PNG path')
    require(not output.exists(), 'Refusing to overwrite an existing figure')
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    train_panel(axes[0, 0], records, ('loss',), 'Train: combined loss | trailing 20 updates')
    train_panel(axes[0, 1], records, ('latent_mse', 'teacher_mse', 'ewarp_l1'),
                'Train: latent components | trailing 20 updates')
    train_panel(axes[0, 2], records, ('pixel_l1', 'lpips'),
                'Train: pixel components | trailing 20 updates')
    dev_panel(axes[1, 0], validation, best, ('pixel_l1', 'lpips'),
              f'Dev: fixed full-spatial regions | {summary["dev_count"]} clips')
    weight = float(summary['config']['lpips_weight'])
    dev_panel(axes[1, 1], validation, best, ('selection_score',),
              f'Dev: selection score = RGB L1 + {weight:g} LPIPS')
    dev_panel(axes[1, 2], validation, best, ('latent_mse',),
              'Dev: HR latent MSE | first 36 real-anchor latents')
    count = summary['completed_updates']
    for ax in axes.flat:
        ax.axvline(best['step'], color='#A96A21', linestyle='--', linewidth=1., alpha=.8)
        ax.set_xlim(-count * .015, count * 1.02)
        ax.set_xlabel('Conv3d hybrid optimizer updates (warm start = step 0)', fontsize=9)
        ax.grid(alpha=.2, linewidth=.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)
    prior = summary['config'].get('prior_updates')
    initialization = (f'Warm start from causal Conv3d eWarp after {prior:,} prior updates'
                      if type(prior) is int else 'Warm start from the selected causal Conv3d eWarp checkpoint')
    fig.suptitle(f'H3 causal Conv3d hybrid upscaler | {count:,} measured updates | selected step {best["step"]:,}',
                 fontsize=16, y=.975)
    fig.text(.5, .938, f'{initialization}. Dashed vertical line / stars: checkpoint chosen by dev pixel score; dotted horizontals: step 0.',
             ha='center', fontsize=9, color='#444444')
    frames = '/'.join(str(n) for n in lpips_counts)
    notes = [
        'Training: each 20-update window mixes changing clips and sampled full/tile regions; these are not full-training-set means.',
        f'Dev pixels: fixed native temporal regions at full spatial size, {frames} LPIPS frames per region; these are not full 120-frame test metrics.',
        'All points come from the saved run. Lines connect observed dev evaluations only; test/legacy results do not select the checkpoint.'
    ]
    for index, note in enumerate(notes):
        fig.text(.055, .075 - .019 * index, note, ha='left', fontsize=9, color='#444444')
    fig.subplots_adjust(left=.065, right=.98, top=.885, bottom=.15, wspace=.29, hspace=.44)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, facecolor='white', metadata={
        'Title': 'Measured H3 causal Conv3d hybrid training/dev curves',
        'Description': f'Trailing 20-update train averages; fixed dev regions; selected step {best["step"]}; no test-based selection.'})
    plt.close(fig)
    print(json.dumps({'output': str(output), 'completed_updates': count, 'best_step': best['step'],
                      'training_average_window': WINDOW, 'dev_evaluations': len(validation),
                      'status': 'plotted'}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Completed run containing summary.json and steps.jsonl')
    parser.add_argument('--output', type=Path, required=True, help='New PNG path')
    args = parser.parse_args()
    plot(args.run, args.output)


if __name__ == '__main__':
    main()
