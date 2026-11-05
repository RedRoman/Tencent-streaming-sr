"""Evaluate a completed H3 CMN upsampler run; the original H3 VAE stays frozen.

First invocation independently reloads best.pt and evaluates all 16 retained
test + 3 legacy clips. --decode-only decodes a fixed four-video subset in shards;
--finalize verifies shard fingerprints and completely decodes every saved MP4
with FFmpeg. No optimizer, checkpoint selection, or decoder training occurs.

These are reused test identities, not a new blind test. The historical Conv3d
baselines have different architecture, initialization and training budgets; this
is a descriptive comparison, never an equal-compute experiment. Causality tests
cover the upsampler on fixed latents, not the noncausal VAE or complete pipeline.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

import torch
from PIL import Image

from data_utils import (decode_latent, empty_cache, ffmpeg_executable,
                        load_video_vae, select_device, synchronize, tensor_load,
                        write_mp4)
from decode_results import comparisons
from evaluate_causal_upscaler_pixels import metric
from h3_cmn import H3LatentUpsampler, state_nbytes
from train_h3_cmn import bilinear, causal_checks, objective
from train_upscaler_trial import atomic_json, atomic_torch_save, sha256
from train_upscaler_ultraflash import load_pairs, tensors


OLD_VARIANTS = ('pretrained', 'long_temporal', 'long_ewarp')
LATENT_VARIANTS = ('bilinear', 'cmn', *OLD_VARIANTS, 'reference')
DECODE_VARIANTS = {'bilinear': 'bilinear', 'cmn': 'cmn',
                   'old_causal_ewarp': 'long_ewarp', 'pretrained': 'pretrained',
                   'reference': 'reference'}
METRIC_KEYS = ('latent_mse', 'ewarp_l1', 'valid_warp_fraction')
SOURCE_FILES = ('evaluate_h3_cmn.py', 'train_h3_cmn.py', 'h3_cmn.py',
                'ultraflash_ewarp.py', 'train_upscaler_ultraflash.py',
                'data_utils.py', 'decode_results.py', 'evaluate_causal_upscaler_pixels.py')


def read_json(path):
    def invalid(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_record(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size > 0, f'Missing/empty file: {path}')
    return {'sha256': sha256(path), 'bytes': path.stat().st_size}


def module_source(name):
    """Resolve the implementation Python actually imports, including shared helpers."""
    require(Path(name).name == name and name.endswith('.py'), 'Unsafe source-module name')
    if name == Path(__file__).name:
        return Path(__file__).resolve()
    spec = importlib.util.find_spec(Path(name).stem)
    require(spec is not None and spec.origin is not None and Path(spec.origin).is_file(),
            f'Cannot resolve imported source: {name}')
    return Path(spec.origin)


def source_hashes():
    return {name: sha256(module_source(name)) for name in SOURCE_FILES}


def parity(a, b, tolerance):
    require(tuple(a.shape) == tuple(b.shape), 'Parity tensors have different shapes')
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), 'Nonfinite parity tensor')
    delta = (a - b).abs()
    return {'passed': bool(torch.allclose(a, b, atol=tolerance, rtol=tolerance)),
            'max_abs': float(delta.max()), 'mean_abs': float(delta.mean()),
            'atol': tolerance, 'rtol': tolerance}


def numeric_metrics(values):
    result = {key: float(value) for key, value in values.items()}
    require(all(math.isfinite(v) for v in result.values()), 'Nonfinite latent metrics')
    return result


def _manifest_rows(args):
    ready = read_json(args.pairs / 'READY.json')
    plan_hash = sha256(args.pairs / 'plan.json')
    require(ready['status'] == 'ready' and ready['plan_sha256'] == plan_hash, 'Data is not finalized')
    rows, manifests = {}, {}
    for split in ('train', 'dev', 'test', 'legacy'):
        path = args.pairs / f'{split}.jsonl'
        manifests[split] = sha256(path)
        require(manifests[split] == ready['manifests'][split]['sha256'], f'{split} manifest changed')
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ident = row['sample_id']
            require(isinstance(ident, str) and Path(ident).name == ident and ident not in ('.', '..'), 'Unsafe sample ID')
            require(row['split'] == split and ident not in rows, 'Split/sample identity collision')
            rows[ident] = row
    sources = [row['source_video_sha256'] for row in rows.values()]
    require(len(set(sources)) == len(sources), 'Source overlap across splits')
    require(sum(r['split'] == 'test' for r in rows.values()) == 16, 'Expected 16 retained test clips')
    require(sum(r['split'] == 'legacy' for r in rows.values()) == 3, 'Expected 3 legacy clips')
    return rows, {'plan_sha256': plan_hash, 'manifest_sha256': manifests,
                  'ready_sha256': sha256(args.pairs / 'READY.json')}


def inputs(args):
    rows, fingerprints = _manifest_rows(args)
    run = read_json(args.run / 'summary.json')
    old = read_json(args.old_evaluation / 'summary.json')
    require(run['status'] == 'completed' and not run['config'].get('preflight'), 'Training run is incomplete/preflight')
    require(run['completed_updates'] == run['config']['steps'], 'Completed update count differs from requested training')
    require(old['status'] == 'completed', 'Historical evaluation is incomplete')
    require(run['contract']['plan_sha256'] == old['plan_sha256'] == fingerprints['plan_sha256'], 'Plan SHA differs')
    require(run['contract']['train_manifest_sha256'] == fingerprints['manifest_sha256']['train'], 'Training manifest differs')
    require(run['contract']['dev_manifest_sha256'] == fingerprints['manifest_sha256']['dev'], 'Dev manifest differs')
    require(run['best_checkpoint_sha256'] == sha256(args.run / 'best.pt'), 'Best checkpoint changed')
    require(run['causality']['passed'], 'Training causality check failed')
    for name, expected in run['contract']['source_sha256'].items():
        require(Path(name).name == name, 'Unsafe training source name')
        require(sha256(module_source(name)) == expected, f'Training implementation changed: {name}')
    selected = old['selected_videos']
    plan = read_json(args.pairs / 'plan.json')
    expected = [sorted(r['sample_id'] for r in plan['entries']
                       if r['split'] == 'test' and r['category'] == category)[0]
                for category in ('animal', 'daily', 'sport')] + ['VIDGEN5S_0380588']
    require(selected == expected and len(set(selected)) == 4, 'Selected videos differ from original fixed rule')
    heldout = {ident for ident, row in rows.items() if row['split'] in ('test', 'legacy')}
    require(set(selected) <= heldout, 'Selected video is outside held-out identities')
    for label in OLD_VARIANTS:
        require(set(old['variants'][label]) == heldout, f'Historical {label} has a different result set')
        for ident, record in old['variants'][label].items():
            require(record['split'] == rows[ident]['split'], 'Historical split differs')
            require(all(math.isfinite(float(record[k])) for k in METRIC_KEYS), 'Nonfinite historical metric')
    fingerprints.update(train_summary_sha256=sha256(args.run / 'summary.json'),
                        checkpoint_sha256=sha256(args.run / 'best.pt'),
                        old_summary_sha256=sha256(args.old_evaluation / 'summary.json'),
                        evaluation_source_sha256=source_hashes())
    return rows, run, old, fingerprints


@torch.no_grad()
def benchmark(model, x, device, repeats):
    """Synchronized, warmed, in-memory upsampler only; no data transfer or VAE."""
    def step_sequence(record_times=False):
        state, outputs, durations = None, [], []
        for j in range(x.shape[2]):
            if record_times:
                synchronize(device)
                tick = time.perf_counter()
            out, state = model.forward_step(x[:, :, j], state, latent_index=j)
            if record_times:
                synchronize(device)
                durations.append(time.perf_counter() - tick)
            outputs.append(out)
        return torch.cat(outputs, dim=2), state, durations
    model(x); model(x); step_sequence()
    synchronize(device)
    full_times, stream_times, first_times, subsequent = [], [], [], []
    for _ in range(repeats):
        synchronize(device)
        tick = time.perf_counter()
        full = model(x)
        synchronize(device)
        full_times.append(time.perf_counter() - tick)
        tick = time.perf_counter()
        stepped, state, durations = step_sequence(True)
        synchronize(device)
        stream_times.append(time.perf_counter() - tick)
        first_times.append(durations[0]); subsequent.extend(durations[1:])
    check = parity(full, stepped, 3e-5)
    require(check['passed'], 'Independent forward_step does not match full forward')
    return {'scope': 'Upsampler only, FP32, B=1, input already on device; excludes VAE, I/O and transfer',
            'warmup': 'Two full forwards plus one complete 37-step stream', 'repeats': repeats,
            'full_37_latent_seconds': full_times, 'full_37_latent_median_seconds': statistics.median(full_times),
            'stream_37_latent_seconds': stream_times, 'stream_37_latent_median_seconds': statistics.median(stream_times),
            'stream_first_latent_median_seconds': statistics.median(first_times),
            'stream_subsequent_latent_median_seconds': statistics.median(subsequent),
            'stream_note': 'Per-step synchronization is included; full and streaming calls use different batching',
            'cache_tensor_bytes': state_nbytes(state), 'full_vs_step': check,
            'historical_seconds_comparable': False}


def mean_by_split(variants):
    return {label: {split: {key: statistics.mean(float(row[key]) for row in records.values() if row['split'] == split)
                            for key in METRIC_KEYS} for split in ('test', 'legacy')}
            for label, records in variants.items()}


def evaluate_latents(args):
    require(not args.output.exists(), 'Use a fresh output directory for latent evaluation')
    rows, run, old, fingerprints = inputs(args)
    saved = tensor_load(args.run / 'best.pt')
    require(saved['contract'] == run['contract'], 'Checkpoint/training contract differs')
    require(saved['step'] == run['best_step'], 'Checkpoint does not match selected dev step')
    require(saved['model_config'] == {k: run['config'][k] for k in ('width', 'blocks')}, 'Model config differs')
    args.output.mkdir(parents=True)
    selected = old['selected_videos']
    summary = {'status': 'evaluating', 'plan_sha256': fingerprints['plan_sha256'],
               'input_fingerprints': fingerprints, 'selected_videos': selected,
               'video_selection': old['video_selection'], 'frames': 120, 'fps': 24, 'seconds_per_video': 5,
               'comparison_order': list(DECODE_VARIANTS), 'decode_latent_mapping': DECODE_VARIANTS,
               'variants': {'bilinear': {}, 'cmn': {}, **{v: old['variants'][v] for v in OLD_VARIANTS}},
               'reload_parity': {'cmn': {}}, 'causality': {'cmn': {}},
               'latent_files': {}, 'saved_prediction_sha256': {}, 'historical_latent_sha256': {},
               'train_run': {k: run[k] for k in ('completed_updates', 'best_step', 'best_validation_mse',
                                                'best_checkpoint_sha256', 'parameter_count', 'prior_training_updates')},
               'historical_train_runs': old['train_runs'],
               'metric_comparison': {'same_plan': True, 'equal_compute': False,
                    'cmn_initialization': 'Random CMN, zero residual over bilinear',
                    'old_initialization': 'Published Conv3d weights; causal variants additionally prior50 + 2000 updates',
                    'historical_metrics': 'Copied from fingerprinted old evaluation; same pairs and metric definitions',
                    'old_seconds': 'Historical unwarmed per-sample timings, not compared with new warmed benchmark'},
               'test_reuse': 'All16 test and3 legacy clips were already inspected in earlier Conv3d trials; not blind',
               'selection': 'Training best.pt selected only by dev latent MSE; this evaluation never selects weights',
               'latent_metric_scope': 'All16 test +3 legacy; first36 real-anchor latents and35 valid warp pairs',
               'pixel_metric_scope': 'Four fixed videos only; primary full120 vs decoded HR reference, source RGB secondary',
               'causal_scope': 'Upsampler on fixed latents only; frozen original H3 decoder remains noncausal',
               'end_to_end_causal': False, 'decoder_trained': False,
               'padding_context': {'latent_supervised': list(range(36)), 'latent_decoded': 37,
                    'excluded_rgb_tail': [120, 124], 'diagnostic_prefix': [0, 102], 'diagnostic_tail': [102, 120],
                    'note': 'Unsupervised latent36 still enters the final noncausal VAE window z30..36. '
                            'Cropping RGB120..123 does not remove its influence on real tail frames; '
                            'frame102 has zero current-window crossfade weight, frames103..119 may depend on it. '
                            'Region-specific temporal-difference metrics omit the cross-region frame101->102 pair.'}}
    atomic_json(args.output / 'summary.json', summary)
    device = select_device(args.device)
    model = H3LatentUpsampler(**saved['model_config']).to(device)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.eval().requires_grad_(False)
    del saved
    pairs = load_pairs(args.pairs / 'test.jsonl') + load_pairs(args.pairs / 'legacy.jsonl')
    require({p['sample_id'] for p in pairs} == set(old['variants']['pretrained']), 'Loaded pair set differs')
    with torch.no_grad():
        for pair in pairs:
            ident, split = pair['sample_id'], pair['split']
            draft, reference, flow, valid = tensors(pair, device)
            prediction_path = args.run / 'predictions' / split / f'{ident}.pt'
            expected = tensor_load(prediction_path)
            require(expected['sample_id'] == ident and expected['split'] == split, 'Saved prediction identity differs')
            require(expected['pair_sha256'] == pair['pair_sha256'], 'Saved prediction pair differs')
            pred = model(draft)
            check = parity(pred[0], expected['latent'], args.tolerance)
            summary['reload_parity']['cmn'][ident] = check
            summary['saved_prediction_sha256'][ident] = sha256(prediction_path)
            require(check['passed'], f'Independent best.pt reload differs: {ident}')
            computed = {}
            for label, value in (('cmn', pred), ('bilinear', bilinear(draft))):
                computed[label] = value
                res = numeric_metrics(objective(value, reference, flow, valid, .1))
                summary['variants'][label][ident] = {'split': split, 'metric_weight': .1, **res}
                # Training reports its configured weight, but these component metrics must match.
                prior = run[split if label == 'cmn' else f'bilinear_{split}']['samples'][ident]
                require(all(math.isclose(res[k], float(prior[k]), abs_tol=args.tolerance, rel_tol=args.tolerance)
                            for k in METRIC_KEYS), f'Reloaded metric differs: {label}/{ident}')
            if ident in selected:
                dst = args.output / 'latents' / ident
                summary['latent_files'][ident] = {}
                for label in ('cmn', 'bilinear', 'reference'):
                    value = reference if label == 'reference' else computed[label]
                    path = dst / f'{label}.pt'
                    atomic_torch_save(path, {'latent': value[0].cpu(), 'sample_id': ident,
                                            'split': split, 'pair_sha256': pair['pair_sha256']})
                    summary['latent_files'][ident][label] = file_record(path)
                summary['historical_latent_sha256'][ident] = {}
                old_ref_path = args.old_evaluation / 'latents' / ident / 'reference.pt'
                old_ref = tensor_load(old_ref_path)['latent']
                require(torch.equal(old_ref.float(), pair['reference'].float()), 'Historical HR reference differs')
                summary['historical_latent_sha256'][ident]['reference'] = sha256(old_ref_path)
                for label in OLD_VARIANTS:
                    path = args.old_evaluation / 'latents' / ident / f'{label}.pt'
                    old_latent = tensor_load(path)['latent'].float()
                    require(tuple(old_latent.shape) == (24, 37, 24, 42) and bool(torch.isfinite(old_latent).all()), 'Bad old latent')
                    # Cross-check cached baseline values against its published per-clip metrics.
                    old_metrics = numeric_metrics(objective(old_latent[None].to(device), reference, flow, valid, .1))
                    require(all(math.isclose(old_metrics[k], float(old['variants'][label][ident][k]),
                                             abs_tol=1e-5, rel_tol=1e-5) for k in METRIC_KEYS), 'Historical latent/metric mismatch')
                    summary['historical_latent_sha256'][ident][label] = sha256(path)
                    target = dst / f'{label}.pt'
                    atomic_torch_save(target, {'latent': old_latent, 'sample_id': ident,
                                              'split': split, 'pair_sha256': pair['pair_sha256']})
                    summary['latent_files'][ident][label] = file_record(target)
                summary['causality']['cmn'][ident] = causal_checks(model, draft, args.tolerance)
                require(summary['causality']['cmn'][ident]['passed'], f'Causality failed: {ident}')
                sample_max = max(row['max_abs'] for row in summary['causality']['cmn'][ident]['checks'].values())
                summary['causality']['cmn'][ident]['absolute_max_abs'] = sample_max
                summary['causality']['cmn'][ident]['absolute_max_at_most_1e_minus_4'] = sample_max <= 1e-4
                require(sample_max <= 1e-4, f'Causality absolute error exceeds1e-4: {ident}: {sample_max}')
            atomic_json(args.output / 'summary.json', summary)
            print(json.dumps({'latent_evaluated': ident, 'reload_max_abs': check['max_abs']}), flush=True)
            del draft, reference, flow, valid, pred, computed, expected
        benchmark_pair = next(p for p in pairs if p['sample_id'] == selected[0])
        summary['warmed_upsampler_timing'] = benchmark(model, benchmark_pair['draft'][None].to(device), device, args.timing_repeats)
    summary['mean_by_split'] = mean_by_split(summary['variants'])
    summary['causality_test_count'] = sum(len(v['checks']) for v in summary['causality']['cmn'].values())
    expected_ids = {p['sample_id'] for p in pairs}
    require(len(expected_ids) == 19 and set(summary['reload_parity']['cmn']) == expected_ids,
            'Reload parity does not cover all19 expected samples')
    require(set(summary['latent_files']) == set(summary['causality']['cmn']) == set(selected),
            'Saved latents/causality do not cover the four fixed videos')
    require(summary['causality_test_count'] == 36 and
            all(len(v['checks']) == 9 for v in summary['causality']['cmn'].values()),
            'Expected nine frozen-trainer causal checks on each of four fixed videos')
    summary['causality_passed'] = all(v['passed'] for v in summary['causality']['cmn'].values())
    summary['reload_parity_passed'] = all(v['passed'] for v in summary['reload_parity']['cmn'].values())
    summary['causality_max_abs'] = max(v['absolute_max_abs'] for v in summary['causality']['cmn'].values())
    summary['reload_parity_max_abs'] = max(v['max_abs'] for v in summary['reload_parity']['cmn'].values())
    summary['parity_tolerance'] = {'atol': args.tolerance, 'rtol': args.tolerance,
                                 'causality_additional_absolute_limit': 1e-4}
    summary['status'] = 'latents_completed'
    # Immutable snapshot: decode shards/finalize all bind to this same hash.
    atomic_json(args.output / 'latent_summary.json', summary)
    atomic_json(args.output / 'summary.json', summary)
    del model, pairs
    empty_cache(device)
    print(json.dumps({'status': summary['status'], 'reload_checks': 19,
                      'causal_checks': summary['causality_test_count']}), flush=True)


def verified_latent_summary(args):
    snapshot = args.output / 'latent_summary.json'
    summary = read_json(snapshot)
    require(summary['status'] == 'latents_completed', 'No immutable completed latent evaluation')
    rows, _, _, fingerprints = inputs(args)
    require(fingerprints == summary['input_fingerprints'], 'Input/source fingerprints changed after latent evaluation')
    require(summary['causality_passed'] and summary['reload_parity_passed'], 'Latent checks did not pass')
    for ident, files in summary['latent_files'].items():
        require(set(files) == set(LATENT_VARIANTS), 'Latent variant set differs')
        for label, expected in files.items():
            require(file_record(args.output / 'latents' / ident / f'{label}.pt') == expected, 'Saved latent changed')
    return summary, rows, sha256(snapshot)


def vae_fingerprint(base_model):
    directory = Path(base_model) / 'vae'
    require((directory / 'config.json').is_file(), 'Expected base-model/vae/config.json')
    weights = sorted(set(directory.glob('*.safetensors')) | set(directory.glob('*.bin')))
    require(bool(weights), 'No VAE weight files')
    files = [directory / 'config.json', *weights]
    return {path.name: file_record(path) for path in files}


def audit_video(path, width, height):
    """Full CPU decode, checking all frame PTS; no ffprobe dependency or rewrite."""
    command = [ffmpeg_executable(), '-hide_banner', '-nostdin', '-v', 'info', '-xerror',
               '-i', str(path), '-map', '0:v:0', '-an', '-sn', '-dn', '-vf', 'showinfo',
               '-vsync', '0', '-f', 'null', '-']
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    require(result.returncode == 0, f'Full video decode failed: {path.name}: {result.stderr[-1000:]}')
    durations = re.findall(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)', result.stderr)
    require(len(durations) == 1, 'Could not uniquely read input container duration')
    hours, minutes, seconds = durations[0]
    duration = 3600 * int(hours) + 60 * int(minutes) + float(seconds)
    require(abs(duration - 5.) < .005, 'Container duration differs from5seconds')
    pattern = r'\[Parsed_showinfo[^\]]*\].*?\bn:\s*(\d+)\s+pts:\s*-?\d+\s+pts_time:([-+0-9.eE]+).*?\bs:(\d+)x(\d+)'
    decoded = [(int(n), float(t), int(w), int(h)) for n, t, w, h in re.findall(pattern, result.stderr)]
    require(len(decoded) == 120 and [row[0] for row in decoded] == list(range(120)), 'Video does not decode exactly120 frames')
    require(all((w, h) == (width, height) for _, _, w, h in decoded), 'Decoded dimensions differ')
    require(all(abs(t - n / 24.) < 1e-5 for n, t, _, _ in decoded), 'Video PTS differs from zero-based24fps')
    return {'passed': True, 'decoder': 'FFmpeg full decode with showinfo and -xerror',
            'decoded_frames': 120, 'width': width, 'height': height,
            'first_pts_seconds': decoded[0][1], 'last_pts_seconds': decoded[-1][1],
            'fps': 24, 'duration_seconds': duration, **file_record(path)}


def audit_media(directory):
    evidence = {}
    for label in (*DECODE_VARIANTS, 'comparison'):
        width, height = (3360, 412) if label == 'comparison' else (672, 384)
        evidence[f'{label}.mp4'] = audit_video(directory / f'{label}.mp4', width, height)
    png = directory / 'comparison.png'
    with Image.open(png) as image:
        require(image.format == 'PNG' and image.size == (3360, 1236), 'Contact-sheet format/size differs')
        image.verify()
    with Image.open(png) as image:
        image.load()
    evidence['comparison.png'] = {'passed': True, 'width': 3360, 'height': 1236, **file_record(png)}
    return evidence


def decode_shard(args):
    summary, rows, snapshot_hash = verified_latent_summary(args)
    labels = [ident for i, ident in enumerate(summary['selected_videos']) if i % args.num_shards == args.shard_index]
    require(bool(labels), 'Empty decode shard')
    done_path = args.output / f'decode_complete_shard{args.shard_index}.json'
    require(not done_path.exists(), 'Decode shard already completed; use finalize or a fresh evaluation output')
    directories = [args.output / 'decoded' / ident for ident in labels]
    require(not any(path.exists() for path in directories), 'Decode output exists; preserve evidence and use a fresh output')
    base_identity = vae_fingerprint(args.base_model)
    device = select_device(args.device)
    vae = load_video_vae(args.base_model, device)
    vae.eval().requires_grad_(False)
    require(not any(p.requires_grad for p in vae.parameters()), 'VAE must be frozen')
    vae_source = Path(inspect.getfile(type(vae)))
    decoder_identity = {'weights': base_identity, 'implementation_sha256': sha256(vae_source),
                        'recipe': 'data_utils.load_video_vae + decode_latent, original temporal windows, FP32, spatial tile256/overlap64'}
    output, media = {}, {}
    for ident, dst in zip(labels, directories):
        dst.mkdir(parents=True)
        row = rows[ident]
        pair_path = Path(row['path'])
        pair_path = pair_path if pair_path.is_absolute() else args.pairs / pair_path
        require(sha256(pair_path) == row['sha256'], 'Source pair changed')
        pair = tensor_load(pair_path)
        source = pair['source_pixels']
        require(tuple(source.shape) == (120, 384, 672, 3) and source.dtype == torch.uint8, 'Source RGB geometry/dtype differs')
        pixels, timings, codecs = {}, {}, {}
        for label, latent_label in DECODE_VARIANTS.items():
            latent = tensor_load(args.output / 'latents' / ident / f'{latent_label}.pt')['latent']
            if label == 'reference':
                require(torch.equal(latent.float(), pair['reference'].float()), 'Decode reference differs from pair')
            synchronize(device)
            tick = time.perf_counter()
            rgb = decode_latent(vae, latent, device, keep_frames=120)
            synchronize(device)
            timings[label] = time.perf_counter() - tick
            require(tuple(rgb.shape) == (120, 384, 672, 3), 'Decoded RGB shape differs')
            codecs[label] = write_mp4(rgb, dst / f'{label}.mp4', fps=24.)
            pixels[label] = rgb
            print(json.dumps({'decoded': ident, 'variant': label, 'vae_seconds': timings[label]}), flush=True)
        entry = {'split': row['split'], 'decode_seconds': timings, 'video_codecs': codecs,
                 'metric_domain': 'uint8 decoded RGB before MP4 compression, scaled to [0,1]',
                 'variants': {}, 'reference_vs_source_rgb_secondary': metric(pixels['reference'], source)}
        for label, rgb in pixels.items():
            if label == 'reference':
                continue
            entry['variants'][label] = {
                'vs_decoded_reference': metric(rgb, pixels['reference']),
                'vs_source_rgb_secondary': metric(rgb, source),
                'prefix102': {'vs_decoded_reference': metric(rgb[:102], pixels['reference'][:102]),
                              'vs_source_rgb_secondary': metric(rgb[:102], source[:102])},
                'tail18': {'vs_decoded_reference': metric(rgb[102:], pixels['reference'][102:]),
                           'vs_source_rgb_secondary': metric(rgb[102:], source[102:])}}
        comparisons(pixels, dst, fps=24., crf=16, comparison_video=True)
        media[ident] = audit_media(dst)
        output[ident] = entry
        atomic_json(args.output / f'pixel_metrics_shard{args.shard_index}.json', output)
        del pair, pixels, source
    # Recheck before publishing a shard's completion evidence.
    require(sha256(args.output / 'latent_summary.json') == snapshot_hash, 'Latent snapshot changed while decoding')
    metrics = args.output / f'pixel_metrics_shard{args.shard_index}.json'
    atomic_json(done_path, {'status': 'completed', 'shard_index': args.shard_index,
                'num_shards': args.num_shards, 'samples': labels, 'latent_summary_sha256': snapshot_hash,
                'metrics_sha256': sha256(metrics), 'decoder_fingerprint': decoder_identity,
                'media': media, 'comparison_order': list(DECODE_VARIANTS)})
    del vae
    empty_cache(device)
    print(json.dumps({'decode_shard_completed': args.shard_index, 'samples': labels}), flush=True)


def finalize(args):
    summary, _, snapshot_hash = verified_latent_summary(args)
    combined, media, seen = {}, {}, set()
    decoder_identity = None
    for index in range(args.num_shards):
        done = read_json(args.output / f'decode_complete_shard{index}.json')
        metrics_path = args.output / f'pixel_metrics_shard{index}.json'
        part = read_json(metrics_path)
        expected = {ident for i, ident in enumerate(summary['selected_videos']) if i % args.num_shards == index}
        require(done['status'] == 'completed' and done['shard_index'] == index and done['num_shards'] == args.num_shards, 'Shard identity differs')
        require(done['latent_summary_sha256'] == snapshot_hash and done['metrics_sha256'] == sha256(metrics_path), 'Shard fingerprint differs')
        require(set(part) == set(done['samples']) == set(done['media']) == expected and not (seen & expected), 'Missing/overlapping shard samples')
        require(done['comparison_order'] == list(DECODE_VARIANTS), 'Comparison order differs')
        if decoder_identity is None:
            decoder_identity = done['decoder_fingerprint']
        require(done['decoder_fingerprint'] == decoder_identity, 'Different VAE weights/implementation across shards')
        for ident in expected:
            # Full decode again at finalization; file existence alone is not completion.
            actual = audit_media(args.output / 'decoded' / ident)
            require(actual == done['media'][ident], 'Saved media changed or failed full decode')
            media[ident] = actual
        combined.update(part); seen.update(expected)
    require(seen == set(summary['selected_videos']), 'Not all selected videos decoded')
    summary.update(status='completed', pixel_metrics=combined, decode_shards=args.num_shards,
                   media_audit=media, decoder_fingerprint=decoder_identity,
                   latent_summary_sha256=snapshot_hash, decode_complete=True)
    atomic_json(args.output / 'summary.json', summary)
    print(json.dumps({'completed': True, 'selected_videos': len(seen),
                      'full_decoded_mp4_files': sum(len(DECODE_VARIANTS) + 1 for _ in seen),
                      'causality_checks': summary['causality_test_count']}), flush=True)


def main():
    root = Path(os.environ.get('EXP_ROOT', Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--pairs', type=Path, default=root / 'upscaler_ultraflash/pairs')
    parser.add_argument('--old-evaluation', type=Path, default=root / 'upscaler_ultraflash/evaluation')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--base-model', default=os.environ.get('H3_BASE_MODEL', '/path/to/h3-base'))
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--tolerance', type=float, default=3e-5)
    parser.add_argument('--timing-repeats', type=int, default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--decode-only', action='store_true')
    mode.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    require(1 <= args.num_shards <= 4 and 0 <= args.shard_index < args.num_shards, 'Invalid shard identity')
    require(args.tolerance == 3e-5 and args.timing_repeats > 0, 'This evaluation fixes atol/rtol at3e-5; repeats must be positive')
    torch.set_num_threads(4)
    if args.finalize:
        finalize(args)
    elif args.decode_only:
        decode_shard(args)
    else:
        evaluate_latents(args)


if __name__ == '__main__':
    main()
