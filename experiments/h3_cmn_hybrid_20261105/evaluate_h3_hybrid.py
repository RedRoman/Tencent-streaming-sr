"""Evaluate a dev-selected CMN hybrid checkpoint against fixed historical results.

The first invocation independently reloads best.pt on all 16 reused test and
3 legacy clips. --decode-only evaluates four predetermined videos in shards;
--finalize checks every shard and fully decodes its saved media. No training or
checkpoint selection occurs here. Historical upsamplers are not rerun: their
fingerprinted latents are decoded again to measure LPIPS on uncompressed RGB.

The hybrid is the CMN architecture after complementary supervision, not an
inference ensemble. The historical 345M causal Conv3d model is named
old_causal_ewarp, never Method2. Causality is conditional on fixed input latents;
the original frozen H3 VAE is noncausal and is not trained in this experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
from PIL import Image

from data_utils import (decode_latent, empty_cache, load_video_vae, select_device,
                        synchronize, tensor_load, write_mp4)
from decode_results import comparisons
from evaluate_causal_upscaler_pixels import metric
from evaluate_h3_cmn import (_manifest_rows, audit_video, benchmark, file_record,
                             mean_by_split, module_source, numeric_metrics,
                             parity, read_json, require, vae_fingerprint)
from h3_cmn import H3LatentUpsampler
from train_h3_cmn import causal_checks, objective
from train_upscaler_trial import atomic_json, atomic_torch_save, sha256
from train_upscaler_ultraflash import load_pairs, tensors


HISTORICAL_VARIANTS = ('bilinear', 'pretrained', 'long_temporal', 'long_ewarp', 'cmn')
DECODE_VARIANTS = {'pretrained': 'pretrained', 'old_causal_ewarp': 'long_ewarp',
                   'cmn': 'cmn', 'hybrid': 'hybrid', 'reference': 'reference'}
LATENT_VARIANTS = tuple(DECODE_VARIANTS.values())
METRIC_KEYS = ('latent_mse', 'ewarp_l1', 'valid_warp_fraction')
PIXEL_KEYS = ('mse', 'psnr_db', 'temporal_difference_mse', 'lpips_alex_v0_1')
SOURCE_FILES = ('evaluate_h3_hybrid.py', 'evaluate_h3_cmn.py', 'train_h3_cmn.py',
                'h3_cmn.py', 'ultraflash_ewarp.py', 'train_upscaler_ultraflash.py',
                'train_upscaler_trial.py', 'data_utils.py', 'decode_results.py',
                'evaluate_causal_upscaler_pixels.py')


def source_hashes():
    return {name: sha256(Path(__file__) if name == Path(__file__).name else module_source(name))
            for name in SOURCE_FILES}


def selected_score(run):
    # The checkpoint uses best_score; the trainer's summary uses the more
    # explicit best_validation_score. Accept either summary spelling.
    values = [float(run[key]) for key in ('best_validation_score', 'best_score') if key in run]
    require(bool(values) and all(math.isfinite(v) for v in values), 'Missing/nonfinite selected dev score')
    require(all(v == values[0] for v in values), 'Conflicting selected dev scores')
    return values[0]


def inputs(args):
    """Bind inputs without imposing the old CMN training objective/schema."""
    rows, fingerprints = _manifest_rows(args)
    require(sum(r['split'] == 'train' for r in rows.values()) == 128, 'Expected128 training clips')
    require(sum(r['split'] == 'dev' for r in rows.values()) == 16, 'Expected16 development clips')
    run = read_json(args.run / 'summary.json')
    old = read_json(args.cmn_evaluation / 'summary.json')
    require(run['status'] == 'completed' and not run['config'].get('preflight'), 'Hybrid training is incomplete/preflight')
    require(run['completed_updates'] == run['config']['steps'], 'Hybrid update count differs from requested training')
    require(run['best_checkpoint_sha256'] == sha256(args.run / 'best.pt'), 'Hybrid best checkpoint changed')
    selected_score(run)
    require(0 <= run['best_step'] <= run['completed_updates'], 'Invalid selected checkpoint step')
    require(old['status'] == 'completed' and old['decode_complete'], 'Historical CMN evaluation is incomplete')
    require(old['causality_passed'] and old['reload_parity_passed'], 'Historical CMN checks failed')
    require(run['contract']['initial_sha256'] == old['train_run']['best_checkpoint_sha256'], 'Hybrid warmstart is not the evaluated CMN checkpoint')
    require(run['contract']['plan_sha256'] == old['plan_sha256'] == fingerprints['plan_sha256'], 'Plan SHA differs')
    # These common data keys are optional for schema compatibility; plan/READY,
    # checkpoint contract and actual held-out predictions are always verified.
    for split in ('train', 'dev'):
        key = f'{split}_manifest_sha256'
        if key in run['contract']:
            require(run['contract'][key] == fingerprints['manifest_sha256'][split], f'Hybrid {split} manifest differs')
    training_sources = run['contract'].get('source_sha256', {})
    require(bool(training_sources), 'Hybrid training contract must record source_sha256')
    for name, expected in training_sources.items():
        require(sha256(module_source(name)) == expected, f'Training implementation changed: {name}')
    old_snapshot_path = args.cmn_evaluation / 'latent_summary.json'
    require(sha256(old_snapshot_path) == old['latent_summary_sha256'], 'Historical latent snapshot changed')
    old_snapshot = read_json(old_snapshot_path)
    require(old_snapshot['status'] == 'latents_completed', 'Historical latent snapshot is incomplete')
    for key in ('plan_sha256', 'variants', 'selected_videos', 'latent_files', 'input_fingerprints'):
        require(old_snapshot[key] == old[key], f'Historical final/snapshot {key} differs')
    require(old['input_fingerprints']['manifest_sha256'] == fingerprints['manifest_sha256'], 'Historical manifests differ')
    selected = old['selected_videos']
    plan = read_json(args.pairs / 'plan.json')
    expected = [sorted(r['sample_id'] for r in plan['entries']
                       if r['split'] == 'test' and r['category'] == category)[0]
                for category in ('animal', 'daily', 'sport')] + ['VIDGEN5S_0380588']
    require(selected == expected and len(set(selected)) == 4, 'Fixed video selection changed')
    heldout = {ident for ident, row in rows.items() if row['split'] in ('test', 'legacy')}
    require(set(selected) <= heldout, 'Selected videos are outside test/legacy')
    for label in HISTORICAL_VARIANTS:
        require(set(old['variants'][label]) == heldout, f'Historical {label} result identities differ')
        for ident, row in old['variants'][label].items():
            require(row['split'] == rows[ident]['split'], 'Historical result split differs')
            require(all(math.isfinite(float(row[k])) for k in METRIC_KEYS), 'Nonfinite historical latent metrics')
    for ident in selected:
        for label in ('pretrained', 'long_ewarp', 'cmn', 'reference'):
            path = args.cmn_evaluation / 'latents' / ident / f'{label}.pt'
            require(file_record(path) == old['latent_files'][ident][label], 'Historical saved latent changed')
    fingerprints.update(train_summary_sha256=sha256(args.run / 'summary.json'),
                        checkpoint_sha256=sha256(args.run / 'best.pt'),
                        cmn_summary_sha256=sha256(args.cmn_evaluation / 'summary.json'),
                        cmn_latent_summary_sha256=sha256(old_snapshot_path),
                        evaluation_source_sha256=source_hashes())
    return rows, run, old, fingerprints


def paired_latent_deltas(variants):
    """Per-clip new-minus-baseline values; no claim of statistical significance."""
    result = {}
    for baseline in HISTORICAL_VARIANTS:
        samples = {ident: {'split': row['split'], **{
            key: float(row[key]) - float(variants[baseline][ident][key])
            for key in ('latent_mse', 'ewarp_l1')}}
                   for ident, row in variants['hybrid'].items()}
        result[baseline] = {'samples': samples, 'mean_by_split': {
            split: {key: statistics.mean(row[key] for row in samples.values() if row['split'] == split)
                    for key in ('latent_mse', 'ewarp_l1')}
            for split in ('test', 'legacy')}}
    return result


def evaluate_latents(args):
    require(not args.output.exists(), 'Use a fresh output directory for latent evaluation')
    rows, run, old, fingerprints = inputs(args)
    saved = tensor_load(args.run / 'best.pt')
    require(saved['contract'] == run['contract'], 'Checkpoint/training contract differs')
    require(saved['step'] == run['best_step'], 'Checkpoint is not the selected dev step')
    require(saved['model_config'] == {'width': 64, 'blocks': 2}, 'Expected warmstarted CMN width64/blocks2')
    require(saved['best_step'] == run['best_step'] and float(saved['best_score']) == selected_score(run),
            'Checkpoint selected-step/score differs from training summary')
    args.output.mkdir(parents=True)
    selected = old['selected_videos']
    summary = {'status': 'evaluating', 'plan_sha256': fingerprints['plan_sha256'],
               'input_fingerprints': fingerprints, 'selected_videos': selected,
               'video_selection': old['video_selection'], 'frames': 120, 'fps': 24, 'seconds_per_video': 5,
               'comparison_order': list(DECODE_VARIANTS), 'decode_latent_mapping': DECODE_VARIANTS,
               'variants': {'hybrid': {}, **{label: old['variants'][label] for label in HISTORICAL_VARIANTS}},
               'reload_parity': {'hybrid': {}}, 'causality': {'hybrid': {}},
               'latent_files': {}, 'saved_prediction_sha256': {}, 'historical_latent_sha256': {},
               'train_run': {key: run[key] for key in ('completed_updates', 'best_step', 'best_validation_score', 'best_score',
                    'best_checkpoint_sha256', 'parameter_count', 'prior_training_updates') if key in run},
               'training_config': run['config'], 'historical_cmn_run': old['train_run'],
               'historical_conv3d_runs': old['historical_train_runs'],
               'selection': run['config'].get('selection', 'Lowest fixed dev-patch RGB L1 +0.2 LPIPS; test never selects weights'),
               'metric_comparison': {'same_plan': True, 'equal_compute': False,
                   'historical_latent_metrics': 'Copied from SHA-bound completed CMN evaluation; identical pairs and metric definitions',
                   'hybrid': 'Warmstarted CMN architecture after complementary teacher/pixel supervision; no inference ensemble',
                   'old_causal_ewarp': 'Historical345M causal Conv3d upscaler, not Method2',
                   'common_latent_objective': 'HR latent MSE and eWarp components; not the hybrid composite training objective'},
               'test_reuse': 'All16 test and3 legacy identities were inspected in earlier trials; not a new blind test',
               'latent_metric_scope': 'All16 test +3 legacy; first36 real-anchor latents and35 valid warp pairs',
               'pixel_metric_scope': 'Only four fixed videos (3 retained test +1 legacy); full120frames, not the all16 test mean',
               'pixel_metric_targets': {'primary': 'Matched HR latent decoded by the same frozen VAE',
                                        'secondary': 'Original120 HR source RGB frames, including VAE reconstruction error'},
               'lpips_recipe': {'network': 'alex', 'version': '0.1', 'frame_batch_size': args.lpips_batch_size,
                                'input': 'Uncompressed decoded uint8 RGB, converted to [-1,1]; no resize/crop',
                                'aggregation': 'Mean of per-frame scalar LPIPS over120frames'},
               'causal_scope': 'Hybrid upsampler on fixed latents only; original frozen H3 decoder is noncausal; upstream encoding is outside this audit',
               'end_to_end_causal': False, 'decoder_trained': False,
               'padding_context': old['padding_context']}
    atomic_json(args.output / 'summary.json', summary)
    device = select_device(args.device)
    model = H3LatentUpsampler(**saved['model_config']).to(device)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.eval().requires_grad_(False)
    del saved
    pairs = load_pairs(args.pairs / 'test.jsonl') + load_pairs(args.pairs / 'legacy.jsonl')
    require({p['sample_id'] for p in pairs} == set(old['variants']['cmn']), 'Loaded held-out pairs differ')
    with torch.no_grad():
        for pair in pairs:
            ident, split = pair['sample_id'], pair['split']
            draft, reference, flow, valid = tensors(pair, device)
            prediction_path = args.run / 'predictions' / split / f'{ident}.pt'
            expected = tensor_load(prediction_path)
            require(expected['sample_id'] == ident and expected['split'] == split, 'Prediction identity differs')
            require(expected['pair_sha256'] == pair['pair_sha256'], 'Prediction pair differs')
            pred = model(draft)
            check = parity(pred[0], expected['latent'], args.tolerance)
            summary['reload_parity']['hybrid'][ident] = check
            summary['saved_prediction_sha256'][ident] = sha256(prediction_path)
            atomic_json(args.output / 'summary.json', summary)
            require(check['passed'], f'Independent best.pt reload differs: {ident}')
            res = numeric_metrics(objective(pred, reference, flow, valid, .1))
            summary['variants']['hybrid'][ident] = {'split': split, 'metric_weight': .1, **res}
            if ident in selected:
                dst = args.output / 'latents' / ident
                summary['latent_files'][ident] = {}
                summary['historical_latent_sha256'][ident] = {}
                for label in LATENT_VARIANTS:
                    if label == 'hybrid':
                        value = pred[0].cpu()
                    else:
                        path = args.cmn_evaluation / 'latents' / ident / f'{label}.pt'
                        value = tensor_load(path)['latent'].float()
                        require(tuple(value.shape) == (24, 37, 24, 42) and bool(torch.isfinite(value).all()), 'Bad historical latent')
                        summary['historical_latent_sha256'][ident][label] = sha256(path)
                        if label == 'reference':
                            require(torch.equal(value, pair['reference'].float()), 'Historical HR reference differs')
                        else:
                            old_metrics = numeric_metrics(objective(value[None].to(device), reference, flow, valid, .1))
                            require(all(math.isclose(old_metrics[k], float(old['variants'][label][ident][k]),
                                                     abs_tol=1e-5, rel_tol=1e-5) for k in METRIC_KEYS),
                                    f'Historical latent/metric mismatch: {label}/{ident}')
                    path = dst / f'{label}.pt'
                    atomic_torch_save(path, {'latent': value, 'sample_id': ident, 'split': split,
                                            'pair_sha256': pair['pair_sha256']})
                    summary['latent_files'][ident][label] = file_record(path)
                causal = causal_checks(model, draft, args.tolerance)
                sample_max = max(v['max_abs'] for v in causal['checks'].values())
                causal.update(absolute_max_abs=sample_max, absolute_max_at_most_1e_minus_4=sample_max <= 1e-4)
                summary['causality']['hybrid'][ident] = causal
                atomic_json(args.output / 'summary.json', summary)
                require(causal['passed'] and sample_max <= 1e-4, f'Hybrid causality failed: {ident}')
            atomic_json(args.output / 'summary.json', summary)
            print(json.dumps({'latent_evaluated': ident, 'reload_max_abs': check['max_abs']}), flush=True)
            del draft, reference, flow, valid, pred, expected
        benchmark_pair = next(p for p in pairs if p['sample_id'] == selected[0])
        summary['warmed_upsampler_timing'] = benchmark(model, benchmark_pair['draft'][None].to(device), device, args.timing_repeats)
    expected_ids = {p['sample_id'] for p in pairs}
    require(len(expected_ids) == 19 and set(summary['reload_parity']['hybrid']) == expected_ids, 'Expected19 reload checks')
    require(set(summary['latent_files']) == set(summary['causality']['hybrid']) == set(selected), 'Expected four fixed causal/video samples')
    summary['causality_test_count'] = sum(len(v['checks']) for v in summary['causality']['hybrid'].values())
    require(summary['causality_test_count'] == 36 and all(len(v['checks']) == 9 for v in summary['causality']['hybrid'].values()),
            'Expected4samples x9 future/prefix/chunk checks')
    summary.update(mean_by_split=mean_by_split(summary['variants']),
                   paired_latent_deltas=paired_latent_deltas(summary['variants']),
                   causality_passed=all(v['passed'] for v in summary['causality']['hybrid'].values()),
                   reload_parity_passed=all(v['passed'] for v in summary['reload_parity']['hybrid'].values()),
                   causality_max_abs=max(v['absolute_max_abs'] for v in summary['causality']['hybrid'].values()),
                   reload_parity_max_abs=max(v['max_abs'] for v in summary['reload_parity']['hybrid'].values()),
                   parity_tolerance={'atol': args.tolerance, 'rtol': args.tolerance, 'causality_additional_absolute_limit': 1e-4},
                   status='latents_completed')
    atomic_json(args.output / 'latent_summary.json', summary)
    atomic_json(args.output / 'summary.json', summary)
    del model, pairs
    empty_cache(device)
    print(json.dumps({'status': 'latents_completed', 'reload_checks': 19, 'causal_checks': 36}), flush=True)


def verified_latent_summary(args):
    path = args.output / 'latent_summary.json'
    summary = read_json(path)
    require(summary['status'] == 'latents_completed', 'No completed immutable latent evaluation')
    rows, _, old, fingerprints = inputs(args)
    require(summary['input_fingerprints'] == fingerprints, 'Inputs/source changed after latent evaluation')
    require(summary['causality_passed'] and summary['reload_parity_passed'], 'Latent checks failed')
    require(set(summary['latent_files']) == set(summary['selected_videos']), 'Latent sample set differs')
    for ident, files in summary['latent_files'].items():
        require(set(files) == set(LATENT_VARIANTS), 'Latent variant set differs')
        for label, expected in files.items():
            require(file_record(args.output / 'latents' / ident / f'{label}.pt') == expected, 'Saved latent changed')
    return summary, rows, old, sha256(path)


def load_lpips(device):
    """Only load already cached AlexNet + LPIPS v0.1 weights; never download."""
    import lpips
    backbone = Path(torch.hub.get_dir()) / 'checkpoints' / 'alexnet-owt-7be5be79.pth'
    linear = Path(lpips.__file__).resolve().parent / 'weights' / 'v0.1' / 'alex.pth'
    require(backbone.is_file() and linear.is_file(), 'Offline LPIPS alex weights are missing; prepare the torch hub/package cache first')
    model = lpips.LPIPS(net='alex', version='0.1', pretrained=True, pnet_rand=False,
                        model_path=str(linear), verbose=False).eval().requires_grad_(False).to(device)
    # Bind the actual loaded tensors as well as the files. This catches changes
    # in package construction/default scaling even when filenames stay stable.
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode()); digest.update(array.tobytes())
    identity = {'network': 'alex', 'version': '0.1', 'pretrained': True, 'spatial': False,
                'input_range': [-1, 1], 'resize': False, 'normalize_argument': False,
                'backbone': file_record(backbone), 'linear_weights': file_record(linear),
                'implementation_sha256': sha256(Path(inspect.getfile(type(model)))),
                'loaded_state_sha256': digest.hexdigest()}
    require(not any(p.requires_grad for p in model.parameters()), 'LPIPS must be frozen')
    return model, identity


@torch.no_grad()
def lpips_frames(model, output, reference, device, batch_size):
    require(tuple(output.shape) == tuple(reference.shape) == (120, 384, 672, 3), 'LPIPS RGB shape differs')
    require(output.dtype == reference.dtype == torch.uint8, 'LPIPS expects original uncompressed uint8 RGB')
    values = []
    for start in range(0, 120, batch_size):
        a = output[start:start + batch_size].to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).contiguous() / 127.5 - 1.
        b = reference[start:start + batch_size].to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).contiguous() / 127.5 - 1.
        value = model(a, b, normalize=False).detach().float().flatten().cpu()
        require(value.numel() == len(a) and bool(torch.isfinite(value).all()), 'LPIPS must return one finite scalar per frame')
        values.extend(float(v) for v in value)
    require(len(values) == 120, 'Incomplete LPIPS frame coverage')
    return values


def pixel_metrics(rgb, reference, source, perceptual, device, batch_size):
    """Compute regional LPIPS from the same per-frame full-resolution values."""
    primary = lpips_frames(perceptual, rgb, reference, device, batch_size)
    secondary = lpips_frames(perceptual, rgb, source, device, batch_size)

    def region(start, stop):
        result = {'vs_decoded_reference': metric(rgb[start:stop], reference[start:stop]),
                  'vs_source_rgb_secondary': metric(rgb[start:stop], source[start:stop])}
        result['vs_decoded_reference']['lpips_alex_v0_1'] = statistics.mean(primary[start:stop])
        result['vs_source_rgb_secondary']['lpips_alex_v0_1'] = statistics.mean(secondary[start:stop])
        return result

    return {**region(0, 120), 'prefix102': region(0, 102), 'tail18': region(102, 120),
            'lpips_per_frame': {'vs_decoded_reference': primary, 'vs_source_rgb_secondary': secondary}}


def check_historical_pixels(actual, prior):
    """The freshly decoded baseline must reproduce old raw-RGB metrics."""
    for region in (None, 'prefix102', 'tail18'):
        a, b = (actual, prior) if region is None else (actual[region], prior[region])
        for target in ('vs_decoded_reference', 'vs_source_rgb_secondary'):
            for key in ('mse', 'psnr_db', 'temporal_difference_mse'):
                require(math.isclose(float(a[target][key]), float(b[target][key]), abs_tol=1e-5, rel_tol=1e-5),
                        f'Redecoded historical pixels differ: {region}/{target}/{key}')


def audit_media(directory):
    evidence = {}
    for label in (*DECODE_VARIANTS, 'comparison'):
        width, height = (672 * len(DECODE_VARIANTS), 412) if label == 'comparison' else (672, 384)
        evidence[f'{label}.mp4'] = audit_video(directory / f'{label}.mp4', width, height)
    path = directory / 'comparison.png'
    with Image.open(path) as image:
        require(image.format == 'PNG' and image.size == (672 * len(DECODE_VARIANTS), 1236), 'Contact-sheet geometry differs')
        image.verify()
    with Image.open(path) as image:
        image.load()
    evidence['comparison.png'] = {'passed': True, 'width': 672 * len(DECODE_VARIANTS), 'height': 1236, **file_record(path)}
    return evidence


def decode_shard(args):
    summary, rows, old, snapshot_hash = verified_latent_summary(args)
    labels = [ident for i, ident in enumerate(summary['selected_videos']) if i % args.num_shards == args.shard_index]
    require(bool(labels), 'Empty decode shard')
    done_path = args.output / f'decode_complete_shard{args.shard_index}.json'
    require(not done_path.exists(), 'Decode shard already completed; use finalize')
    directories = [args.output / 'decoded' / ident for ident in labels]
    require(not any(p.exists() for p in directories), 'Decode output exists; preserve it and use a fresh evaluation output')
    base_identity = vae_fingerprint(args.base_model)
    device = select_device(args.device)
    lpips_device = select_device(args.lpips_device) if args.lpips_device else device
    perceptual, perceptual_identity = load_lpips(lpips_device)
    vae = load_video_vae(args.base_model, device).eval().requires_grad_(False)
    require(not any(p.requires_grad for p in vae.parameters()), 'VAE must be frozen')
    decoder_identity = {'weights': base_identity, 'implementation_sha256': sha256(Path(inspect.getfile(type(vae)))),
                        'recipe': 'data_utils.load_video_vae + decode_latent, original temporal windows, FP32, spatial tile256/overlap64'}
    require(decoder_identity == old['decoder_fingerprint'], 'Decoder differs from completed CMN baseline evaluation')
    output, media = {}, {}
    batch_size = summary['lpips_recipe']['frame_batch_size']
    for ident, dst in zip(labels, directories):
        dst.mkdir(parents=True)
        row = rows[ident]
        pair_path = Path(row['path'])
        pair_path = pair_path if pair_path.is_absolute() else args.pairs / pair_path
        require(sha256(pair_path) == row['sha256'], 'Decode source pair changed')
        pair = tensor_load(pair_path)
        source = pair['source_pixels']
        require(tuple(source.shape) == (120, 384, 672, 3) and source.dtype == torch.uint8, 'Source RGB geometry/dtype differs')
        pixels, timings, codecs = {}, {}, {}
        with torch.no_grad():
            for label, latent_label in DECODE_VARIANTS.items():
                latent = tensor_load(args.output / 'latents' / ident / f'{latent_label}.pt')['latent']
                if label == 'reference':
                    require(torch.equal(latent.float(), pair['reference'].float()), 'Reference differs from pair')
                synchronize(device)
                tick = time.perf_counter()
                rgb = decode_latent(vae, latent, device, keep_frames=120)
                synchronize(device)
                timings[label] = time.perf_counter() - tick
                require(tuple(rgb.shape) == (120, 384, 672, 3) and rgb.dtype == torch.uint8, 'Decoded RGB shape/dtype differs')
                pixels[label] = rgb
                codecs[label] = write_mp4(rgb, dst / f'{label}.mp4', fps=24.)
                print(json.dumps({'decoded': ident, 'variant': label, 'vae_seconds': timings[label]}), flush=True)
        entry = {'split': row['split'], 'decode_seconds': timings, 'video_codecs': codecs,
                 'metric_domain': 'Uncompressed uint8 RGB; PSNR/temporal use[0,1], LPIPS uses[-1,1]',
                 'frame_count': 120, 'variants': {}, 'historical_pixel_metrics_verified': []}
        synchronize(lpips_device)
        tick = time.perf_counter()
        for label, rgb in pixels.items():
            if label == 'reference':
                continue
            entry['variants'][label] = pixel_metrics(rgb, pixels['reference'], source, perceptual, lpips_device, batch_size)
            if label != 'hybrid':
                check_historical_pixels(entry['variants'][label], old['pixel_metrics'][ident]['variants'][label])
                entry['historical_pixel_metrics_verified'].append(label)
        entry['reference_vs_source_rgb_secondary'] = metric(pixels['reference'], source)
        ref_lpips = lpips_frames(perceptual, pixels['reference'], source, lpips_device, batch_size)
        entry['reference_vs_source_rgb_secondary']['lpips_alex_v0_1'] = statistics.mean(ref_lpips)
        synchronize(lpips_device)
        entry['pixel_metric_seconds'] = time.perf_counter() - tick
        comparisons(pixels, dst, fps=24., crf=16, comparison_video=True)
        media[ident] = audit_media(dst)
        output[ident] = entry
        atomic_json(args.output / f'pixel_metrics_shard{args.shard_index}.json', output)
        print(json.dumps({'pixel_evaluated': ident, 'frames': 120, 'variants': list(entry['variants'])}), flush=True)
        del pair, pixels, source
    require(sha256(args.output / 'latent_summary.json') == snapshot_hash, 'Latent snapshot changed during decode')
    metrics_path = args.output / f'pixel_metrics_shard{args.shard_index}.json'
    atomic_json(done_path, {'status': 'completed', 'shard_index': args.shard_index, 'num_shards': args.num_shards,
                'samples': labels, 'latent_summary_sha256': snapshot_hash, 'metrics_sha256': sha256(metrics_path),
                'decoder_fingerprint': decoder_identity, 'lpips_fingerprint': perceptual_identity,
                'lpips_device': str(lpips_device), 'lpips_batch_size': batch_size,
                'media': media, 'comparison_order': list(DECODE_VARIANTS)})
    del vae, perceptual
    empty_cache(device)
    if str(lpips_device) != str(device):
        empty_cache(lpips_device)
    print(json.dumps({'decode_shard_completed': args.shard_index, 'samples': labels}), flush=True)


def summarize_pixels(combined):
    """Four-video descriptive means and paired differences, never all-test means."""
    variants = tuple(label for label in DECODE_VARIANTS if label != 'reference')
    groups = {'all_four_fixed_videos': list(combined),
              'three_selected_test_videos': [i for i, row in combined.items() if row['split'] == 'test'],
              'one_selected_legacy_video': [i for i, row in combined.items() if row['split'] == 'legacy']}
    result = {}
    for name, ids in groups.items():
        require(bool(ids), 'Empty fixed-video metric group')
        means = {label: {target: {key: statistics.mean(float(combined[i]['variants'][label][target][key]) for i in ids)
                                  for key in PIXEL_KEYS}
                        for target in ('vs_decoded_reference', 'vs_source_rgb_secondary')} for label in variants}
        deltas = {label: {target: {key: means['hybrid'][target][key] - means[label][target][key] for key in PIXEL_KEYS}
                         for target in ('vs_decoded_reference', 'vs_source_rgb_secondary')}
                  for label in variants if label != 'hybrid'}
        result[name] = {'sample_ids': ids, 'count': len(ids), 'mean': means,
                        'hybrid_minus_baseline': deltas}
    return {'scope': 'Only four fixed videos; each video has equal weight; PSNR is mean per-video PSNR, not pooled-MSE PSNR',
            'delta_direction': 'Positive PSNR favors hybrid; negative MSE/temporal/LPIPS favors hybrid',
            'groups': result}


def validate_pixel_entry(entry, split):
    require(entry['split'] == split and entry['frame_count'] == 120, 'Pixel identity/frame coverage differs')
    labels = set(DECODE_VARIANTS) - {'reference'}
    require(set(entry['variants']) == labels, 'Pixel variant coverage differs')
    require(set(entry['historical_pixel_metrics_verified']) == labels - {'hybrid'}, 'Historical pixel verification missing')
    for value in entry['variants'].values():
        for region in (value, value['prefix102'], value['tail18']):
            for target in ('vs_decoded_reference', 'vs_source_rgb_secondary'):
                require(all(math.isfinite(float(region[target][k])) for k in PIXEL_KEYS), 'Nonfinite/incomplete pixel metrics')
        for target in ('vs_decoded_reference', 'vs_source_rgb_secondary'):
            frames = value['lpips_per_frame'][target]
            require(len(frames) == 120 and all(math.isfinite(float(v)) for v in frames), 'Incomplete LPIPS frame values')
            for start, stop, region in ((0, 120, value), (0, 102, value['prefix102']), (102, 120, value['tail18'])):
                require(math.isclose(statistics.mean(frames[start:stop]), region[target]['lpips_alex_v0_1'], abs_tol=1e-12, rel_tol=1e-12),
                        'LPIPS aggregate does not match per-frame evidence')


def finalize(args):
    summary, rows, old, snapshot_hash = verified_latent_summary(args)
    combined, media, seen = {}, {}, set()
    decoder_identity = perceptual_identity = None
    for index in range(args.num_shards):
        done = read_json(args.output / f'decode_complete_shard{index}.json')
        path = args.output / f'pixel_metrics_shard{index}.json'
        part = read_json(path)
        expected = {ident for i, ident in enumerate(summary['selected_videos']) if i % args.num_shards == index}
        require(done['status'] == 'completed' and done['shard_index'] == index and done['num_shards'] == args.num_shards, 'Shard identity differs')
        require(done['latent_summary_sha256'] == snapshot_hash and done['metrics_sha256'] == sha256(path), 'Shard fingerprint differs')
        require(set(part) == set(done['samples']) == set(done['media']) == expected and not (seen & expected), 'Missing/overlapping shard samples')
        require(done['comparison_order'] == list(DECODE_VARIANTS), 'Video column order differs')
        require(done['lpips_batch_size'] == summary['lpips_recipe']['frame_batch_size'], 'LPIPS batching differs')
        if decoder_identity is None:
            decoder_identity, perceptual_identity = done['decoder_fingerprint'], done['lpips_fingerprint']
        require(done['decoder_fingerprint'] == decoder_identity == old['decoder_fingerprint'], 'VAE fingerprint differs')
        require(done['lpips_fingerprint'] == perceptual_identity, 'LPIPS network/weights differ across shards')
        for ident in expected:
            validate_pixel_entry(part[ident], rows[ident]['split'])
            for label in ('pretrained', 'old_causal_ewarp', 'cmn'):
                check_historical_pixels(part[ident]['variants'][label], old['pixel_metrics'][ident]['variants'][label])
            actual = audit_media(args.output / 'decoded' / ident)
            require(actual == done['media'][ident], 'Media changed or failed full decode')
            media[ident] = actual
        combined.update(part); seen.update(expected)
    require(seen == set(summary['selected_videos']) and len(seen) == 4, 'Not all four fixed videos are complete')
    # Use the predefined order, independent of decode completion/shard order.
    combined = {ident: combined[ident] for ident in summary['selected_videos']}
    summary.update(status='completed', pixel_metrics=combined, pixel_comparison=summarize_pixels(combined),
                   decode_shards=args.num_shards, media_audit=media, decoder_fingerprint=decoder_identity,
                   lpips_fingerprint=perceptual_identity, latent_summary_sha256=snapshot_hash, decode_complete=True)
    atomic_json(args.output / 'summary.json', summary)
    print(json.dumps({'completed': True, 'latent_samples': 19, 'pixel_samples': 4,
                      'full_decoded_mp4_files': 24, 'causality_checks': summary['causality_test_count']}), flush=True)


def main():
    root = Path(os.environ.get('EXP_ROOT', Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=root / 'upscaler_hybrid/cmn_kd_pixel_steps2000')
    parser.add_argument('--pairs', type=Path, default=root / 'upscaler_ultraflash/pairs')
    parser.add_argument('--cmn-evaluation', type=Path, default=root / 'upscaler_cmn/evaluation')
    parser.add_argument('--output', type=Path, default=root / 'upscaler_hybrid/evaluation')
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--base-model', default=os.environ.get('H3_BASE_MODEL'))
    parser.add_argument('--lpips-device', default=None, help='Defaults to --device; cpu is supported')
    parser.add_argument('--lpips-batch-size', type=int, default=4, help='Stored by initial latent evaluation; reused by every shard')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--tolerance', type=float, default=3e-5)
    parser.add_argument('--timing-repeats', type=int, default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--decode-only', action='store_true')
    mode.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    require(1 <= args.num_shards <= 4 and 0 <= args.shard_index < args.num_shards, 'Invalid shard identity')
    require(args.tolerance == 3e-5 and args.timing_repeats > 0, 'Tolerance is fixed at3e-5; timing repeats must be positive')
    require(1 <= args.lpips_batch_size <= 8, 'LPIPS batch size must be1..8')
    torch.set_num_threads(4)
    if args.finalize:
        finalize(args)
    elif args.decode_only:
        require(args.base_model is not None, '--base-model or H3_BASE_MODEL is required for decoding')
        decode_shard(args)
    else:
        evaluate_latents(args)


if __name__ == '__main__':
    main()
