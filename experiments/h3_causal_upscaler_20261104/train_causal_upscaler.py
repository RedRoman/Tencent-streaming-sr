"""Supervised fine-tuning after explicit LBH/Open-H3 upscaler causal surgery.

Fixed training schedule; holdout never influences updates or checkpoint choice.
No H3 refiner, DMD, or VAE gradient is involved. The saved optimizer is retained
for future tooling, but this driver deliberately does not implement resume.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from causal_upscaler import CausalUpscalerAdapter, initialize_aligned_past_
from data_utils import empty_cache, select_device, synchronize, tensor_load
from prepare_data import load_author_upscaler
from train_upscaler_trial import (atomic_json, atomic_torch_save, enable_activation_checkpointing,
                                 losses, memory_stats, metrics, parameter_sample, read_pairs,
                                 save_latent, sha256, to_cpu)


def compare(left, right, tolerance):
    left, right = left.detach().float().cpu(), right.detach().float().cpu()
    if left.shape != right.shape:
        return {'passed': False, 'error': 'shape mismatch', 'left_shape': list(left.shape),
                'right_shape': list(right.shape)}
    if not (torch.isfinite(left).all() and torch.isfinite(right).all()):
        return {'passed': False, 'error': 'nonfinite tensor'}
    delta = left.double() - right.double()
    maximum = float(delta.abs().max())
    return {'max_abs': maximum, 'rms': float(delta.square().mean().sqrt()),
            'bit_exact': bool(torch.equal(left, right)), 'tolerance': tolerance,
            'passed': maximum <= tolerance}


def evaluate(adapter, pairs, output, label, temporal_weight, device):
    results = {}
    with torch.no_grad():
        for pair in pairs:
            started = time.perf_counter()
            draft = pair['draft'][None].to(device)
            value = adapter.native_forward(draft, pair['reference'].shape[-3:])[0].cpu()
            synchronize(device)
            result = metrics(value, pair['reference'], temporal_weight)
            result.update({'split': pair['split'], 'seconds': time.perf_counter()-started,
                           'latent_path': save_latent(output, pair, label, value)})
            results[pair['sample_id']] = result
            print(json.dumps({'evaluation': label, 'sample_id': pair['sample_id'], **result}), flush=True)
    return results


def save_training_checkpoint(output, model, optimizer, step, signature, config, adapter_config):
    directory = output / 'checkpoint'
    atomic_torch_save(directory / 'model.pt', {
        'model_state_dict': to_cpu(model.state_dict()), 'step': step,
        'adapter_config': adapter_config, 'data_signature': signature})
    atomic_torch_save(directory / 'trainer.pt', {
        'optimizer_state_dict': to_cpu(optimizer.state_dict()), 'step': step,
        'adapter_config': adapter_config, 'data_signature': signature, 'config': config,
        'torch_rng_state': torch.get_rng_state(), 'resume_supported': False})
    atomic_json(directory / 'checkpoint.json', {
        'step': step, 'model': 'model.pt', 'trainer': 'trainer.pt',
        'adapter_config': adapter_config, 'data_signature': signature,
        'resume_supported': False,
        'reload_rule': 'Load saved state into fresh author model, install causal adapter; NEVER apply aligned_past again'})
    return directory


def reload_causal_model(args, checkpoint_path, device, expected_adapter_config):
    """Saved weights already contain their initialization; do NOT transform them."""
    saved = tensor_load(checkpoint_path)
    if saved['adapter_config'] != expected_adapter_config:
        raise ValueError('Checkpoint adapter configuration differs from this trial')
    node = load_author_upscaler(args.upstream, args.upscaler, device)
    model = node.load_model('upscaler', device, 'fp32')
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.eval().requires_grad_(False)
    # Record the already-applied initialization to reject accidental reapplication.
    model._aligned_past_initialized = expected_adapter_config['initialization'] == 'aligned_past'
    adapter = CausalUpscalerAdapter(node, model)
    del saved
    return node, model, adapter


def semantic_checks(adapter, pairs, args, device):
    """Post-training diagnostics only; no schedule/checkpoint selection uses them."""
    records = {}
    for pair in pairs:
        sample_id = pair['sample_id']
        draft = pair['draft'][None].to(device)
        target = pair['reference'].shape[-3:]
        if draft.shape[2] != 13:
            raise ValueError('The specified cut/chunk diagnostics require 13 latent frames')
        sample_checks = {}
        evidence = args.output / 'semantic_evidence' / sample_id
        with torch.no_grad():
            full = adapter.native_forward(draft, target).cpu()
            atomic_torch_save(evidence / 'full.pt', {'latent': full[0]})
            for cut in (1, 5, 10):
                seed = args.seed + 1000 + cut
                poison = draft.clone()
                noise = torch.randn(poison[:, :, cut:].shape,
                                    generator=torch.Generator().manual_seed(seed)).to(device)
                poison[:, :, cut:] = 3 * noise
                assert torch.equal(poison[:, :, :cut], draft[:, :, :cut])
                changed = adapter.native_forward(poison, target).cpu()
                prefix = adapter.native_forward(draft[:, :, :cut], (cut, *target[-2:])).cpu()
                change_record = compare(full[:, :, :cut], changed[:, :, :cut], args.tolerance)
                change_record.update({'cut': cut, 'replacement_seed': seed,
                                      'future_input_max_abs_change': float((poison[:, :, cut:]-draft[:, :, cut:]).abs().max())})
                if change_record['future_input_max_abs_change'] <= 0:
                    change_record['passed'] = False
                prefix_record = compare(full[:, :, :cut], prefix, args.tolerance)
                prefix_record['cut'] = cut
                sample_checks[f'future_cut{cut}'] = change_record
                sample_checks[f'prefix_cut{cut}'] = prefix_record
                # Actual compared outputs are small and retained even on failure.
                atomic_torch_save(evidence / f'cut{cut}.pt', {
                    'future_changed_prefix': changed[0, :, :cut], 'separate_prefix': prefix[0],
                    'future_check': change_record, 'prefix_check': prefix_record})
            for partition in ([5, 5, 3], [1] * 13, [2, 3, 1, 7]):
                adapter.reset_stream()
                chunks, offset = [], 0
                for frames in partition:
                    chunks.append(adapter.stream_step(draft[:, :, offset:offset+frames], target[-2:]).cpu())
                    offset += frames
                streamed = torch.cat(chunks, dim=2)
                value = compare(full, streamed, args.tolerance)
                value.update({'partition': partition, 'stream_state': adapter.stream_state()})
                key = 'chunks_' + '_'.join(map(str, partition))
                sample_checks[key] = value
                atomic_torch_save(evidence / f'{key}.pt', {'latent': streamed[0], 'check': value})
            adapter.reset_stream()
        records[sample_id] = {'split': pair['split'], 'checks': sample_checks,
                              'passed': all(value['passed'] for value in sample_checks.values())}
        atomic_json(evidence / 'checks.json', records[sample_id])
        print(json.dumps({'semantic_checks': sample_id, **records[sample_id]}), flush=True)
    return {'scope': 'All three paired clips after training; fixed input latents, no VAE/refiner causality claim',
            'samples': records, 'passed': all(value['passed'] for value in records.values())}


def run(args, summary):
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    train, holdout = read_pairs(args.train, 'train'), read_pairs(args.holdout, 'holdout')
    pairs = train + holdout
    identifiers = [pair['sample_id'] for pair in pairs]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('Train/holdout sample IDs must be unique and disjoint')
    if any(pair['draft'].shape[1] != 13 for pair in pairs):
        raise ValueError('This controlled causal trial requires 13 latent frames')
    signature = {
        'train': [{'sample_id': pair['sample_id'], 'sha256': pair['sha256']} for pair in train],
        'holdout': [{'sample_id': pair['sample_id'], 'sha256': pair['sha256']} for pair in holdout],
        'released_upscaler_sha256': sha256(args.upscaler)}
    summary['data_signature'] = signature
    device = select_device(args.device)
    backend = getattr(torch, device.type, None)
    if backend is not None and hasattr(backend, 'reset_peak_memory_stats'):
        backend.reset_peak_memory_stats(device)
    started = time.perf_counter()
    node = load_author_upscaler(args.upstream, args.upscaler, device)
    model = node.load_model('upscaler', device, 'fp32')
    if args.initialization == 'aligned_past':
        initialization = initialize_aligned_past_(model)
    else:
        initialization = {'kind': 'shifted', 'description': 'Published kernel values unchanged; causal left padding changes temporal offsets'}
    adapter = CausalUpscalerAdapter(node, model)
    adapter_config = {
        'version': 1, 'initialization': args.initialization,
        'temporal_padding': 'left_only_zero', 'groupnorm': 'per_frame',
        'normalization': 'D(U(N(native_normalized_H3_latent)))',
        'spatial_scale': 2., 'temporal_chunking': False, 'dtype': 'float32',
        'saved_weights_already_initialized': True,
        'reload_must_not_reinitialize_weights': True}
    summary.update({'initialization_report': initialization, 'adapter_config': adapter_config,
                    'structure': adapter.structure,
                    'parameter_count': sum(value.numel() for value in model.parameters())})
    initial_path = args.output / 'initial_model.pt'
    atomic_torch_save(initial_path, {'model_state_dict': to_cpu(model.state_dict()), 'step': 0,
                                     'adapter_config': adapter_config, 'data_signature': signature})
    if args.activation_checkpointing:
        enable_activation_checkpointing(model)
    model.eval().requires_grad_(True)
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay, foreach=False)
    initial_sample = parameter_sample(model)
    previous_sample = initial_sample.clone()
    summary['status'] = 'training'
    atomic_json(args.output / 'summary.json', summary)
    train_started = time.perf_counter()
    step_records = []
    summary['training_steps'] = step_records
    with (args.output / 'steps.jsonl').open('w') as log:
        for step in range(args.steps):
            step_started = time.perf_counter()
            pair = train[step % len(train)]
            draft, reference = pair['draft'][None].to(device), pair['reference'][None].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = adapter.native_forward(draft, reference.shape[-3:])
            loss, mse, temporal_mse = losses(prediction, reference, args.temporal_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss at step {step+1}')
            loss.backward()
            missing = [name for name, value in model.named_parameters() if value.grad is None]
            if missing:
                atomic_json(args.output / 'disconnected_parameters.json', {'step': step+1, 'parameters': missing})
                raise RuntimeError(f'{len(missing)} parameters disconnected from loss')
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float('inf'),
                                                       error_if_nonfinite=True, foreach=False)
            if not float(grad_norm) > 0:
                raise RuntimeError('Zero global gradient norm')
            optimizer.step()
            synchronize(device)
            sample = parameter_sample(model)
            change = (sample - previous_sample).abs()
            if not torch.isfinite(sample).all() or not bool((change > 0).any()):
                raise RuntimeError('Optimizer produced nonfinite or unchanged sampled weights')
            previous_sample = sample
            record = {
                'step': step+1, 'sample_id': pair['sample_id'], 'loss': float(loss.detach()),
                'latent_mse': float(mse.detach()), 'temporal_difference_mse': float(temporal_mse.detach()),
                'grad_norm': float(grad_norm), 'gradients_finite': True,
                'parameters_with_grad': len(parameters), 'all_parameters_connected': True,
                'sampled_update_max_abs': float(change.max()), 'sampled_update_mean_abs': float(change.mean()),
                'sampled_changed_values': int((change > 0).sum()), 'sampled_parameter_values': len(change),
                'lr': optimizer.param_groups[0]['lr'], 'seconds': time.perf_counter()-step_started}
            step_records.append(record)
            summary['completed_updates'] = step+1
            log.write(json.dumps(record, allow_nan=False)+'\n')
            log.flush()
            print(json.dumps(record), flush=True)
            del draft, reference, prediction, loss, mse, temporal_mse, grad_norm
            if args.checkpoint_every and (step+1) % args.checkpoint_every == 0 and step+1 < args.steps:
                save_training_checkpoint(args.output, model, optimizer, step+1, signature, summary['config'], adapter_config)
    summary['training_seconds'] = time.perf_counter()-train_started
    summary['training_steps'] = step_records
    total_change = (previous_sample-initial_sample).abs()
    summary['parameter_update'] = {
        'sampling': 'Up to64 evenly spaced values from every parameter',
        'sampled_values': len(total_change), 'changed_values': int((total_change > 0).sum()),
        'max_abs': float(total_change.max()), 'mean_abs': float(total_change.mean())}
    optimizer.zero_grad(set_to_none=True)
    checkpoint_dir = save_training_checkpoint(args.output, model, optimizer, args.steps, signature,
                                             summary['config'], adapter_config)
    summary['checkpoint'] = {'directory': str(checkpoint_dir), 'step': args.steps,
                             'model_sha256': sha256(checkpoint_dir / 'model.pt'),
                             'optimizer_saved': True, 'resume_supported': False,
                             'adapter_config_saved': True}
    summary['variants'] = {'causal_after': evaluate(adapter, pairs, args.output, 'causal_after', args.temporal_weight, device)}
    summary['completed_updates'] = args.steps
    summary['status'] = 'checking_saved_model'
    atomic_json(args.output / 'summary.json', summary)
    adapter.remove()
    model.to('cpu')
    node.MODEL_CACHE.clear()
    del optimizer, adapter, model, node, parameters
    empty_cache(device)

    node, model, adapter = reload_causal_model(args, checkpoint_dir / 'model.pt', device, adapter_config)
    reload_checks = {}
    with torch.no_grad():
        for pair in pairs:
            actual = adapter.native_forward(pair['draft'][None].to(device), pair['reference'].shape[-3:])[0].cpu()
            expected = tensor_load(args.output / summary['variants']['causal_after'][pair['sample_id']]['latent_path'])['latent']
            reload_checks[pair['sample_id']] = compare(actual, expected, args.tolerance)
            if not reload_checks[pair['sample_id']]['passed']:
                atomic_torch_save(args.output / 'reload_failures' / f'{pair["sample_id"]}.pt',
                                  {'expected': expected, 'actual': actual, 'check': reload_checks[pair['sample_id']]})
    summary['checkpoint_reload'] = {'samples': reload_checks, 'passed': all(v['passed'] for v in reload_checks.values()),
                                     'aligned_initialization_reapplied': False}
    summary['semantic_checks'] = semantic_checks(adapter, pairs, args, device)
    atomic_json(args.output / 'summary.json', summary)
    adapter.remove()
    model.to('cpu')
    node.MODEL_CACHE.clear()
    del adapter, model, node
    empty_cache(device)

    # Only now inspect the held-out output of the initial causal weights. Neither
    # those outputs nor any final diagnostics can affect the fixed training loop.
    node, model, adapter = reload_causal_model(args, initial_path, device, adapter_config)
    summary['variants']['causal_before'] = evaluate(adapter, pairs, args.output, 'causal_before', args.temporal_weight, device)
    for pair in pairs:
        save_latent(args.output, pair, 'reference', pair['reference'])
        save_latent(args.output, pair, 'draft', pair['draft'])
    summary['memory'] = memory_stats(device)
    summary['total_seconds'] = time.perf_counter()-started
    passed = summary['checkpoint_reload']['passed'] and summary['semantic_checks']['passed']
    summary['passed'] = passed
    summary['status'] = 'completed' if passed else 'completed_with_semantic_failure'
    atomic_json(args.output / 'summary.json', summary)
    print(json.dumps({'status': summary['status'], 'summary': str(args.output / 'summary.json'),
                      'completed_updates': args.steps, 'passed': passed}), flush=True)
    if not passed:
        raise AssertionError('Reload or causal/cache consistency failed; checkpoints and compared outputs retained')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(os.environ.get('EXP_ROOT', Path(__file__).resolve().parents[1]))
    p.add_argument('--train', type=Path, default=root / 'upscaler_trial/pairs/train.jsonl')
    p.add_argument('--holdout', type=Path, default=root / 'upscaler_trial/pairs/holdout.jsonl')
    p.add_argument('--upstream', type=Path, default=Path(os.environ.get('H3_UPSTREAM_DIR', root / 'upstream')))
    p.add_argument('--upscaler', type=Path, default=root / 'assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors')
    p.add_argument('--initialization', choices=('aligned_past', 'shifted'), default='aligned_past')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='npu:0', help='cpu, npu:0, or another supported torch device')
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-6)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--temporal-weight', type=float, default=.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu-threads', type=int, default=8)
    p.add_argument('--activation-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--checkpoint-every', type=int, default=0)
    p.add_argument('--tolerance', type=float, default=1e-4)
    args = p.parse_args()
    if (args.steps < 1 or args.cpu_threads < 1 or args.checkpoint_every < 0
            or not all(math.isfinite(v) for v in (args.lr, args.weight_decay, args.temporal_weight, args.tolerance))
            or args.lr <= 0 or min(args.weight_decay, args.temporal_weight, args.tolerance) < 0):
        p.error('Invalid training configuration')
    if args.output.exists():
        p.error('Use a new output directory; this driver never overwrites a prior trial')
    args.output.mkdir(parents=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    summary = {
        'status': 'initializing', 'config': config, 'compute_dtype': 'float32',
        'objective': 'Our supervised normalized-latent MSE + temporal_weight * temporal-difference MSE; not the author recipe',
        'training_scope': 'All upscaler parameters, eval mode disables dropout; offline native_forward with causal operators',
        'holdout_policy': 'Fixed schedule; train only. Holdout metrics/semantics computed after all updates, never select checkpoints.',
        'resume_supported': False, 'no_refiner': True, 'no_dmd': True, 'no_vae_gradients': True,
        'limitations': ['Tiny paired experiment, no generalization claim.',
                       'Causality is relative to input latents; VAE and end-to-end pixel pipeline are not made causal.',
                       'Optimizer state is saved for future tooling; this driver does not implement resume.',
                       'Causal initialization changes the released model function; pretraining equivalence is not claimed.']}
    atomic_json(args.output / 'summary.json', summary)
    try:
        run(args, summary)
    except Exception as error:
        if summary.get('status') != 'completed_with_semantic_failure':
            summary['status'] = 'failed'
        summary['error'] = {'type': type(error).__name__, 'message': str(error)}
        atomic_json(args.output / 'summary.json', summary)
        raise


if __name__ == '__main__':
    main()
