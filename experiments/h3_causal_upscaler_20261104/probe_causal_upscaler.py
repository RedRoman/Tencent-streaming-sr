"""Isolated full-checkpoint causal upscaler diagnostics; no H3 refiner or training.

Test causality with fixed latent inputs, independently of the noncausal VAE.
Evaluate direct conversion before any fine-tuning, retaining failed checks.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from causal_upscaler import CausalUpscalerAdapter, initialize_aligned_past_
from data_utils import select_device, synchronize
from prepare_data import author_upscale, load_author_upscaler
from train_upscaler_trial import (atomic_json, memory_stats, metrics,
    parameter_sample, read_pairs, save_latent, sha256, wrapper_forward)


def difference(a, b, tolerance):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    if a.shape != b.shape:
        raise ValueError(f'Shape mismatch: {a.shape} != {b.shape}')
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise FloatingPointError('Nonfinite output')
    delta = a.double() - b.double()
    error = float(delta.abs().max())
    return {'max_abs': error, 'rms': float(delta.square().mean().sqrt()),
            'bit_exact': bool(torch.equal(a, b)), 'passed': error <= tolerance}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='npu:0')
    p.add_argument('--tolerance', type=float, default=1e-4)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory; failed checks are retained')
    if not args.tolerance > 0:
        raise ValueError('Positive tolerance required')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(20261104)
    device = select_device(args.device)
    started = time.perf_counter()
    pairs_dir = args.root / 'upscaler_trial/pairs'
    pairs = read_pairs(pairs_dir / 'train.jsonl', 'train') + read_pairs(pairs_dir / 'holdout.jsonl', 'holdout')
    if len({v['sample_id'] for v in pairs}) != len(pairs):
        raise ValueError('Overlapping sample IDs')
    heldout = next(pair for pair in pairs if pair['split'] == 'holdout')
    if heldout['draft'].shape != (24, 13, 12, 21):
        raise ValueError('This probe uses the prepared 13-frame 2x pairs')
    checkpoint = args.root / 'assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors'
    node = load_author_upscaler(args.upstream, checkpoint, device)
    model = node.load_model('upscaler', device, 'fp32').eval().requires_grad_(False)
    original_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    report = {'status': 'running', 'no_refiner': True, 'optimization_updates': 0,
              'device': str(device), 'precision': 'fp32', 'tolerance': args.tolerance,
              'checkpoint_sha256': sha256(checkpoint),
              'parameter_count': sum(v.numel() for v in model.parameters()),
              'diagnostic_sample': heldout['sample_id'],
              'pairs': [{'sample_id': q['sample_id'], 'split': q['split'],
                         'pair_sha256': q['sha256']} for q in pairs],
              'variants': {}, 'causality': {},
              'limitations': ['Fixed VAE-produced latent inputs; not pixel-level end-to-end causality.',
                  'Three clips for quality, one held-out clip for numerical diagnostics.',
                  'VAE is noncausal and used only to visualize/measure the same latent outputs.',
                  'No optimization, refiner, text, audio, or DMD is used.',
                  'FP32 parameter dtype does not specify every accelerator kernel internal precision.',
                  'Causality checks and decoded quality are separate outcomes.']}

    def checkpoint_report():
        atomic_json(args.output / 'summary.json', report)

    def evaluate(label, forward):
        values = {}
        with torch.no_grad():
            for pair in pairs:
                before = time.perf_counter()
                prediction = forward(pair['draft'][None].to(device), tuple(pair['reference'].shape[-3:]))[0].cpu()
                synchronize(device)
                entry = metrics(prediction, pair['reference'], .1)
                entry.update(seconds=time.perf_counter()-before, split=pair['split'],
                             latent_path=save_latent(args.output, pair, label, prediction))
                values[pair['sample_id']] = entry
                print(json.dumps({'evaluation': label, 'sample_id': pair['sample_id'], **entry}), flush=True)
        report['variants'][label] = values
        checkpoint_report()

    native = heldout['draft'][None].to(device)
    target = tuple(heldout['reference'].shape[-3:])
    with torch.no_grad():
        released = author_upscale(node, heldout['draft'], height=target[-2], width=target[-1], precision='fp32')
        full_original = wrapper_forward(node, model, native, target)
        report['released_node_parity'] = difference(released, full_original[0], 1e-6)
        if not report['released_node_parity']['passed']:
            checkpoint_report()
            raise AssertionError('Original-node wrapper parity failed')
        changed = native.clone()
        changed[:, :, 5:] = 3 * torch.randn(changed[:, :, 5:].shape,
            generator=torch.Generator().manual_seed(20261105)).to(device)
        altered = wrapper_forward(node, model, changed, target)
        control = difference(full_original[:, :, :5], altered[:, :, :5], args.tolerance)
        control['passed'] = control['max_abs'] > args.tolerance
        report['original_noncausal_future_response'] = control
        del released, full_original, altered, changed
    evaluate('pretrained', lambda x, size: wrapper_forward(node, model, x, size))
    report['variants']['trilinear'] = {}
    for pair in pairs:
        prediction = F.interpolate(pair['draft'][None], size=pair['reference'].shape[-3:],
                                   mode='trilinear', align_corners=False)[0]
        entry = metrics(prediction, pair['reference'], .1)
        entry.update(split=pair['split'], latent_path=save_latent(args.output, pair, 'trilinear', prediction))
        report['variants']['trilinear'][pair['sample_id']] = entry
        save_latent(args.output, pair, 'reference', pair['reference'])

    for label in ('causal_shifted', 'causal_aligned'):
        model.load_state_dict(original_state, strict=True)
        initialization = {'kind': 'unchanged_coefficients_with_left_padding'}
        if label == 'causal_aligned':
            initialization = initialize_aligned_past_(model)
        sampled_weights = parameter_sample(model)
        adapter = CausalUpscalerAdapter(node, model)
        result = {'initialization': initialization, 'structure': adapter.structure,
                  'future_checks': [], 'prefix_checks': [], 'stream_checks': []}
        with torch.no_grad():
            full = adapter.native_forward(native, target)
            for cut in (1, 5, 10):
                altered_input = native.clone()
                altered_input[:, :, cut:] = 3 * torch.randn(altered_input[:, :, cut:].shape,
                    generator=torch.Generator().manual_seed(20261104 + cut)).to(device)
                changed_output = adapter.native_forward(altered_input, target)
                check = difference(full[:, :, :cut], changed_output[:, :, :cut], args.tolerance)
                check.update(cut_latent_frame=cut, perturbation_seed=20261104+cut)
                result['future_checks'].append(check)
                prefix = adapter.native_forward(native[:, :, :cut].contiguous(), (cut, *target[-2:]))
                result['prefix_checks'].append({'prefix_frames': cut,
                    **difference(full[:, :, :cut], prefix, args.tolerance)})
                del altered_input, changed_output, prefix
        for partition in ([5, 5, 3], [1] * 13, [2, 3, 1, 7]):
            adapter.reset_stream()
            position, outputs = 0, []
            before = time.perf_counter()
            for frames in partition:
                outputs.append(adapter.stream_step(native[:, :, position:position+frames].contiguous(), target[-2:]))
                position += frames
            synchronize(device)
            result['stream_checks'].append({'partition': partition, 'seconds': time.perf_counter()-before,
                **difference(full, torch.cat(outputs, dim=2), args.tolerance), 'state': adapter.stream_state()})
            del outputs
        adapter.reset_stream()
        replay = adapter.stream_step(native, target[-2:])
        result['reset_replay'] = difference(full, replay, args.tolerance)
        adapter.reset_stream()
        del replay, full
        # Freeze every parameter; backpropagate only to the actual full-model input.
        # This directly checks a past output has zero derivative w.r.t. future inputs.
        differentiable = native.detach().clone().requires_grad_(True)
        prediction = adapter.native_forward(differentiable, target)
        prediction[:, :, 4].square().mean().backward()
        future_gradient = differentiable.grad[:, :, 5:]
        current_gradient = differentiable.grad[:, :, 4]
        result['input_gradient'] = {'output_latent_frame': 4,
            'future_max_abs': float(future_gradient.abs().max()),
            'current_max_abs': float(current_gradient.abs().max()),
            'finite': bool(torch.isfinite(differentiable.grad).all())}
        result['input_gradient']['passed'] = (result['input_gradient']['finite']
            and result['input_gradient']['future_max_abs'] == 0
            and result['input_gradient']['current_max_abs'] > 0)
        del prediction, differentiable, future_gradient, current_gradient
        result['sampled_weight_invariance'] = difference(sampled_weights, parameter_sample(model), 0.)
        evaluate(label, adapter.native_forward)
        result['passed'] = all(v['passed'] for group in ('future_checks','prefix_checks','stream_checks')
                               for v in result[group]) and all(result[k]['passed'] for k in
                               ('reset_replay','input_gradient','sampled_weight_invariance'))
        report['causality'][label] = result
        print(json.dumps({'causal_variant': label, 'passed': result['passed'],
            'future': result['future_checks'], 'prefix': result['prefix_checks'],
            'stream_max_abs': [v['max_abs'] for v in result['stream_checks']],
            'input_gradient': result['input_gradient']}), flush=True)
        checkpoint_report()
        adapter.remove()
    report['checks_passed'] = (report['released_node_parity']['passed']
        and report['original_noncausal_future_response']['passed']
        and all(v['passed'] for v in report['causality'].values()))
    report['memory'] = memory_stats(device)
    report['seconds'] = time.perf_counter()-started
    report['status'] = 'completed'
    checkpoint_report()
    print(json.dumps({'completed': True, 'checks_passed': report['checks_passed'],
                      'seconds': report['seconds'], 'memory': report['memory']}), flush=True)
    if not report['checks_passed']:
        raise AssertionError('One or more causal upscaler checks failed; evidence saved')


if __name__ == '__main__':
    main()
