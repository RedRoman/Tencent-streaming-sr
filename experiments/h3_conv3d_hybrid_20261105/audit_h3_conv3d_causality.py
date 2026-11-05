"""Long-context conditional-causality audit for the H3 Conv3d hybrid student.

Without --initial this uses the real upstream classes at full depth and reduced
width on CPU. With --initial it loads actual 345M initial or hybrid weights.
This audits fixed latent inputs, not the H3 encoder/decoder or video quality.
Existing evidence is never overwritten. Failures retain a JSON error record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import torch
from torch import nn

import causal_upscaler
import h3_conv3d_hybrid
import prepare_data
import train_upscaler_trial
from causal_upscaler import initialize_aligned_past_
from data_utils import select_device, synchronize
from h3_conv3d_hybrid import (CausalConv3dStudent, build_student,
                              load_student_checkpoint)
from prepare_data import load_author_upscaler


FRAMES = 163
CUTS = (32, 149, 160)
GRADIENT_OUTPUT_INDEX = 149
ATOL = RTOL = 3e-5
MAX_ABS = 1e-4


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(part)
    return dict(path=str(path.resolve()), sha256=digest.hexdigest(), bytes=path.stat().st_size)


def comparison(reference, actual):
    if reference.shape != actual.shape:
        return dict(passed=False, reference_shape=list(reference.shape), actual_shape=list(actual.shape))
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(actual).all())
    maximum = float((reference - actual).abs().max()) if finite else None
    close = bool(torch.allclose(reference, actual, atol=ATOL, rtol=RTOL)) if finite else False
    return dict(passed=finite and close and maximum <= MAX_ABS, finite=finite,
                allclose=close, max_abs=maximum, bit_exact=bool(torch.equal(reference, actual)),
                atol=ATOL, rtol=RTOL, max_abs_limit=MAX_ABS)


def synthetic_student(upstream, device, activation_checkpointing=False):
    node = load_author_upscaler(upstream, Path('unused-no-checkpoint-loaded'), device)
    model = node.LatentResizer3D(in_channels=24, channels=32, in_blocks=12,
        out_blocks=12, dropout=.1, temporal_every=2, temporal_kernel=5, attn=False).eval()
    # Constructors zero residual tails. Activate them so all temporal blocks
    # contribute, then apply the same one-time alignment as the original run.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'out_layers.2.weight' in name or 'pwconv.weight' in name:
                parameter.normal_(0, .02)
    initialize_aligned_past_(model)
    config = dict(version=1, initialization='aligned_past', temporal_padding='left_only_zero',
        groupnorm='per_frame', normalization='D(U(N(native_normalized_H3_latent)))',
        spatial_scale=2.0, temporal_chunking=False, dtype='float32',
        saved_weights_already_initialized=True, reload_must_not_reinitialize_weights=True)
    return CausalConv3dStudent(node, model.to(device), config, activation_checkpointing)


def architecture(student):
    model = student.model
    modules = dict(model.named_modules())
    return dict(model_class=type(model).__name__, model_config=student.model_config,
        input_channels=model.conv_in.in_channels, channels=model.conv_in.out_channels,
        in_block_types=[type(block).__name__ for block in model.in_blocks],
        out_block_types=[type(block).__name__ for block in model.out_blocks],
        parameter_count=sum(value.numel() for value in model.parameters()),
        parameter_dtypes=sorted({str(value.dtype) for value in model.parameters()}),
        original_model_training=model.training, wrapper_training=student.training,
        activation_checkpointing=student.activation_checkpointing,
        temporal_structure=student.adapter.structure,
        convolutions={name: dict(kernel=list(module.kernel_size), stride=list(module.stride),
            original_padding=list(module.padding), dilation=list(module.dilation), groups=module.groups)
            for name, module in modules.items() if isinstance(module, nn.Conv3d)},
        groupnorm={name: dict(groups=module.num_groups, channels=module.num_channels, eps=module.eps,
            effective_time_scope='per_frame') for name, module in modules.items()
            if isinstance(module, nn.GroupNorm)},
        dropout={name: dict(p=module.p, training=module.training) for name, module in modules.items()
            if isinstance(module, nn.Dropout)},
        attention_modules=[name for name, module in modules.items()
            if 'attn' in type(module).__name__.lower() or isinstance(module, nn.MultiheadAttention)])


def execute(args, report):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    if args.initial is None and args.device != 'cpu':
        raise ValueError('Reduced-width semantic mode is CPU-only; provide --initial for actual accelerator audit')
    device = select_device(args.device)
    report['sources'] = {name: fingerprint(path) for name, path in (
        ('audit', __file__), ('wrapper', h3_conv3d_hybrid.__file__),
        ('adapter', causal_upscaler.__file__), ('loader_helper', prepare_data.__file__),
        ('activation_checkpointing_helper', train_upscaler_trial.__file__),
        ('upstream_loader', args.upstream / 'open_h3/upscaler.py'),
        ('upstream_architecture', args.upstream / 'open_h3/upscaler_node.py'))}
    report['mode'] = 'actual_345m_checkpoint' if args.initial is not None else 'reduced_width_cpu_semantics'
    if args.initial is not None:
        if args.weights is None:
            raise ValueError('--weights is required with --initial')
        report['checkpoint'] = fingerprint(args.initial)
        report['released_weights'] = fingerprint(args.weights)
        loader = build_student if args.checkpoint_kind == 'initial' else load_student_checkpoint
        student = loader(args.initial, args.upstream, args.weights, device,
                         activation_checkpointing=False)
    else:
        student = synthetic_student(args.upstream, device)
        report['synthetic_initialization'] = ('Random full-depth width32 upstream model, residual tails '
            'activated, aligned_past applied once; no pretrained checkpoint loaded')
    student.eval().requires_grad_(False)
    report['architecture'] = architecture(student)
    arch = report['architecture']
    if arch['temporal_structure']['longest_convolution_receptive_field'] != 149:
        raise ValueError('Expected 149-latent-frame structural receptive field')
    if arch['attention_modules'] or arch['original_model_training']:
        raise ValueError('Expected attention-free deterministic eval execution')
    if any(value != 'torch.float32' for value in arch['parameter_dtypes']):
        raise ValueError('This audit is specified for FP32')
    if args.initial is not None and arch['parameter_count'] != 345280216:
        raise ValueError('Actual checkpoint audit requires the complete 345280216-parameter model')
    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    native = torch.randn(1, 24, FRAMES, 2, 2, generator=generator).to(device)
    checks = report['checks'] = {}
    with torch.no_grad():
        full = student(native)
        if full.shape != (1, 24, FRAMES, 4, 4):
            raise ValueError(f'Unexpected output shape: {tuple(full.shape)}')
        for cut in CUTS:
            altered = native.clone()
            altered[:, :, cut:] = (torch.randn(altered[:, :, cut:].shape, generator=generator) * 3 + 2).to(device)
            checks[f'future_cut_{cut}'] = comparison(full[:, :, :cut], student(altered)[:, :, :cut])
            checks[f'prefix_cut_{cut}'] = comparison(full[:, :, :cut], student(native[:, :, :cut].contiguous()))
        for label, parts in (('single_frames', [1] * FRAMES), ('chunks32', [32] * 5 + [3]),
                             ('uneven', [3, 146, 14])):
            student.reset_stream()
            offset, outputs, sizes = 0, [], []
            for length in parts:
                outputs.append(student.stream_step(native[:, :, offset:offset + length].contiguous()))
                offset += length
                sizes.append(student.stream_state()['cache_bytes'])
            if offset != FRAMES:
                raise AssertionError('Chunk partition did not cover the whole input')
            result = comparison(full, torch.cat(outputs, dim=2))
            state = student.stream_state()
            result.update(partition=parts, cache=state,
                          cache_bytes_constant=len(set(sizes)) == 1,
                          processed_frames_correct=state['processed_latent_frames'] == FRAMES)
            result['passed'] = (result['passed'] and result['cache_bytes_constant']
                and result['processed_frames_correct'] and state['cached_convolutions'] == 62)
            checks[f'stream_{label}'] = result
        student.reset_stream()
        checks['reset_replay'] = comparison(full, student.stream_step(native))
        # Exercise the new wrapper's state-token path independently of the old
        # adapter interface, including generation/offset accounting.
        outputs, offset, state = [], 0, None
        for length in [3, 146, 14]:
            output, state = student.forward_chunk(native[:, :, offset:offset + length].contiguous(),
                                                  state=state, latent_index=offset)
            outputs.append(output)
            offset += length
        checks['wrapper_forward_chunk'] = comparison(full, torch.cat(outputs, dim=2))
        checks['wrapper_forward_chunk']['next_latent_index'] = state.next_latent_index
        if state.next_latent_index != FRAMES:
            checks['wrapper_forward_chunk']['passed'] = False
    student.reset_stream()
    differentiable = native.detach().clone().requires_grad_(True)
    output = student(differentiable)
    output[:, :, GRADIENT_OUTPUT_INDEX].square().mean().backward()
    gradient = differentiable.grad
    if gradient is None:
        raise AssertionError('Missing input gradient')
    future = float(gradient[:, :, 150:].abs().max())
    before_history = float(gradient[:, :, :1].abs().max())
    current = float(gradient[:, :, 149].abs().max())
    finite = bool(torch.isfinite(gradient).all())
    checks['input_gradient'] = dict(output_index=149, structural_history_range=[1, 149],
        future_150plus_max_abs=future, excluded_index0_max_abs=before_history,
        current149_max_abs=current, finite=finite,
        passed=finite and future == 0 and before_history == 0 and current > 0,
        parameter_gradients_computed=False)
    if args.initial is None:
        # CPU semantic coverage for the new training wrapper. Compare both the
        # input gradient and the real output-head gradient under checkpointing.
        checkpointed = synthetic_student(args.upstream, device, activation_checkpointing=True)
        checkpointed.load_original_state_dict(student.original_state_dict())
        checkpointed.eval().requires_grad_(False)
        student.model.conv_out.weight.requires_grad_(True)
        checkpointed.model.conv_out.weight.requires_grad_(True)
        left = native[:, :, :9].detach().clone().requires_grad_(True)
        right = left.detach().clone().requires_grad_(True)
        left_output, right_output = student(left), checkpointed(right)
        left_output.square().mean().backward()
        right_output.square().mean().backward()
        forward_match = comparison(left_output, right_output)
        input_gradient_match = comparison(left.grad, right.grad)
        head_gradient_match = comparison(student.model.conv_out.weight.grad,
                                         checkpointed.model.conv_out.weight.grad)
        head_nonzero = float(student.model.conv_out.weight.grad.abs().max()) > 0
        checks['checkpointing_training_path_cpu'] = dict(forward=forward_match,
            input_gradient=input_gradient_match, head_gradient=head_gradient_match,
            head_gradient_nonzero=head_nonzero, frames=9,
            passed=all(value['passed'] for value in (forward_match, input_gradient_match,
                head_gradient_match)) and head_nonzero)
    synchronize(device)
    report['check_count'] = len(checks)
    report['all_passed'] = all(value['passed'] for value in checks.values())
    if not report['all_passed']:
        raise AssertionError('Conditional-causality check failed; see retained checks')
    # Verify files did not change during this audit. Record both fingerprints
    # when a change is detected rather than silently changing the contract.
    report['sources_after'] = {name: fingerprint(value['path']) for name, value in report['sources'].items()}
    report['sources_unchanged'] = report['sources_after'] == report['sources']
    if not report['sources_unchanged']:
        raise RuntimeError('An audited source changed during execution')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--initial', type=Path, help='Actual old eWarp or new hybrid checkpoint; omit for CPU reduced-width mode')
    parser.add_argument('--checkpoint-kind', choices=('initial', 'hybrid'), default='initial')
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--weights', type=Path, help='Original released upscaler weights required by the author loader')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261105)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also rejects symlinks/existing failed evidence. Only
    # this invocation's newly-created record is subsequently rewritten.
    with args.output.open('x') as handle:
        started = time.perf_counter()
        report = dict(status='running', config={name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()}, seed=args.seed, frames=FRAMES,
            input_spatial=[2, 2], output_spatial=[4, 4], dtype='float32',
            scope='Student upscaler conditional on fixed native H3 latent inputs only',
            limitations=['Encoder, decoder, and source-pixel end-to-end causality are outside this audit.',
                'Synthetic small spatial inputs test dependencies and caches, not long-video quality.',
                'CPU reduced-width mode does not verify actual345M weights or NPU kernels.'])
        json.dump(report, handle, indent=2); handle.flush()
        try:
            execute(args, report)
            report['status'] = 'passed'
        except BaseException as error:
            report.update(status='failed', error_type=type(error).__name__, error=str(error),
                          traceback=traceback.format_exc())
            raise
        finally:
            report['seconds'] = time.perf_counter() - started
            handle.seek(0); handle.truncate()
            json.dump(report, handle, indent=2); handle.write('\n'); handle.flush()
            print(json.dumps({key: report.get(key) for key in
                ('status', 'mode', 'check_count', 'all_passed', 'sources_unchanged', 'seconds', 'error')}))


if __name__ == '__main__':
    main()
