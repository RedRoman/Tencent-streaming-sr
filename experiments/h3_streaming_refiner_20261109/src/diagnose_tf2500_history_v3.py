"""Read-only matched-history diagnostic; no training and no background waiter.

Admission uses only stdlib. The existing evaluation must finish or have a verified user-directed
deferral with complete2500 artifacts, and be reaped before accelerator imports. Truncated offline calls are diagnostic
training-context references, never described as incremental inference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from decoder_validation_admission import predecessor_state, verify_parent_locks


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def pinned(path, digest):
    if sha(path) != digest:
        raise ValueError('Pinned file changed: ' + Path(path).name)
    return json.loads(Path(path).read_text())


def check_payload(row):
    path = Path(row['path'])
    if 'bytes' in row and path.stat().st_size != row['bytes']:
        raise ValueError('Payload size mismatch: ' + path.name)
    if sha(path) != row['sha256']:
        raise ValueError('Payload digest mismatch: ' + path.name)


def require_ready(report):
    if not report['ready_for_NPU']:
        raise RuntimeError('Evaluation completion or verified deferral required and predecessor must be reaped; no accelerator import')


def admit(args):
    stage = Path(args.stage)
    sources = pinned(stage / 'manifest.json', args.manifest_sha)
    for name, digest in sources['files'].items():
        if sha(stage / name) != digest:
            raise ValueError('Diagnostic source changed: ' + name)
    plan = json.loads((stage / 'config/history_diagnostic_33b_v3.json').read_text())
    for name, digest in plan['runtime_source_sha256'].items():
        if sha(Path(args.runtime_stage) / 'src' / name) != digest:
            raise ValueError('Frozen runtime changed: ' + name)
    train = pinned(args.train_manifest, plan['train_manifest_sha256'])
    dev = pinned(args.dev_manifest, plan['dev_manifest_sha256'])
    if (train['upsampler_sha256'] != plan['upsampler_sha256']
            or dev['accepted_upsampler_sha256'] != plan['upsampler_sha256']
            or dev['test_unlocked']):
        raise ValueError('Frozen baseline or split changed')
    if [r['sample_id'] for r in train['samples']] != plan['train']:
        raise ValueError('Train cohort changed')
    if [r['sample_id'] for r in dev['samples']] != plan['dev']:
        raise ValueError('Dev cohort changed')
    for row in train['samples'] + dev['samples']:
        check_payload(row)
    check_payload(dev['condition'])
    if train['condition_sha256'] != dev['condition']['sha256']:
        raise ValueError('Conditioning differs between train/dev')
    if sha(args.checkpoint) != plan['checkpoint_sha256']:
        raise ValueError('Use the exact saved step2500 adapter')
    stop = pinned(args.stop_receipt, plan['stop_receipt_sha256'])
    if (stop['status'] != 'user_requested_checkpoint_stop_complete'
            or stop['actual_logged_optimizer_updates'] != 2500
            or not stop['child_reaped'] or not stop['parent_and_child_absent']):
        raise ValueError('Training stop not fully verified')
    assets = pinned(args.assets_manifest, plan['assets_manifest_sha256'])
    if assets['status'] != 'complete':
        raise ValueError('Asset audit incomplete')
    for name, desc in assets['files'].items():
        st = (Path(args.root) / name).stat()
        if (st.st_size, st.st_mtime_ns, st.st_ino) != (desc['bytes'], desc['mtime_ns'], desc['inode']):
            raise ValueError('Audited frozen asset identity changed: ' + name)
    launch = json.loads(Path(args.predecessor_launch).read_text())
    command = launch['command']
    if (sha(command[1]) != plan['evaluation_source_sha256']
            or command[command.index('--manifest-sha') + 1] != plan['evaluation_manifest_sha256']
            or Path(command[command.index('--output') + 1]).resolve() != Path(args.evaluation).resolve()):
        raise ValueError('Unexpected predecessor evaluation')
    predecessor = predecessor_state(args.predecessor_launch)
    result_path = Path(args.evaluation) / 'result.json'
    cohort = json.loads(result_path.read_text()) if result_path.exists() else None
    complete = (cohort is not None
                and cohort.get('status') == 'full226_cohort_complete_visual_and_decoder_acceptance_pending'
                and len(cohort.get('pixel_results', [])) == 12
                and cohort.get('optimizer_updates_this_evaluation') == 0
                and launch.get('status') == 'complete' and launch.get('exit_code') == 0)
    deferred = None
    if args.evaluation_defer_receipt and Path(args.evaluation_defer_receipt).exists():
        if not args.evaluation_defer_sha:
            raise ValueError('Explicit completed deferral receipt SHA required')
        deferred = pinned(args.evaluation_defer_receipt, args.evaluation_defer_sha)
        if (deferred['status'] != 'evaluation_deferred_for_user_requested_diagnosis'
                or deferred['request_sha256'] != plan['evaluation_deferral_request_sha256']
                or deferred['controller_source_sha256'] != plan['evaluation_deferral_controller_sha256']
                or deferred['evaluation_launch_sha256'] != sha(args.predecessor_launch)
                or not deferred['child_reaped'] or not deferred['all_owned_processes_absent']
                or deferred['actual_training_updates'] != 2500 or deferred['new_optimizer_updates'] != 0
                or launch.get('status') != 'failed' or launch.get('exit_code') != -15
                or [r['sample_id'] for r in deferred['completed_step2500']] != plan['dev']):
            raise ValueError('User-directed evaluation deferral not fully verified')
    reuses = {}
    if complete or deferred:
        for sid in plan['dev']:
            path = Path(args.evaluation) / ('step2500_' + sid + '_latent/result.json')
            row = json.loads(path.read_text())
            if (row['sample_id'] != sid or row['training_step'] != 2500
                    or row['checkpoint_sha256'] != plan['checkpoint_sha256']
                    or row['pipeline_shape'] != [1, 24, 67, 68, 120]
                    or row['protocol_sha256'] != plan['evaluation_protocol_sha256']
                    or row['chunk_invariance_first4']['max_abs'] != 0):
                raise ValueError('Dev history lineage mismatch')
            if deferred:
                binding = next(r for r in deferred['completed_step2500'] if r['sample_id'] == sid)
                pixel = Path(args.evaluation) / ('step2500_' + sid + '_pixels/result.json')
                if sha(path) != binding['latent_receipt_sha256'] or sha(pixel) != binding['pixel_receipt_sha256']:
                    raise ValueError('Completed2500 receipt changed since deferral')
            check_payload(row['output'])
            reuses[sid] = {'receipt_sha256': sha(path), 'output': row['output']}
    report = {'status': 'CPU_admission_complete', 'ready_for_NPU': bool(complete or deferred) and predecessor['ready'],
              'predecessor': predecessor, 'fixed_evaluation_complete': complete, 'intentional_evaluation_deferral': deferred,
              'fixed_evaluation_receipt_sha256': sha(result_path) if complete else None,
              'manifest_sha256': args.manifest_sha, 'plan_sha256': sha(stage / 'config/history_diagnostic_33b_v3.json'),
              'checkpoint_sha256': plan['checkpoint_sha256'], 'reused_dev_histories': reuses,
              'train_samples': plan['train'], 'dev_samples': plan['dev'],
              'torch_imported': 'torch' in sys.modules, 'torch_npu_imported': 'torch_npu' in sys.modules,
              'new_optimizer_updates': 0, 'new_accelerator_forwards': 0}
    return report, plan, train, dev


def compare(a, b):
    delta = a.float() - b.float()
    return {'max_abs': float(delta.abs().max()), 'mse': float(delta.square().mean()),
            'l1': float(delta.abs().mean()),
            'relative_rms': float(delta.square().mean().sqrt() / b.float().square().mean().sqrt().clamp_min(1e-8))}


def cache_identity(rt):
    return (rt.next_frame, tuple((id(v.k), id(v.v), v.k._version, v.v._version, v.frames)
                                for v in (rt.cache or [])))


def diagnose_sample(rt, sample, condition, self_history, plan, noise_fn, log):
    """Hold each noisy current latent fixed across three history contexts.

    Point predictions use HR/noise mixtures and FM targets. Independent Euler
    probes start from the frozen draft/noise mixture and never commit their
    outputs; only the designated clean history advances the diagnostic cache.
    """
    import torch
    positions = plan['positions']
    sigmas = plan['sigmas']
    hr, draft = sample['reference'], sample['upscaled']
    probes, rows, cache_trace = {}, [], []
    forwards = 0
    parity = []

    def query(mode, pos, x, sigma):
        nonlocal forwards
        before = cache_identity(rt)
        if mode == 'gt_truncated_offline':
            frames = sorted(set([0, max(0, pos - 1), pos]))
            context = hr[:, frames].clone().to(rt.device)
            context[:, -1:] = x
            value = rt.offline(context, draft[:, frames], condition, frames, sigma)
        else:
            if rt.next_frame != pos:
                raise ValueError('Wrong absolute current position')
            value = rt.stream_velocity(x, draft[:, pos:pos+1], sigma)
        forwards += 1
        if cache_identity(rt) != before:
            raise ValueError('A diagnostic query changed clean history')
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite diagnostic velocity')
        return value

    with torch.no_grad():
        for mode in ('gt_incremental', 'self_incremental', 'gt_truncated_offline'):
            if mode != 'gt_truncated_offline':
                rt.start_stream(condition)
                forwards += 1
            for pos in range(max(positions) + 1):
                if pos in positions:
                    clean = hr[:, pos:pos+1].float().to(rt.device)
                    cond = draft[:, pos:pos+1].float().to(rt.device)
                    noise = noise_fn(clean.shape, sample['sample_id'], pos, plan['seed']).to(rt.device)
                    for index, sigma in enumerate(sigmas[:-1]):
                        x = (1 - sigma) * clean + sigma * noise
                        velocity = query(mode, pos, x, sigma)
                        x0 = x + sigma * velocity
                        probes[(mode, pos, index)] = velocity.detach().cpu()
                        rows.append({'mode': mode, 'position': pos, 'sigma': sigma,
                                     'fm_mse': float((velocity - (clean - noise)).square().mean()),
                                     'one_step_x0_vs_HR': compare(x0, clean),
                                     'one_step_x0_vs_draft': compare(x0, cond)})
                        if mode == 'gt_incremental' and pos == 3 and index == 0:
                            # Four-latent full-prefix oracle only; never used as streaming inference.
                            context = hr[:, :4].clone().to(rt.device)
                            context[:, -1:] = x
                            before = cache_identity(rt)
                            reference = rt.offline(context, draft[:, :4], condition, [0, 1, 2, 3], sigma)
                            forwards += 1
                            gap = compare(velocity, reference)
                            if cache_identity(rt) != before:
                                raise ValueError('Offline reference mutated stream cache')
                            if gap['max_abs'] > .03 or gap['relative_rms'] > .005:
                                raise ValueError('Four-latent same-history parity failed: ' + str(gap))
                            parity.append(gap)
                    x = (1 - sigmas[0]) * cond + sigmas[0] * noise
                    for sigma, following in zip(sigmas, sigmas[1:]):
                        x = x + (sigma - following) * query(mode, pos, x, sigma)
                    if not torch.isfinite(x).all():
                        raise ValueError('Nonfinite Euler probe')
                    probes[(mode, pos, 'rollout')] = x.detach().cpu()
                    row = {'mode': mode, 'position': pos, 'kind': 'three_step_fixed_history_probe',
                           'rollout_vs_HR': compare(x, clean), 'rollout_vs_draft': compare(x, cond),
                           'no_refiner_vs_HR': compare(cond, clean)}
                    if mode == 'self_incremental':
                        replay = compare(x, self_history[:, pos:pos+1].to(rt.device))
                        row['saved_self_history_replay'] = replay
                        if replay['max_abs'] > .03 or replay['relative_rms'] > .005:
                            raise ValueError('Saved self-history replay mismatch: ' + str(replay))
                    rows.append(row)
                    log('matched_position_complete', sample_id=sample['sample_id'], mode=mode, position=pos)
                if mode != 'gt_truncated_offline':
                    history = hr if mode == 'gt_incremental' else self_history
                    rt.stream_velocity(history[:, pos:pos+1], draft[:, pos:pos+1], 0, commit=True)
                    forwards += 1
                    frames = max(len(set(f for f in v.frames if f >= 0)) for v in rt.cache)
                    size = sum(v.k.numel()*v.k.element_size() + v.v.numel()*v.v.element_size() for v in rt.cache)
                    if frames > 2 or rt.next_frame != pos + 1:
                        raise ValueError('History cache bound or cursor failed')
                    cache_trace.append({'mode': mode, 'position': pos, 'bytes': size, 'video_frames': frames})
        gaps = []
        for pos in positions:
            for index in [0, 1, 2, 'rollout']:
                reference = probes[('gt_incremental', pos, index)]
                for mode in ('self_incremental', 'gt_truncated_offline'):
                    delta = compare(probes[(mode, pos, index)], reference)
                    if pos == 0 and (delta['max_abs'] > .03 or delta['relative_rms'] > .005):
                        raise ValueError('No-history control differs across arms: ' + str(delta))
                    gaps.append({'position': pos, 'probe': index, 'mode': mode,
                                 'reference': 'gt_incremental', 'difference': delta})
    return {'sample_id': sample['sample_id'], 'rows': rows, 'history_differences': gaps,
            'four_latent_full_prefix_oracle_parity': parity, 'cache_trace': cache_trace,
            'model_forwards_including_static_and_history_writes': forwards,
            'scope': 'fixed-current-latent history interventions and 12-latent prefix; not full226 quality acceptance',
            'new_optimizer_updates': 0, 'root_cause_proven': False}


def diagnose_initial_condition(rt, sample, condition, plan, noise_fn):
    """Current-HR conditioning is a labeled oracle; no future HR is supplied."""
    import torch
    sigma_list = plan['sigmas']
    clean = sample['reference'][:, :1].float().to(rt.device)
    draft = sample['upscaled'][:, :1].float().to(rt.device)
    noise = noise_fn(clean.shape, sample['sample_id'], 0, plan['seed']).to(rt.device)
    rows = []
    with torch.no_grad():
        rt.start_stream(condition)
        for label, injected in [('zero_draft_condition', torch.zeros_like(draft)),
                                ('current_HR_condition_oracle', clean)]:
            points = []
            before = cache_identity(rt)
            for sigma in sigma_list[:-1]:
                x = (1-sigma)*clean + sigma*noise
                v = rt.stream_velocity(x, injected, sigma)
                if not torch.isfinite(v).all():
                    raise ValueError('Nonfinite condition-ablation velocity')
                points.append({'sigma': sigma, 'fm_mse': float((v-(clean-noise)).square().mean()),
                               'one_step_x0_vs_HR': compare(x+sigma*v, clean)})
            x = (1-sigma_list[0])*draft + sigma_list[0]*noise
            for sigma, following in zip(sigma_list, sigma_list[1:]):
                x = x + (sigma-following)*rt.stream_velocity(x, injected, sigma)
            if cache_identity(rt) != before or not torch.isfinite(x).all():
                raise ValueError('Condition ablation mutated clean history or became nonfinite')
            rows.append({'condition': label, 'points': points,
                         'rollout_vs_HR': compare(x, clean), 'rollout_vs_draft': compare(x, draft)})
    return {'position': 0, 'rows': rows, 'model_forwards_including_static': 13,
            'noise_and_noisy_start_unchanged': True, 'history': 'none',
            'baseline_rows': 'gt_incremental position0 in the main matched diagnostic',
            'oracle_scope': 'only current HR latent used as diagnostic condition; never deployment input',
            'interpretation': 'condition sensitivity and fidelity diagnostic; not a trained candidate or root-cause proof'}


def main():
    parser = argparse.ArgumentParser()
    for name in ('stage', 'manifest-sha', 'runtime-stage', 'root', 'train-manifest',
                 'dev-manifest', 'checkpoint', 'assets-manifest', 'predecessor-launch',
                 'evaluation', 'stop-receipt', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--physical-device', type=int, default=7)
    parser.add_argument('--admission-only', action='store_true')
    parser.add_argument('--evaluation-defer-receipt')
    parser.add_argument('--evaluation-defer-sha')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + '\n')
    def log(event, **values):
        row = {'event': event, 'seconds': time.monotonic() - started, **values}
        with (out / 'events.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    try:
        admission, plan, train, dev = admit(args)
        write('admission.json', admission)
        if args.admission_only:
            print(json.dumps(admission), flush=True)
            return
        require_ready(admission)
        write('device_lock_owner.json', verify_parent_locks(args.physical_device))
        smi = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, check=True).stdout
        (out / 'npu_smi_before_model.txt').write_text(smi)
        if f'No running processes found in NPU {args.physical_device}' not in smi:
            raise RuntimeError('Card busy before model import')
        memory = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
        if memory['MemAvailable'] < 100*(1 << 30):
            raise RuntimeError('Insufficient available host memory')
        sys.path.insert(0, str(Path(args.runtime_stage) / 'src'))
        import numpy as np
        import torch
        import torch_npu
        from paged_runtime import Runtime
        from causal_attention import position_noise
        from streaming_refiner import StreamingRefiner
        torch.set_num_threads(4)
        torch.manual_seed(plan['seed'])
        torch.npu.set_device('npu:0')
        torch.npu.conv.allow_hf32 = False
        torch.npu.matmul.allow_hf32 = False
        torch.set_float32_matmul_precision('highest')
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if (checkpoint['step'] != 2500 or checkpoint['format'] != 'streaming-refiner-adapter-v1'
                or checkpoint['upsampler_sha256'] != plan['upsampler_sha256']):
            raise ValueError('Checkpoint metadata mismatch')
        condition = torch.load(dev['condition']['path'], map_location='cpu', weights_only=True)
        rt = Runtime(args.root, resident=plan['resident_layers'], canonical_rows=True, log=log)
        rt.load_adapter(checkpoint['adapter'])
        del checkpoint
        all_results = []
        for rec in dev['samples'] + train['samples']:
            sample = torch.load(rec['path'], map_location='cpu', weights_only=True)
            if sample['sample_id'] != rec['sample_id']:
                raise ValueError('Sample identity mismatch')
            if any(tuple(sample[k].shape) != (24, 67, 68, 120) for k in ('reference', 'upscaled')):
                raise ValueError('Full spatial cached sample required')
            sid = rec['sample_id']
            if sid in admission['reused_dev_histories']:
                desc = admission['reused_dev_histories'][sid]['output']
                self_history = torch.from_numpy(np.load(desc['path'], allow_pickle=False)[:, :12].copy())
                prefix_forwards = 0
                lineage = admission['reused_dev_histories'][sid]
            else:
                runner = StreamingRefiner(rt, condition, sample_id=sid, seed=plan['seed'])
                values = [runner.push(sample['upscaled'][None, :, i:i+1]) for i in range(12)]
                self_history = torch.cat(values, 2)[0]
                path = out / (sid + '_self_history_first12.npy')
                with path.open('xb') as f:
                    np.save(f, self_history.numpy(), allow_pickle=False)
                prefix_forwards = runner.model_forwards + 1
                lineage = {'path': str(path), 'sha256': sha(path), 'shape': list(self_history.shape),
                           'production_runner_audit': runner.audit()}
            result = diagnose_sample(rt, sample, condition, self_history, plan, position_noise, log)
            result['first_latent_condition_ablation'] = diagnose_initial_condition(rt, sample, condition, plan, position_noise)
            result.update({'split': 'train' if sid in plan['train'] else 'dev',
                           'sample_payload_sha256': rec['sha256'], 'self_history_lineage': lineage,
                           'new_self_prefix_forwards': prefix_forwards, 'checkpoint_sha256': plan['checkpoint_sha256']})
            write(sid + '.json', result)
            all_results.append({'sample_id': sid, 'receipt_sha256': sha(out / (sid + '.json'))})
            rt.cache = None
            log('sample_diagnostic_complete', sample_id=sid)
        write('result.json', {'status': 'matched_history_diagnostic_complete_interpretation_pending',
              'samples': all_results, 'plan_sha256': admission['plan_sha256'],
              'checkpoint_sha256': plan['checkpoint_sha256'], 'new_optimizer_updates': 0,
              'full226_evaluation': False, 'continuous_visual_reviewed': False,
              'model_accepted': False, 'root_cause_proven': False,
              'seconds': time.monotonic() - started})
    except BaseException as error:
        write('failure.json', {'type': type(error).__name__, 'error': str(error),
              'traceback': traceback.format_exc(), 'torch_imported': 'torch' in sys.modules,
              'torch_npu_imported': 'torch_npu' in sys.modules, 'new_optimizer_updates': 0})
        raise


if __name__ == '__main__':
    main()
