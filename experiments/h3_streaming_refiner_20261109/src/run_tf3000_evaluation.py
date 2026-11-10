"""Finite held-out evaluation after TF3000 exits and relinquishes its locks.

No torch imports in admission. Reuse the frozen native dense first clip only
after verifying its original lineage and complete payloads. All new model
workers are byte-identical to the already tested frozen workers.
"""
import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

from decoder_validation_admission import predecessor_state, verify_parent_locks
from run_evaluation_continuation_v2 import (
    validate_dense_latent, validate_dense_pixels, validate_latent, validate_pixels,
)


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def pinned(path, digest):
    if sha(path) != digest:
        raise ValueError('Pinned file changed: ' + Path(path).name)
    return read(path)


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def validate_protocol(policy, original):
    allowed = {'schema', 'frozen_utc', 'optimizer_updates_observed_before_freeze',
               'checkpoints', 'arms'}
    for key, value in original.items():
        if key not in allowed and policy.get(key) != value:
            raise ValueError('Frozen evaluation contract changed: ' + key)
    if policy['checkpoints'] != [1000, 2000, 3000]:
        raise ValueError('Expected exactly the user-directed checkpoint cohort')
    expected = ['frozen_upsampler_no_refiner', 'released_dense_refiner']
    expected += ['strict_streaming_step' + str(s) for s in policy['checkpoints']]
    if policy['arms'] != expected or policy['execution_checkpoint_order'] != [3000, 1000, 2000]:
        raise ValueError('Unexpected arms or execution order')
    if policy['test_unlocked'] or policy['step0_identity_guaranteed']:
        raise ValueError('Unsupported acceptance claim')


def training_complete(status, launch, predecessor):
    return (status.get('status') == 'TF3000_complete_evaluation_pending'
            and status.get('optimizer_updates') == 3000
            and status.get('new_optimizer_updates') == 2800
            and launch.get('status') == 'complete' and launch.get('exit_code') == 0
            and launch.get('child_reaped') is True and predecessor['ready'])


def admit(args):
    stage = Path(args.stage)
    sources = pinned(stage / 'manifest.json', args.manifest_sha)
    for name, digest in sources['files'].items():
        if sha(stage / name) != digest:
            raise ValueError('Staged source changed: ' + name)
    policy = read(stage / 'config/evaluation_tf3000_v1.json')
    original = pinned(stage / 'config/evaluation_parent_v1.json', sources['parent_protocol_sha256'])
    validate_protocol(policy, original)
    data = pinned(args.data_manifest, sources['parent_dev_manifest_sha256'])
    if data['protocol_sha256'] != sources['parent_protocol_sha256']:
        raise ValueError('Original dev manifest/protocol binding changed')
    if data['accepted_upsampler_sha256'] != policy['shared_input']['upsampler_sha256']:
        raise ValueError('Frozen upsampler changed')
    if [r['sample_id'] for r in data['samples']] != policy['dev']:
        raise ValueError('Dev sample order changed')
    assets = pinned(args.assets_manifest, sources['assets_manifest_sha256'])
    if assets['status'] != 'complete':
        raise ValueError('Asset inventory incomplete')
    prior_plan = Path(args.prior_stage) / 'h3_production67_limiteddev_full226_plan_v1.json'
    pinned(prior_plan, sources['prior_decoder_plan_sha256'])
    launch = read(args.training_launch)
    command = launch['command']
    if sha(command[1]) != sources['training_source_sha256']:
        raise ValueError('Training worker changed')
    if command[command.index('--config-sha') + 1] != sources['training_config_sha256']:
        raise ValueError('Training config binding changed')
    pinned(command[command.index('--config') + 1], sources['training_config_sha256'])
    if Path(command[command.index('--output') + 1]).resolve() != Path(args.training).resolve():
        raise ValueError('Training output binding changed')
    predecessor = predecessor_state(args.training_launch)
    status = read(Path(args.training) / 'status.json')
    checkpoints = {}
    missing = []
    for step in policy['checkpoints']:
        name = Path(args.training) / f'checkpoint_step{step:04d}.json'
        if not name.exists():
            missing.append(step)
            continue
        record = read(name)
        payload = Path(args.training) / f'adapter_step{step:04d}.pt'
        if record['step'] != step or payload.stat().st_size != record['adapter_bytes']:
            raise ValueError('Checkpoint receipt mismatch')
        if sha(payload) != record['adapter_sha256']:
            raise ValueError('Checkpoint payload changed')
        if step in (1000, 2000) and record['adapter_sha256'] != sources['known_adapter_sha256'][str(step)]:
            raise ValueError('Audited checkpoint changed')
        checkpoints[step] = record
    ready = not missing and training_complete(status, launch, predecessor)
    report = {'status': 'CPU_admission_complete', 'ready_for_NPU': ready,
              'manifest_sha256': args.manifest_sha,
              'protocol_sha256': sha(stage / 'config/evaluation_tf3000_v1.json'),
              'parent_dev_manifest_sha256': sources['parent_dev_manifest_sha256'],
              'training_status': status, 'training_predecessor': predecessor,
              'checkpoint_receipts': checkpoints, 'missing_checkpoints': missing,
              'runtime_started': False, 'new_optimizer_updates': 0,
              'new_accelerator_forwards': 0, 'dense_reuse_payload_check_pending': True}
    return report, sources, policy, data


def reuse_dense(args, sources, policy):
    folder = Path(args.dense_preview)
    result = pinned(folder / 'result.json', sources['dense_result_sha256'])
    if (result['status'] != 'full226_dense_reference_complete_visual_review_pending'
            or result['source_manifest_sha256'] != sources['native_dense_parent_manifest_sha256']
            or result['optimizer_updates'] != 0):
        raise ValueError('Original dense reference lineage changed')
    def checked(desc):
        return pinned(desc['path'], desc['sha256'])
    latent = checked(result['latent'])
    gate = pinned(folder / 'latent/upstream_gate.json', sources['dense_gate_sha256'])
    validate_dense_latent(latent, policy['dev'][0], sources['parent_dev_manifest_sha256'],
                          sources['parent_protocol_sha256'], gate)
    pixels = checked(result['pixels'])
    validate_dense_pixels(pixels, policy['dev'][0], sources['parent_dev_manifest_sha256'],
                          result['latent']['sha256'])
    return result['pixels'], {'latent': result['latent'], 'pixels': result['pixels'],
                             'gate_sha256': sha(folder / 'latent/upstream_gate.json'),
                             'same_inputs_geometry_decoder_metrics_verified': True,
                             'original_receipts_preserved': True}


def main():
    p = argparse.ArgumentParser()
    for name in ('stage', 'manifest-sha', 'root', 'training', 'training-launch',
                 'data-manifest', 'assets-manifest', 'prior-stage', 'dense-preview', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--physical-device', type=int, default=7)
    p.add_argument('--admission-only', action='store_true')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    completed = []
    try:
        admission, sources, policy, data = admit(args)
        write(out / 'admission.json', admission)
        if args.admission_only:
            print(json.dumps(admission), flush=True)
            return
        if not admission['ready_for_NPU']:
            raise RuntimeError('TF3000 must finish successfully, save all checkpoints, and exit/reap before evaluation')
        write(out / 'device_lock_owner.json', verify_parent_locks(args.physical_device))
        dense_pixels, reuse = reuse_dense(args, sources, policy)
        stage = Path(args.stage)
        protocol = stage / 'config/evaluation_tf3000_v1.json'
        derived = dict(data)
        derived['protocol_sha256'] = sha(protocol)
        data_path = out / 'dev_manifest_tf3000.json'
        write(data_path, derived)
        data_sha = sha(data_path)
        checkpoint_hashes = {s: r['adapter_sha256'] for s, r in admission['checkpoint_receipts'].items()}
        write(out / 'inputs.json', {'manifest_sha256': args.manifest_sha,
              'protocol_sha256': sha(protocol), 'data_manifest_sha256': data_sha,
              'parent_data_manifest_sha256': sources['parent_dev_manifest_sha256'],
              'derived_manifest_changes_only_protocol_binding': True,
              'checkpoint_sha256': checkpoint_hashes, 'reused_dense': reuse,
              'training_updates': 3000, 'stream_resident_layers': 40,
              'new_optimizer_updates': 0, 'strict_decoder_validated': False})
        common = ['--root', args.root, '--data-manifest', str(data_path),
                  '--data-manifest-sha', data_sha, '--protocol', str(protocol),
                  '--assets-manifest', args.assets_manifest,
                  '--assets-manifest-sha', sources['assets_manifest_sha256']]
        pixel_common = ['--prior-stage', args.prior_stage,
                        '--prior-plan-sha', sources['prior_decoder_plan_sha256'],
                        '--data-manifest', str(data_path), '--data-manifest-sha', data_sha]
        def run(worker, arguments, name):
            folder = out / name
            print(json.dumps({'event': 'start_worker', 'worker': worker, 'output': str(folder)}), flush=True)
            with (out / (name + '.log')).open('x') as log:
                subprocess.run([sys.executable, str(stage / 'src' / worker), *arguments,
                                '--output', str(folder)], stdout=log, stderr=subprocess.STDOUT, check=True)
            record = {'path': str(folder / 'result.json'), 'sha256': sha(folder / 'result.json')}
            completed.append(record)
            write(out / 'progress.json', {'completed': completed, 'reused_dense': reuse,
                                         'elapsed_seconds': time.time() - started})
            return record
        pixels = [dense_pixels]
        jobs = [(s, sid) for s in policy['execution_checkpoint_order'] for sid in policy['dev']]
        jobs += [(None, sid) for sid in policy['dev'][1:]]
        for step, sid in jobs:
            stem = ('dense' if step is None else f'step{step:04d}') + '_' + sid
            extra = [] if step is None else [
                '--checkpoint', str(Path(args.training) / f'adapter_step{step:04d}.pt'),
                '--checkpoint-sha', checkpoint_hashes[step], '--resident', '40']
            worker = 'evaluate_dense_reference.py' if step is None else 'evaluate_streaming.py'
            latent = run(worker, common + extra + ['--sample-id', sid], stem + '_latent')
            lr = pinned(latent['path'], latent['sha256'])
            if step is not None:
                validate_latent(lr, step, checkpoint_hashes[step], sid, data_sha, sha(protocol))
            else:
                gate = read(Path(latent['path']).parent / 'upstream_gate.json')
                validate_dense_latent(lr, sid, data_sha, sha(protocol), gate)
            pixel = run('evaluate_pixels.py', pixel_common + ['--prediction', latent['path'],
                         '--prediction-sha', latent['sha256']], stem + '_pixels')
            pr = pinned(pixel['path'], pixel['sha256'])
            if step is None:
                validate_dense_pixels(pr, sid, data_sha, latent['sha256'])
            else:
                validate_pixels(pr, step, checkpoint_hashes[step], sid, data_sha, latent['sha256'])
            pixels.append(pixel)
        records = [pinned(v['path'], v['sha256']) for v in pixels]
        if len(records) != 12:
            raise ValueError('Incomplete fixed cohort')
        aggregates = {}
        for label in [f'strict_streaming_step{s}' for s in policy['checkpoints']] + ['released_dense_refiner']:
            rows = [r for r in records if r['arm'] == label]
            if len(rows) != 3 or {r['sample_id'] for r in rows} != set(policy['dev']):
                raise ValueError('Missing/duplicate held-out result')
            mean = {k: statistics.mean(r['nativeHR_metrics']['regions']['full226'][k] for r in rows)
                    for k in ('pixel_l1', 'lpips', 'psnr_db', 'delta_l1')}
            base = {k: statistics.mean(r['no_refiner_nativeHR_metrics']['regions']['full226'][k] for r in rows)
                    for k in ('pixel_l1', 'lpips')}
            ratios = {k: mean[k] / base[k] for k in base}
            aggregates[label] = {'equal_clip': mean, 'ratios_vs_no_refiner': ratios,
                'historical_2percent_metric_gate_passed': all(v <= 1.02 for v in ratios.values()),
                'zero_regression_on_reported_L1_LPIPS_means': all(v <= 1 for v in ratios.values())}
        write(out / 'result.json', {'status': 'full226_cohort_complete_visual_and_decoder_acceptance_pending',
              'aggregates': aggregates, 'pixel_results': pixels, 'accepted_refiner': None,
              'strict_decoder_validated': False, 'continuous_visual_reviewed': False,
              'test_unlocked': False, 'optimizer_updates_this_evaluation': 0,
              'elapsed_seconds': time.time() - started})
    except BaseException as error:
        write(out / 'failure.json', {'type': type(error).__name__, 'error': str(error),
              'traceback': traceback.format_exc(), 'completed': completed, 'new_optimizer_updates': 0})
        raise


if __name__ == '__main__':
    main()
