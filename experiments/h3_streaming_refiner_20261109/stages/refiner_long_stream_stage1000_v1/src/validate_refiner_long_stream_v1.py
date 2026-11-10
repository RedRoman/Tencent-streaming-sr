"""Real 33B, full-spatial 137-latent engineering stress of exact stage1000.

The authorized 67-latent draft is looped; this is not 137 unique video latents.
No optimizer, decoder, perceptual acceptance or real-time claim is made here.
All inference classes are imported unchanged from the pinned evaluation stage.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(p)


def pinned(path, digest):
    assert sha(path) == digest, str(path)
    return read(path)


def source_indices(start, count):
    assert start >= 0 and count > 0
    return [i % 67 for i in range(start, start + count)]


def predecessor_ready(plan):
    launch = read(plan['predecessor_launch'])
    absent = all(not Path('/proc', str(launch[k])).exists()
                 for k in ('parent_pid', 'child_pid'))
    terminal = (launch.get('status') == 'complete' and
                launch.get('exit_code') == 0 and launch.get('child_reaped') is True)
    result_path = Path(plan['predecessor_result'])
    result = read(result_path) if result_path.exists() else None
    video_ready = bool(result and result['status'] ==
                       'anchored_stage2000_full226_video_ready_visual_review_pending'
                       and result['sample_id'] == 'WingIt_dev_300_525')
    return {'ready': bool(absent and terminal and video_ready),
            'processes_absent': absent, 'exit0_reaped': terminal,
            'full_video_ready': video_ready, 'launch_sha256': sha(plan['predecessor_launch'])}


def admission(args):
    stage = Path(args.validation_stage)
    manifest = pinned(stage / 'manifest.json', args.manifest_sha)
    for name, digest in manifest['files'].items():
        assert sha(stage / name) == digest, name
    plan = pinned(stage / 'config/refiner_long_stream_v1.json', manifest['files']['config/refiner_long_stream_v1.json'])
    assert plan['stage_step'] == 1000 and plan['latent_count'] == 137
    assert plan['seed'] == 20261109 and plan['save_before_latents'] == [66, 134]
    assert plan['replay_count'] == 3 and plan['physical_device'] == 4
    old = SimpleNamespace(**plan['evaluation_admission_args'])
    sys.path.insert(0, str(Path(old.stage) / 'src'))
    from anchored_evaluation_admission import admit
    old_receipt, _, data = admit(old)
    cp = old_receipt['checkpoint']
    for key, value in plan['checkpoint_pins'].items():
        assert cp['record'][key] == value, key
    assert cp['receipt_sha256'] == plan['checkpoint_receipt_sha256']
    reference = pinned(plan['reference_latent_receipt'], plan['reference_latent_receipt_sha256'])
    assert reference['sample_id'] == plan['sample_id'] and reference['stage_step'] == 1000
    assert reference['checkpoint_sha256'] == plan['checkpoint_pins']['adapter_sha256']
    assert reference['output']['shape'] == [24, 67, 68, 120]
    assert sha(reference['output']['path']) == reference['output']['sha256']
    source_launch = read(plan['reference_launch'])
    assert source_launch['status'] == 'complete' and source_launch['exit_code'] == 0
    assert source_launch['child_reaped'] and all(not Path('/proc', str(source_launch[k])).exists()
                                               for k in ('parent_pid', 'child_pid'))
    pred = predecessor_ready(plan)
    return {'status': 'stdlib_actual_asset_admission', 'ready': pred['ready'],
            'predecessor': pred, 'source_and_checkpoint_verified': True,
            'validation_manifest_sha256': args.manifest_sha, 'checkpoint': cp,
            'reference': reference, 'torch_imported': 'torch' in sys.modules,
            'accelerator_imported': 'torch_npu' in sys.modules,
            'optimizer_updates': 0, 'accelerator_forwards': 0}, plan, data


def verify_direct_lock_parent(device):
    owner = os.getppid()
    cmd = Path('/proc', str(owner), 'cmdline').read_bytes().split(b'\0')
    assert Path(cmd[1].decode()).name == 'run_locked.py'
    assert sha(cmd[1].decode()) == '0dd684ffaf2b172426566dcfdee74b204052c6ad3d53622b9dc6fb7aea7bb7c6'
    identities = []
    for line in Path('/proc/locks').read_text().splitlines():
        p = line.split()
        if len(p) > 5 and p[1] == 'FLOCK' and p[3] == 'WRITE' and p[4] == str(owner):
            ma, mi, ino = p[5].split(':')
            identities.append((int(ma, 16), int(mi, 16), int(ino)))
    for path in (f'<AUTHORIZED_CACHE>/device_locks/npu{device}.lock',
                 f'/tmp/yunfeng_npu_{device}.lock', f'/tmp/codex-h3-vae-npu-{device}.lock'):
        s = Path(path).stat()
        assert (os.major(s.st_dev), os.minor(s.st_dev), s.st_ino) in identities
    assert all(os.environ.get(k) == str(device) for k in
               ('ASCEND_VISIBLE_DEVICES', 'ASCEND_RT_VISIBLE_DEVICES'))
    return {'owner_pid': owner, 'physical_device': device, 'all_three_locks_verified': True}


def validate_snapshot(snapshot, identity, expected_next):
    assert snapshot['format'] == 'private-refiner-stream-snapshot-v1'
    assert snapshot['identity'] == identity
    assert snapshot['next_frame'] == expected_next
    assert snapshot['model_forwards'] == expected_next * 4
    assert snapshot['text_count'] == 0
    assert len(snapshot['cache']) == 50
    wanted = {-1, 0, expected_next - 1}
    total = 0
    for item in snapshot['cache']:
        k, v, frames = item['k'], item['v'], item['frames']
        assert k.device.type == v.device.type == 'cpu'
        assert not k.requires_grad and not v.requires_grad
        assert k.shape == v.shape and k.shape[1] == len(frames)
        assert set(frames) == wanted
        assert k.dtype == v.dtype
        assert k.isfinite().all() and v.isfinite().all()
        total += k.numel() * k.element_size() + v.numel() * v.element_size()
    assert total == 6930022400


def run(args, out, receipt, plan, data, log):
    assert receipt['ready'], 'Predecessor remains active or incomplete'
    write(out / 'device_lock_owner.json', verify_direct_lock_parent(plan['physical_device']))
    import torch
    import torch_npu
    import numpy as np
    from paged_runtime import Runtime
    from anchored_refiner_v1 import AnchoredStreamingRefiner, ResidualHead, align_input_projection
    from causal_attention import KV, position_noise
    torch.set_num_threads(4)
    torch.manual_seed(plan['seed'])
    torch.npu.set_device('npu:0')
    torch.npu.conv.allow_hf32 = False
    torch.npu.matmul.allow_hf32 = False
    torch.set_float32_matmul_precision('highest')
    w = torch.load(receipt['checkpoint']['adapter_path'], map_location='cpu', weights_only=True)
    assert w['format'] == 'anchored-streaming-refiner-v1' and w['stage_step'] == 1000
    assert w['step'] == 3500 and w['residual_bound'] == .25
    rt = Runtime(plan['evaluation_admission_args']['root'], resident=24, canonical_rows=True, log=log)
    align_input_projection(rt)
    rt.load_adapter(w['adapter'])
    head = ResidualHead(device=rt.device, bound=.25)
    head.load_state_dict(w['residual_head'])
    head.eval()
    sr = next(v for v in data['samples'] if v['sample_id'] == plan['sample_id'])
    draft = torch.load(sr['path'], map_location='cpu', weights_only=True)['upscaled']
    assert draft.shape == (24, 67, 68, 120)
    condition = torch.load(data['condition']['path'], map_location='cpu', weights_only=True)
    reference = torch.from_numpy(np.load(receipt['reference']['output']['path'], allow_pickle=False))
    runner = AnchoredStreamingRefiner(rt, head, condition, sample_id=plan['sample_id'], seed=plan['seed'])
    assert rt.text_count == 0
    identity = {'manifest_sha256': args.manifest_sha, 'adapter_sha256': sha(receipt['checkpoint']['adapter_path']),
                'sample_id': plan['sample_id'], 'sample_sha256': sr['sha256'],
                'condition_sha256': data['condition']['sha256'], 'seed': plan['seed']}
    projection_rows = [[] for _ in rt.model.transformer_blocks]
    hooks = []
    for layer, block in enumerate(rt.model.transformer_blocks):
        def capture(module, inputs, layer=layer):
            projection_rows[layer].append(int(inputs[0].shape[1]))
        hooks.append(block.attn.to_k.register_forward_pre_hook(capture))
    private = out / 'private_stream_cache'
    private.mkdir()
    snapshots = {}
    values, audits, latencies = [], [], []
    try:
        for i in range(plan['latent_count']):
            if i in plan['save_before_latents']:
                state = {'format': 'private-refiner-stream-snapshot-v1', 'identity': identity,
                         'next_frame': rt.next_frame, 'model_forwards': runner.model_forwards,
                         'text_count': rt.text_count,
                         'cache': [{'k': c.k, 'v': c.v, 'frames': c.frames} for c in rt.cache]}
                validate_snapshot(state, identity, i)
                path = private / f'before_latent{i:03d}.pt'
                with path.open('xb') as f:
                    torch.save(state, f)
                snapshots[i] = {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
                del state
                log('private_cache_saved', next_latent=i, bytes=snapshots[i]['bytes'])
            start = time.monotonic()
            current = draft[None, :, source_indices(i, 1)]
            z = runner.push(current)
            torch.npu.synchronize()
            audit = runner.audit()
            assert z.shape == (1, 24, 1, 68, 120) and torch.isfinite(z).all()
            assert float((z - current).abs().max()) <= .25001
            assert audit['next_latent'] == i + 1 and audit['model_forwards'] == 4 * (i + 1)
            assert audit['recomputed_history_latents'] == 0 and audit['max_cached_video_frames'] <= 2
            assert all(rows[-4:] == [2040] * 4 and len(rows) == 4 * (i + 1) for rows in projection_rows)
            for c in rt.cache:
                assert c.k.device.type == c.v.device.type == 'cpu'
                assert set(c.frames) == {-1, 0, i}
            if i >= 1:
                assert audit['cache_bytes'] == 6930022400
            if i < 67:
                delta = float((z[0] - reference[:, i:i+1]).abs().max())
                assert delta == 0, f'Existing67 reproduction differs at {i}: {delta}'
            values.append(z)
            audits.append(audit)
            latencies.append(time.monotonic() - start)
            log('latent_complete', latent_zero_based=i, source_index=i % 67,
                latency_seconds=latencies[-1], cache_bytes=audit['cache_bytes'])
    finally:
        for hook in hooks:
            hook.remove()
    full = torch.cat(values, 2)
    assert full.shape == (1, 24, 137, 68, 120)
    payload = out / 'streaming_latents137.npy'
    with payload.open('xb') as f:
        np.save(f, full[0].numpy(), allow_pickle=False)
    replays = []
    for i in plan['save_before_latents']:
        record = snapshots[i]
        assert sha(record['path']) == record['sha256']
        state = torch.load(record['path'], map_location='cpu', weights_only=True)
        validate_snapshot(state, identity, i)
        rt.load_adapter(w['adapter'])
        head.load_state_dict(w['residual_head'])
        rt.cache = [KV(v['k'], v['v'], tuple(v['frames'])) for v in state['cache']]
        rt.next_frame = state['next_frame']
        rt.text_count = state['text_count']
        rt.cache_version = tuple(p._version for p in rt.params)
        resumed = AnchoredStreamingRefiner.__new__(AnchoredStreamingRefiner)
        resumed.runtime, resumed.head = rt, head
        resumed.sample_id, resumed.seed = plan['sample_id'], plan['seed']
        resumed.model_forwards = state['model_forwards']
        del state
        again = resumed.push(draft[None, :, source_indices(i, 3)])
        error = float((again - full[:, :, i:i+3]).abs().max())
        assert error == 0, f'Restored/chunked replay differs at {i}: {error}'
        assert resumed.audit()['next_latent'] == i + 3
        replays.append({'next_latent_at_restore': i, 'changed_push_chunk': 3, 'max_abs': error,
                        'weight_reload': True, 'serialized_real50layer_cache': True})
        log('restore_chunk_replay_passed', **replays[-1])
    noise = {}
    for i in (0, 66, 67, 68, 133, 134, 136):
        tensor = position_noise((24, 1, 68, 120), plan['sample_id'], i, plan['seed'])
        noise[str(i)] = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    assert len(set(noise.values())) == len(noise)
    write(out / 'result.json', {'status': 'real33B_long137_engineering_passed',
          'stage_step': 1000, 'training_step': 3500, 'identity': identity,
          'scope': 'Real50block33B, full68x120, looped authorized67draft, absolute137positions; not137unique video',
          'source_indices': source_indices(0, 137), 'first67_existing_latent_max_abs': 0,
          'replays': replays, 'cache_bytes_trace': [a['cache_bytes'] for a in audits],
          'main_model_forwards': 548, 'additional_replay_model_forwards': 24,
          'current_only_K_projection_calls_per_layer': [len(v) for v in projection_rows],
          'all_K_projection_rows': 2040, 'full_prefix_recomputation': False,
          'private_cache_snapshots': snapshots, 'private_caches_must_not_be_published': True,
          'output': {'path': str(payload), 'sha256': sha(payload), 'shape': [24, 137, 68, 120]},
          'absolute_noise_SHA': noise, 'per_latent_seconds': latencies,
          'optimizer_updates': 0, 'full137_offline_parity_tested': False,
          'stage2000_longstream_tested': False, 'continuous_visual_reviewed': False,
          'strict_decoder_validated': False, 'model_accepted': False})


def main():
    p = argparse.ArgumentParser()
    for k in ('validation-stage', 'manifest-sha', 'output'):
        p.add_argument('--' + k, required=True)
    p.add_argument('--admission-only', action='store_true')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    def log(event, **fields):
        row = {'event': event, 'elapsed_seconds': time.monotonic() - start, **fields}
        with (out / 'events.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    try:
        receipt, plan, data = admission(args)
        write(out / 'admission.json', receipt)
        if not args.admission_only:
            run(args, out, receipt, plan, data, log)
    except BaseException as e:
        write(out / 'failure.json', {'type': type(e).__name__, 'error': str(e),
                                   'traceback': traceback.format_exc(), 'optimizer_updates': 0})
        raise


if __name__ == '__main__':
    main()
