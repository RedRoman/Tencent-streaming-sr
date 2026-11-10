"""Read-only full-model localization of the preserved four-latent parity failure.

No optimizer, weight changes or production-runtime edits. Counterfactual grouped
embeddings are diagnostic only. Full-prefix calls are offline references.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from diagnose_tf2500_history_v3 import admit, require_ready, sha, pinned, compare
from decoder_validation_admission import predecessor_state, verify_parent_locks


def main():
    p = argparse.ArgumentParser()
    for key in ('stage', 'manifest-sha', 'runtime-stage', 'root', 'train-manifest',
                'dev-manifest', 'checkpoint', 'assets-manifest', 'predecessor-launch',
                'evaluation', 'stop-receipt', 'evaluation-defer-receipt',
                'evaluation-defer-sha', 'diagnostic-launch', 'diagnostic-failure', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--physical-device', type=int, default=7)
    p.add_argument('--admission-only', action='store_true')
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + '\n')
    def log(event, **values):
        row = {'event': event, 'seconds': time.monotonic()-started, **values}
        with (out / 'events.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    try:
        admission, plan, train, dev = admit(a)
        pinned(a.diagnostic_launch, '23f9114b100e3871129311d4c0511648a1a8e44a6daef9d62bd08ffee547390a')
        pinned(a.diagnostic_failure, 'e7a2cac731d162419ab9ca6f16055a23ac7dd4adb4763c1d3c80ec73bfca6e35')
        prior = predecessor_state(a.diagnostic_launch)
        if not prior['ready']:
            raise RuntimeError('Failed diagnostic not fully reaped')
        admission['failed_diagnostic_predecessor'] = prior
        write('admission.json', admission)
        require_ready(admission)
        if a.admission_only:
            print(json.dumps(admission), flush=True)
            return
        write('device_lock_owner.json', verify_parent_locks(a.physical_device))
        smi = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, check=True).stdout
        (out / 'npu_smi_before_model.txt').write_text(smi)
        if f'No running processes found in NPU {a.physical_device}' not in smi:
            raise RuntimeError('Device occupied')
        mem = {v.split(':')[0]: int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines()}
        if mem['MemAvailable'] < 100*(1 << 30):
            raise RuntimeError('Insufficient host memory')
        sys.path.insert(0, str(Path(a.runtime_stage) / 'src'))
        import torch
        import torch_npu
        from paged_runtime import Runtime
        from causal_attention import position_noise
        torch.set_num_threads(4)
        torch.manual_seed(plan['seed'])
        torch.npu.set_device('npu:0')
        torch.npu.conv.allow_hf32 = False
        torch.npu.matmul.allow_hf32 = False
        torch.set_float32_matmul_precision('highest')
        rec = dev['samples'][0]
        sample = torch.load(rec['path'], map_location='cpu', weights_only=True)
        condition = torch.load(dev['condition']['path'], map_location='cpu', weights_only=True)
        ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
        rt = Runtime(a.root, resident=24, canonical_rows=True, log=log)
        rt.load_adapter(ckpt['adapter'])
        del ckpt
        hr, draft, sigma = sample['reference'], sample['upscaled'], plan['sigmas'][0]
        noise = position_noise((24, 1, 68, 120), sample['sample_id'], 3, plan['seed']).to(rt.device)
        noisy = (1-sigma)*hr[:, 3:4].to(rt.device) + sigma*noise
        context = hr[:, :4].clone().to(rt.device)
        context[:, -1:] = noisy
        report = {'sample_id': sample['sample_id'], 'checkpoint_sha256': plan['checkpoint_sha256'],
                  'source_manifest_sha256': a.manifest_sha, 'new_optimizer_updates': 0,
                  'input_dtypes': {k: str(sample[k].dtype) for k in ('reference', 'upscaled')},
                  'scope': 'four full-spatial latents; full current layer tensors; sampled history KV rows',
                  'production_runtime_modified': False, 'model_accepted': False}
        original_embeds = rt.embeds
        original_layer = rt.layer
        phase, saved, layer_rows = 'none', {}, []

        def trace_layer(index, x, call, release=True, capture=False):
            nonlocal saved
            if phase == 'offline':
                before = x[:, -2040:].detach().cpu()
                y = original_layer(index, x, call, release=release, capture=True)
                kv = rt.processors[index].captured
                keys = sorted(set(kv.frames))
                selected = [i for f in keys for i in [j for j, v in enumerate(kv.frames) if v == f][:4]]
                saved[index] = {'input': before, 'output': y[:, -2040:].detach().cpu(),
                                'k': kv.k[:, selected].clone(), 'v': kv.v[:, selected].clone(),
                                'frames': tuple(kv.frames[j] for j in selected)}
                rt.processors[index].captured = None
                return y
            if phase == 'current':
                old = saved[index]
                before = compare(x.cpu(), old['input'])
                past = call.attention.past
                ids = [i for f in sorted(set(past.frames)) for i in [j for j, v in enumerate(past.frames) if v == f][:4]]
                refids = [i for i, f in enumerate(old['frames']) if f in set(past.frames)]
                history = {'k': compare(past.k[:, ids], old['k'][:, refids]),
                           'v': compare(past.v[:, ids], old['v'][:, refids])}
                y = original_layer(index, x, call, release=release, capture=capture)
                layer_rows.append({'layer': index, 'input': before, 'output': compare(y.cpu(), old['output']),
                                   'history_KV_sampled_rows': history})
                return y
            return original_layer(index, x, call, release=release, capture=capture)

        def grouped_embeds(x, c):
            return torch.cat([original_embeds(x[:, i:i+1], c[:, i:i+1]) for i in range(x.shape[1])], 1)

        with torch.no_grad():
            rt.static_inputs(condition)
            joint = original_embeds(context, draft[:, :4].to(rt.device))
            grouped = grouped_embeds(context, draft[:, :4].to(rt.device))
            report['embedding_joint_vs_per_latent'] = [compare(joint[:, i*2040:(i+1)*2040], grouped[:, i*2040:(i+1)*2040]) for i in range(4)]
            t, c = rt.time([1., 1.-sigma])
            one = [rt.time([v]) for v in [1., 1.-sigma]]
            report['time_joint_vs_separate'] = {'temb': compare(t, torch.cat([v[0] for v in one])),
                                                'curve': compare(c, torch.cat([v[1] for v in one]))}
            del joint, grouped
            write('input_probes.json', report)
            log('input_probes_complete', embeddings=report['embedding_joint_vs_per_latent'], time=report['time_joint_vs_separate'])
            rt.layer = trace_layer
            phase = 'offline'
            offline = rt.offline(context, draft[:, :4], condition, [0, 1, 2, 3], sigma)
            phase = 'history'
            rt.start_stream(condition)
            for i in range(3):
                rt.stream_velocity(hr[:, i:i+1], draft[:, i:i+1], 0, commit=True)
            phase = 'current'
            stream = rt.stream_velocity(noisy, draft[:, 3:4], sigma)
            rt.layer = original_layer
            report['original_full_output_gap'] = compare(stream, offline)
            report['layers'] = layer_rows
            saved.clear()
            write('original_trace.json', report)
            log('original_trace_complete', gap=report['original_full_output_gap'])
            # A controlled counterfactual only when grouped input projection differs.
            if any(v['max_abs'] for v in report['embedding_joint_vs_per_latent']):
                rt.embeds = grouped_embeds
                corrected = rt.offline(context, draft[:, :4], condition, [0, 1, 2, 3], sigma)
                report['diagnostic_grouped_embedding_output_gap'] = compare(stream, corrected)
                rt.embeds = original_embeds
            report['status'] = 'localization_complete_interpretation_pending'
            report['seconds'] = time.monotonic()-started
            write('result.json', report)
            log('trace_complete', original=report['original_full_output_gap'], grouped=report.get('diagnostic_grouped_embedding_output_gap'))
    except BaseException as error:
        write('failure.json', {'type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc(),
                              'new_optimizer_updates': 0})
        raise


if __name__ == '__main__':
    main()
