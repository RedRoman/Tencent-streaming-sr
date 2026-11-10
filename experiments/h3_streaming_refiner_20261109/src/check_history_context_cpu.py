"""Distinguish bounded KV inference from truncated teacher-forcing context.

Uses the actual runtime and a tiny random three-layer H3 on CPU, at full
68x120 spatial geometry. No pretrained quality or optimizer claim is made.
The deployed source bytes are verified and never changed by this diagnostic.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


PINS = {
    'causal_attention.py': 'f21ba96716d5cf91603e67e6d4d4d603aade47c054f81bd479ab99b341b6751f',
    'paged_runtime.py': '96c18f6d5bbecd7cd1341c0beef34c80c94f474b1bb86638a1e7af2125a7357f',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-stage', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    stage = Path(args.runtime_stage)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve the prior diagnostic output')
    for name, digest in PINS.items():
        if sha(stage / 'src' / name) != digest:
            raise ValueError('Frozen runtime source changed: ' + name)
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    os.environ['ASCEND_VISIBLE_DEVICES'] = ''
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(stage / 'src'))
    import torch
    from torch import nn
    from diffusers import MiniMaxH3Transformer3DModel
    from paged_runtime import Runtime, AdapterLinear, canonicalize
    from causal_attention import CausalAttention, position_noise

    torch.set_num_threads(2)
    torch.manual_seed(20261110)
    started = time.monotonic()
    runtime = Runtime.__new__(Runtime)
    runtime.device = torch.device('cpu')
    runtime.log = lambda *a, **kw: None
    runtime.resident = 0
    runtime.model = MiniMaxH3Transformer3DModel(
        num_attention_heads=1, attention_head_dim=128, hidden_size=32,
        num_layers=3, num_refiner_layers=1, ffn_dim=64,
        time_embed_hidden_dim=32, time_embed_dim=16,
    ).float().eval().requires_grad_(False)
    for name, child in runtime.model.named_children():
        if name not in ('proj_in', 'audio_proj_in', 'time_proj',
                        'time_embedder', 'proj_out', 'audio_proj_out', 'rope'):
            child.bfloat16()
    runtime.host = {}
    runtime.processors = []
    runtime.layer_params = []
    for i, block in enumerate(runtime.model.transformer_blocks):
        for path in ('attn.to_q', 'attn.to_k', 'attn.to_v', 'attn.to_out.0'):
            parent, key = path.rsplit('.', 1)
            setattr(block.get_submodule(parent), key,
                    AdapterLinear(block.get_submodule(path), 2))
        processor = CausalAttention(chunk_size=256)
        block.attn.set_processor(processor)
        runtime.processors.append(processor)
        canonicalize(block, processor)
        runtime.layer_params.append(tuple(p for p in block.parameters() if p.requires_grad))
        runtime.host[i] = [(module, key, p.detach()) for module in block.modules()
                           for key, p in module._parameters.items()
                           if p is not None and not p.requires_grad]
        runtime.host[i] += [(module, key, b.detach()) for module in block.modules()
                            for key, b in module._buffers.items() if b is not None]
    runtime.curves = SimpleNamespace(coords=None)

    def pre(module, arguments, kwargs):
        runtime.curves.coords = torch.zeros(len(kwargs['timestep']), 8)

    runtime.curves.pre = pre
    runtime.condition = nn.Linear(96, 32, bias=False)
    nn.init.zeros_(runtime.condition.weight)
    runtime.params = [p for group in runtime.layer_params for p in group]
    runtime.params += list(runtime.condition.parameters())
    runtime.cache = None
    sample = {'audio_latents': torch.randn(2, 32, 2),
              'prompt_embeds': torch.empty(0, 5120)}
    hr = torch.randn(24, 4, 68, 120)
    draft = hr + .1 * torch.randn_like(hr)
    sigma = .909375
    x = hr.clone()
    x[:, 3:4] = ((1 - sigma) * hr[:, 3:4]
                 + sigma * position_noise((24, 1, 68, 120), 'history-context-cpu', 3, 9))

    def compare(a, b):
        delta = a.float() - b.float()
        return {'max_abs': float(delta.abs().max()),
                'relative_rms': float(delta.square().mean().sqrt()
                                      / b.float().square().mean().sqrt().clamp_min(1e-8))}

    def sequential(history):
        runtime.start_stream(sample)
        for frame in history:
            # Explicit skip only for the truncated diagnostic oracle. The
            # actual production StreamingRefiner never skips frame indices.
            runtime.next_frame = frame
            runtime.stream_velocity(hr[:, frame:frame+1], draft[:, frame:frame+1],
                                    0, commit=True)
        runtime.next_frame = 3
        return runtime.stream_velocity(x[:, 3:4], draft[:, 3:4], sigma)

    with torch.no_grad():
        full = runtime.offline(x, draft, sample, [0, 1, 2, 3], sigma)
        streamed = sequential([0, 1, 2])
        full_parity = compare(streamed, full)
        full_cache_bytes = sum(v.k.numel()*v.k.element_size()
                               + v.v.numel()*v.v.element_size() for v in runtime.cache)
        max_cached_frames = max(len(set(f for f in v.frames if f >= 0))
                                for v in runtime.cache)
        truncated = runtime.offline(x[:, [0, 2, 3]], draft[:, [0, 2, 3]],
                                    sample, [0, 2, 3], sigma)
        truncated_stream = sequential([0, 2])
        truncated_parity = compare(truncated_stream, truncated)
        context_gap = compare(truncated, full)
        # With a complete supplied history, both execution methods agree.
        assert full_parity['max_abs'] <= .03125, full_parity
        assert truncated_parity['max_abs'] <= .03125, truncated_parity
        assert max_cached_frames <= 2
        assert context_gap['max_abs'] > max(full_parity['max_abs'],
                                            truncated_parity['max_abs']), context_gap
        assert context_gap['relative_rms'] > 0

    result = {
        'status': 'passed_context_gap_reproduced_on_tiny_random_H3',
        'scope': 'actual frozen runtime; tiny random three-layer H3; four full68x120 latents; CPU only',
        'same_inputs_checkpoint_condition_positions_and_noise': True,
        'full_history_frames': [0, 1, 2],
        'truncated_history_frames': [0, 2],
        'current_frame': 3,
        'full_prefix_reference_vs_incremental_GT_KV': full_parity,
        'truncated_offline_vs_truncated_KV_oracle': truncated_parity,
        'truncated_vs_full_history_context_gap': context_gap,
        'stream_cache_max_video_frames': max_cached_frames,
        'tiny_stream_cache_bytes': full_cache_bytes,
        'full_prefix_computation_used_only_as_diagnostic_reference': True,
        'production_inference_changed': False,
        'optimizer_updates': 0,
        'new_accelerator_forwards': 0,
        'full33B_context_gap_measured': False,
        'TF2500_quality_root_cause_proven': False,
        'source_sha256': sha(__file__),
        'runtime_source_sha256': PINS,
        'seconds': time.monotonic() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        f.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
