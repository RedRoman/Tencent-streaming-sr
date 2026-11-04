"""NPU FSDP2 runtime for one frozen H3 refiner and two independent low-rank adapters."""
from pathlib import Path
from types import SimpleNamespace
import json
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard, MixedPrecisionPolicy
from safetensors.torch import load_file

from diffusers import MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
from open_h3.adapters import CurveContext, CurveLinear, CURVE_FILE, CURVE_TABLE
from open_h3.geometry import MiniMaxH3Geometry, build_t2va_layout, pack_audio_latents, unpatchify_video_rows
from temporal import build_temporal_plan, H3BlockCausalProcessor


class LowRank(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.a = nn.Linear(in_features, rank, bias=False, dtype=torch.float32)
        self.b = nn.Linear(rank, out_features, bias=False, dtype=torch.float32)
        nn.init.normal_(self.a.weight, std=0.01)
        nn.init.zeros_(self.b.weight)
    def forward(self, value):
        # FP32 masters, compute dtype follows gathered weights under FSDP2.
        return self.b(self.a(value.to(self.a.weight.dtype)))


class RoleLinear(nn.Module):
    def __init__(self, base, rank, state):
        super().__init__()
        self.base = base
        self.generator = LowRank(base.in_features, base.out_features, rank)
        self.fake = LowRank(base.in_features, base.out_features, rank)
        self.state = state
    @property
    def weight(self):
        return self.base.weight
    def forward(self, value):
        result = self.base(value.to(self.base.weight.dtype))
        if self.state.role != 'real':
            result = result + getattr(self, self.state.role)(value).to(result.dtype)
        return result


def unmanaged(module):
    yield from module.parameters(recurse=False)
    for child in module.children():
        if not isinstance(child, FSDPModule):
            yield from unmanaged(child)


def shard_homogeneous(module, **options):
    # FSDP2 requires each group to have one original parameter dtype.
    dtypes = {p.dtype for p in unmanaged(module)}
    if len(dtypes) > 1:
        direct = {p.dtype for p in module.parameters(recurse=False)}
        if len(direct) > 1:
            raise RuntimeError(f'Mixed direct dtypes in {type(module)}: {direct}')
        target = next(iter(direct)) if direct else (torch.bfloat16 if torch.bfloat16 in dtypes else torch.float32)
        def split(parent):
            for child in parent.children():
                if isinstance(child, FSDPModule):
                    continue
                types = {p.dtype for p in unmanaged(child)}
                if not types or types == {target}:
                    continue
                own = {p.dtype for p in child.parameters(recurse=False)}
                if type(child).forward is not nn.Module.forward and (len(types) == 1 or (own and own != {target})):
                    shard_homogeneous(child, **options)
                else:
                    split(child)
        split(module)
    assert len({p.dtype for p in unmanaged(module)}) <= 1
    return fully_shard(module, **options)


def install_curves(model, assets):
    curves = load_file(str(Path(assets) / CURVE_FILE))
    ctx = CurveContext(Path(assets) / CURVE_TABLE)
    ctx.strength = 1.0
    roots = sorted(k[len('transformer.'):-len('.curve_A.weight')] for k in curves if k.endswith('.curve_A.weight'))
    assert len(roots) == 50
    for name in roots:
        parent, key = name.rsplit('.', 1)
        a, b = (curves[f'transformer.{name}.curve_{part}.weight'].float() for part in ('A', 'B'))
        setattr(model.get_submodule(parent), key, CurveLinear(model.get_submodule(name), a, b, ctx))
    model.register_forward_pre_hook(ctx.pre, with_kwargs=True)
    return ctx


class Runtime:
    def __init__(self, merged, assets, rank, world, adapter_rank=8, block_frames=5, *, base_preflight=False):
        self.device = torch.device(f'npu:{rank}')
        self.block_frames = block_frames
        self.state = SimpleNamespace(role='real')
        started = time.time()
        def progress(event, **values):
            if rank == 0:
                print(json.dumps({'event': event, 'runtime_seconds': round(time.time()-started,3), **values}), flush=True)
        self.model = MiniMaxH3Transformer3DModel.from_pretrained(merged, torch_dtype=torch.bfloat16,
                                                               local_files_only=True)
        progress('weights_mapped', path=str(merged), base_preflight=base_preflight)
        self.model.requires_grad_(False)
        # The explicit base-only switch is restricted to the environment preflight.
        # train.py never enables it; actual refiner runs require every Sharp branch.
        self.curves = None if base_preflight else install_curves(self.model, assets)
        progress('frozen_refiner_ready', sharp_branches=0 if base_preflight else 50)
        count = 0
        for block in self.model.transformer_blocks:
            for path in ('attn.to_q', 'attn.to_k', 'attn.to_v', 'attn.to_out.0'):
                parent, key = path.rsplit('.', 1)
                setattr(block.get_submodule(parent), key,
                        RoleLinear(block.get_submodule(path), adapter_rank, self.state))
                count += 1
            block.attn.set_processor(H3BlockCausalProcessor(query_chunk_size=256))
        assert count == 200
        self.model.enable_gradient_checkpointing()
        mesh = init_device_mesh('npu', (world,))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                      cast_forward_inputs=False)
        for index, block in enumerate(self.model.transformer_blocks):
            shard_homogeneous(block, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
            if (index+1) % 10 == 0:
                progress('transformer_blocks_sharded', blocks=index+1)
        for block in self.model.token_refiner.refiner_blocks:
            shard_homogeneous(block, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        root_policy = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32, cast_forward_inputs=False)
        shard_homogeneous(self.model, mesh=mesh, mp_policy=root_policy, reshard_after_forward=True)
        self.model.train()
        self.params = {role: [p for name, p in self.model.named_parameters()
                              if p.requires_grad and f'.{role}.' in name]
                       for role in ('generator', 'fake')}
        assert all(self.params.values())
        self.layouts = {}
        progress('runtime_ready', allocated_gb=torch.npu.memory_allocated()/1e9)

    def set_role(self, role):
        if role not in ('generator', 'fake', 'real'):
            raise ValueError(role)
        self.state.role = role

    def forward(self, value, sample, sigma, *, causal=False, block_index=0, **unused):
        _, frames, height, width = value.shape
        audio = sample['audio_latents'].to(self.device)
        text = sample['prompt_embeds'].to(self.device)
        if text.ndim == 2:
            text = text.unsqueeze(0)
        if audio.ndim == 3:
            audio = audio.unsqueeze(0)
        geometry = MiniMaxH3Geometry(frames, height, width, audio.shape[-1])
        key = (sample['metadata']['id'], frames, height, width, audio.shape[-1])
        if key not in self.layouts:
            self.layouts[key] = build_t2va_layout(sample['text_token_tags'], geometry).to(self.device)
        layout = self.layouts[key]
        plan = build_temporal_plan(value, layout, block_frames=self.block_frames if causal else frames,
                                   current_block=block_index if causal else 0, sigma=sigma)
        for block in self.model.transformer_blocks:
            block.attn.processor.set_plan(plan)
            block.attn.processor.mode = 'causal' if causal else 'dense'
            block.attn.processor.detach_history = causal
        # Match the released refiner's BF16 model input; the Euler state and the
        # checkpoint's projection/time/output FP32 islands retain their precision.
        rows = patchify_video_latents(value.unsqueeze(0), self.model.config.patch_size).reshape(1, geometry.num_video_rows, 96).to(torch.bfloat16)
        result = self.model(hidden_states=rows, audio_hidden_states=pack_audio_latents(audio),
                            encoder_hidden_states=text, timestep=plan.timesteps, timestep_indices=plan.timestep_indices,
                            token_tags=layout.token_tags, position_ids=layout.position_ids,
                            video_indices=layout.video_indices, audio_indices=layout.audio_indices,
                            text_indices=layout.text_indices, attention_kwargs={}, return_dict=False)[0]
        return unpatchify_video_rows(result, geometry)[0].float()

    def adapter_state(self):
        result = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                value = param.detach().full_tensor() if hasattr(param, 'full_tensor') else param.detach()
                result[name] = value.cpu()
        return result

    @torch.no_grad()
    def load_adapter_state(self, state):
        from torch.distributed.tensor import distribute_tensor
        trainable = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        if set(state) != set(trainable):
            raise ValueError('Checkpoint adapter keys do not match this model')
        for name, param in trainable.items():
            value = state[name].to(self.device, dtype=param.dtype)
            if hasattr(param, 'device_mesh'):
                value = distribute_tensor(value, param.device_mesh, param.placements)
            param.copy_(value)
