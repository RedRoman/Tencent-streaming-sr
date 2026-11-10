"""Identity-initialized bounded residual candidate, distinct from the old TF model.

The frozen upsampler draft is the direct output anchor. The original transformer
and trained adapters supply features, not an unconstrained generated output.
Inference remains one-latent-at-a-time with true bounded incremental KV.
"""
import torch
from torch import nn
from causal_attention import position_noise

SIGMAS = (.909375, 6/7, 2/3, 0.)


def align_input_projection(runtime):
    original = runtime.embeds
    def grouped(x, condition):
        return torch.cat([original(x[:, i:i+1], condition[:, i:i+1])
                          for i in range(x.shape[1])], 1)
    runtime.embeds = grouped


class ResidualHead(nn.Module):
    def __init__(self, device='cpu', bound=.25):
        super().__init__()
        self.bound = float(bound)
        self.in_proj = nn.Linear(48, 96, device=device, dtype=torch.float32)
        self.out_proj = nn.Linear(96, 24, device=device, dtype=torch.float32)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, velocity, draft, sigma):
        if x.shape != draft.shape or velocity.shape != draft.shape or draft.shape[0] != 24:
            raise ValueError('Residual head expects matching C,T,H,W tensors')
        raw_x0 = x.float() + float(sigma)*velocity.float()
        features = torch.cat((raw_x0-draft.float(), draft.float()), 0).permute(1, 2, 3, 0)
        raw = self.out_proj(torch.nn.functional.silu(self.in_proj(features.contiguous())))
        residual = self.bound*torch.tanh(raw/self.bound)
        return draft.float() + residual.permute(3, 0, 1, 2)


def schedule(stage_completed):
    if stage_completed < 0:
        raise ValueError('Negative stage step')
    pair = stage_completed//2
    slot = pair % 4
    pos = 0 if slot == 0 else 1+((pair//4*3+slot-1)*17) % 66
    frames = [pos] if slot in (0, 1) else sorted(set([0, pos-1, pos]))
    return {'sample_index': stage_completed % 2, 'latent_index': pos, 'frames': frames,
            'sigma': SIGMAS[pair % 3], 'noise_seed': 20261110+stage_completed//134,
            'history': 'none' if slot in (0, 1) else 'draft' if slot == 2 else 'draft_small_perturbation',
            'history_jitter_std': .025 if slot == 3 else 0.}


def training_input(sample, spec):
    """All forward inputs derive from draft/noise. Only current HR is the target."""
    frames, pos, sigma = spec['frames'], spec['latent_index'], spec['sigma']
    draft = sample['upscaled'][:, frames].float()
    x = draft.clone()
    if spec['history_jitter_std']:
        for i, absolute in enumerate(frames[:-1]):
            jitter = position_noise((24, 1, 68, 120), sample['sample_id'], absolute,
                                    spec['noise_seed']+100000)
            x[:, i:i+1] += (spec['history_jitter_std']*jitter).clamp(-.25, .25)
    noise = position_noise((24, 1, 68, 120), sample['sample_id'], pos, spec['noise_seed'])
    x[:, -1:] = (1-sigma)*draft[:, -1:]+sigma*noise
    target = sample['reference'][:, pos:pos+1].float()
    return x, draft, target


class AnchoredStreamingRefiner:
    def __init__(self, runtime, head, condition, *, sample_id, seed=20261109):
        self.runtime, self.head = runtime, head
        self.sample_id, self.seed = sample_id, seed
        self.runtime.start_stream(condition)
        self.model_forwards = 0

    @torch.no_grad()
    def push(self, draft):
        if draft.ndim != 5 or draft.shape[:2] != (1, 24) or draft.shape[-2:] != (68, 120) or draft.shape[2] < 1:
            raise ValueError('Require [1,24,T,68,120]')
        outputs = []
        for i in range(draft.shape[2]):
            clean = draft[0, :, i:i+1].float().to(self.runtime.device)
            noise = position_noise(clean.shape, self.sample_id, self.runtime.next_frame, self.seed).to(clean.device)
            x = (1-SIGMAS[0])*clean+SIGMAS[0]*noise
            for sigma, following in zip(SIGMAS, SIGMAS[1:]):
                v = self.runtime.stream_velocity(x, clean, sigma)
                predicted = self.head(x, v, clean, sigma)
                # Explicit final assignment preserves exact initialization identity.
                x = predicted if following == 0 else (following/sigma)*x+(1-following/sigma)*predicted
                self.model_forwards += 1
                if not torch.isfinite(x).all():
                    raise ValueError('Nonfinite anchored output')
            self.runtime.stream_velocity(x, clean, 0, commit=True)
            self.model_forwards += 1
            outputs.append(x.cpu())
        return torch.cat(outputs, 1)[None]

    def audit(self):
        cache = self.runtime.cache or []
        return {'next_latent': self.runtime.next_frame, 'model_forwards': self.model_forwards,
                'denoise_steps_per_latent': 3, 'clean_cache_write_forwards_per_latent': 1,
                'lookahead_latents': 0, 'recomputed_history_latents': 0,
                'cache_bytes': sum(v.k.numel()*v.k.element_size()+v.v.numel()*v.v.element_size() for v in cache),
                'max_cached_video_frames': max((len(set(f for f in v.frames if f >= 0)) for v in cache), default=0),
                'residual_elementwise_bound': self.head.bound, 'decoder_validated': False,
                'strict_zero_regression_guaranteed_after_training': False}
