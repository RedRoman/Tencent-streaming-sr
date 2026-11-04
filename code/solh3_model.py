"""Sol-H3 latent refiner: channel-concat LR conditioning, configurable time window.

The HR target and the 3x-upsampled LR condition live on the same [C,T,H,W] grid,
so the condition enters by concatenating channels at the stem: 24 noisy HR + 24
LR = 48. `conv_in` keeps its pretrained weights on the first 24 input channels
and starts at zero on the LR half, so an untrained refiner reproduces the
original upscaler exactly and every later change is attributable to the LR path.

Temporal receptive field is carried solely by the depthwise `dwconv` taps in the
ConvNeXt blocks. `time_taps=1` makes the model strictly per-latent-frame
(image level); 5 is the pretrained width. Nothing else differs between arms.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    """GroupNorm-SiLU-Conv3d with additive timestep modulation."""

    def __init__(self, ch: int, emb_ch: int):
        super().__init__()
        self.in_layers = nn.Sequential(nn.GroupNorm(32, ch), nn.SiLU(), nn.Conv3d(ch, ch, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_ch, ch * 2))
        self.out_norm = nn.GroupNorm(32, ch)
        # Index 2 holds the conv, matching the pretrained upscaler's key layout
        # (`out_layers.2.weight`); index 0 is a no-op placeholder.
        self.out_layers = nn.Sequential(nn.Identity(), nn.SiLU(), nn.Conv3d(ch, ch, 3, padding=1))

    def forward(self, x, emb):
        h = self.in_layers(x)
        scale, shift = self.emb_layers(emb)[:, :, None, None, None].chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        return x + self.out_layers(h)


class ConvNeXtBlock(nn.Module):
    """Depthwise temporal conv + pointwise mix. `time_taps` sets the time window."""

    def __init__(self, ch: int, time_taps: int):
        super().__init__()
        self.time_taps = time_taps
        self.dwconv = nn.Conv3d(ch, ch, (time_taps, 1, 1), padding=(time_taps // 2, 0, 0), groups=ch)
        self.norm = nn.GroupNorm(32, ch)
        self.pwconv = nn.Conv3d(ch, ch, 1)

    def forward(self, x, emb=None):
        return x + self.pwconv(F.silu(self.norm(self.dwconv(x))))


class LatentUpsampleHead(nn.Module):
    """Map an LR H3 latent onto the HR latent grid.

    H3's VAE compresses space 16x with a deep non-linear encoder, so an LR latent
    is NOT a blurred HR latent: trilinear-upsampling one correlates only ~0.67
    with the true HR latent (1.9 dB PSNR). The two resolutions are different
    coordinate systems for the same content, and that mismatch -- not model
    capacity -- is what capped the earlier channel-concat runs.

    This head learns the change of coordinates. Output starts at zero and is
    added to the trilinear upsample, so at step 0 the module reproduces the old
    behaviour exactly and any improvement is attributable to what it learns.
    """

    def __init__(self, latent_ch=24, width=128, scale=3, blocks=4):
        super().__init__()
        self.scale = scale
        self.proj_in = nn.Conv3d(latent_ch, width, (1, 3, 3), padding=(0, 1, 1))
        self.body = nn.ModuleList()
        for _ in range(blocks):
            self.body.append(nn.Sequential(
                nn.GroupNorm(32, width), nn.SiLU(),
                nn.Conv3d(width, width, (3, 3, 3), padding=(1, 1, 1))))
        # PixelShuffle works on 4D, so the spatial expansion is applied per frame.
        self.to_shuffle = nn.Conv2d(width, latent_ch * scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)
        nn.init.zeros_(self.to_shuffle.weight)
        nn.init.zeros_(self.to_shuffle.bias)

    def forward(self, lr):
        """lr: [B,24,T,h,w] -> [B,24,T,3h,3w]."""
        base = F.interpolate(
            lr.permute(0, 2, 1, 3, 4).flatten(0, 1),
            scale_factor=self.scale, mode="bilinear", align_corners=False)
        h = self.proj_in(lr)
        for blk in self.body:
            h = h + blk(h)
        b, c, t = h.shape[0], h.shape[1], h.shape[2]
        h = h.permute(0, 2, 1, 3, 4).flatten(0, 1)        # [B*T,width,h,w]
        delta = self.shuffle(self.to_shuffle(h))           # [B*T,24,3h,3w]
        out = (base + delta).unflatten(0, (b, t)).permute(0, 2, 1, 3, 4)
        return out.contiguous()


class SolH3Refiner(nn.Module):
    """LR-conditioned HR latent refiner.

    Args:
        latent_ch: H3 latent channels (24).
        width: hidden width (512 matches the pretrained upscaler).
        time_taps: temporal kernel of every ConvNeXt block. 1 = image level.
        cond_mode: "concat" stacks LR on the input channels; "none" ablates it.
    """

    def __init__(self, latent_ch=24, width=512, emb_ch=64, time_taps=5,
                 cond_mode="concat", num_blocks=18, learned_upsample=False):
        super().__init__()
        self.latent_ch = latent_ch
        self.cond_mode = cond_mode
        self.time_taps = time_taps
        self.grad_checkpointing = False
        # When set, the LR condition is re-gridded by a trained head instead of
        # plain interpolation; see LatentUpsampleHead for why that matters.
        self.up_head = LatentUpsampleHead(latent_ch) if learned_upsample else None

        in_ch = latent_ch * 2 if cond_mode == "concat" else latent_ch
        self.conv_in = nn.Conv3d(in_ch, width, 3, padding=1)
        self.embed = nn.Sequential(nn.Linear(1, emb_ch), nn.SiLU(), nn.Linear(emb_ch, emb_ch))

        def stack():
            blocks = nn.ModuleList()
            for i in range(num_blocks):
                # Match the pretrained layout: ConvNeXt at 1,4,7,10,13,16.
                if i % 3 == 1:
                    blocks.append(ConvNeXtBlock(width, time_taps))
                else:
                    blocks.append(ResBlock(width, emb_ch))
            return blocks

        self.in_blocks = stack()
        self.out_blocks = stack()
        self.norm_out = nn.GroupNorm(32, width)
        self.conv_out = nn.Conv3d(width, latent_ch, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, hr_noisy, lr_cond, sigma):
        """hr_noisy/lr_cond: [B,24,T,H,W] on the same grid. sigma: [B]."""
        if self.up_head is not None:
            lr_cond = self.up_head(lr_cond)
        if self.cond_mode == "concat":
            x = torch.cat([hr_noisy, lr_cond], dim=1)
        else:
            x = hr_noisy
        emb = self.embed(sigma.view(-1, 1).to(x.dtype))
        h = self.conv_in(x)
        for blk in list(self.in_blocks) + list(self.out_blocks):
            if self.grad_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(blk, h, emb, use_reentrant=False)
            else:
                h = blk(h, emb)
        return self.conv_out(F.silu(self.norm_out(h)))

    @torch.no_grad()
    def load_pretrained_upscaler(self, path):
        """Seed from the 345M H3 latent upscaler.

        Shapes line up except at `conv_in`, where the pretrained 24 input
        channels are copied into the HR half and the LR half starts at zero.
        Returns (loaded, skipped) key counts.
        """
        from safetensors.torch import load_file
        src = load_file(path)
        own = self.state_dict()
        loaded = skipped = 0
        for k, v in src.items():
            if k not in own:
                skipped += 1
                continue
            if own[k].shape == v.shape:
                own[k].copy_(v.to(own[k].dtype))
                loaded += 1
            elif k == "conv_in.weight" and self.cond_mode == "concat":
                own[k].zero_()
                own[k][:, : self.latent_ch].copy_(v.to(own[k].dtype))
                loaded += 1
            elif k.endswith("dwconv.weight") and own[k].shape[2] != v.shape[2]:
                # Narrower time window: keep the centre tap(s) of the pretrained
                # kernel so a W=1 arm still starts from trained features.
                c = v.shape[2] // 2
                w = own[k].shape[2]
                own[k].copy_(v[:, :, c - w // 2 : c - w // 2 + w].to(own[k].dtype))
                loaded += 1
            else:
                skipped += 1
        self.load_state_dict(own)
        return loaded, skipped
