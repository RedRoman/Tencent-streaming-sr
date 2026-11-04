"""H3 latent -> Wan HR latent in one learned module.

Replaces the mapper + frozen UltraFlash upsampler chain. That chain was the
bottleneck: optimizing the upsampler's input directly could only reach 13.3 dB
against the true Wan HR latent, so no mapper could have fixed it.

Geometry: [24, T, 16, 28] -> [16, T, 96, 168], i.e. 6x spatial, channels 24->16,
time unchanged. 6 = 2 x 3, done as two PixelShuffle stages so the body stays at
low resolution and only the last stage touches the full grid.

Temporal handling is causal-friendly: 3x3x3 convs see one frame each side, and
the model is trained on whole 7-latent windows matching the Wan main/tail split.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
    def __init__(self, ch, temporal=True):
        super().__init__()
        k = (3, 3, 3) if temporal else (1, 3, 3)
        p = (1, 1, 1) if temporal else (0, 1, 1)
        self.net = nn.Sequential(
            nn.GroupNorm(32, ch), nn.SiLU(), nn.Conv3d(ch, ch, k, padding=p),
            nn.GroupNorm(32, ch), nn.SiLU(), nn.Conv3d(ch, ch, k, padding=p))

    def forward(self, x):
        return x + self.net(x)


class PixelShuffle3DSpatial(nn.Module):
    """PixelShuffle over H,W only, applied per frame."""

    def __init__(self, ch_in, ch_out, scale):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv3d(ch_in, ch_out * scale * scale, (1, 3, 3), padding=(0, 1, 1))

    def forward(self, x):
        b, c, t, h, w = x.shape
        y = self.conv(x)
        y = y.permute(0, 2, 1, 3, 4).flatten(0, 1)
        y = F.pixel_shuffle(y, self.scale)
        return y.unflatten(0, (b, t)).permute(0, 2, 1, 3, 4).contiguous()


class H3ToWanHR(nn.Module):
    def __init__(self, in_ch=24, out_ch=16, width=256, blocks_low=6, blocks_mid=4, blocks_hi=2):
        super().__init__()
        self.stem = nn.Conv3d(in_ch, width, (1, 3, 3), padding=(0, 1, 1))
        self.low = nn.ModuleList(ResBlock3D(width, i % 2 == 1) for i in range(blocks_low))
        self.up2 = PixelShuffle3DSpatial(width, width, 2)
        self.mid = nn.ModuleList(ResBlock3D(width, i % 2 == 1) for i in range(blocks_mid))
        self.up3 = PixelShuffle3DSpatial(width, width // 2, 3)
        self.hi = nn.ModuleList(ResBlock3D(width // 2, False) for _ in range(blocks_hi))
        self.head = nn.Conv3d(width // 2, out_ch, (1, 3, 3), padding=(0, 1, 1))
        # Start from the trilinear upsample of a channel-projected input so step 0
        # is a sane baseline rather than noise.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.skip = nn.Conv3d(in_ch, out_ch, 1)
        self.grad_checkpointing = False

    def _run(self, blocks, x):
        for b in blocks:
            if self.grad_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(b, x, use_reentrant=False)
            else:
                x = b(x)
        return x

    def forward(self, x):
        b, c, t, h, w = x.shape
        base = self.skip(x)
        base = F.interpolate(base.permute(0, 2, 1, 3, 4).flatten(0, 1), scale_factor=6,
                             mode="bilinear", align_corners=False)
        base = base.unflatten(0, (b, t)).permute(0, 2, 1, 3, 4).contiguous()
        y = self._run(self.low, self.stem(x))
        y = self._run(self.mid, self.up2(y))
        y = self._run(self.hi, self.up3(y))
        return base + self.head(y)
