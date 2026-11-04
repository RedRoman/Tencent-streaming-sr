"""Latent-domain SR for Sol-H3: LR latent + HR keyframe latent -> HR latent.

Both of this model's starting points were measured first (latent_probe.py),
decoding each through the same distilled student and scoring the pixels:

    trilinear 3x of the LR latent      LPIPS 0.5501
    HR keyframe latent, copied         LPIPS 0.2550
    true HR latent (the ceiling)       LPIPS 0.1449
    the pixel pipeline we ship today   LPIPS 0.1910

Two design decisions follow directly from those numbers.

`base = keyframe`, not interpolation. The pixel refiner could zero-init onto
bicubic and start at a sane 0.3178; the latent equivalent, trilinear, is 0.5501
because VAE latents are not smooth and interpolation destroys them rather than
blurring them. The keyframe is the only anchor in this domain worth standing on,
so the tail is zero-initialised onto IT -- step 0 is exactly "copy the keyframe".

The LR latent enters per frame. It is the only input carrying what is true at
time t rather than at the keyframe, which is why the keyframe alone decays from
0.2550 to 0.3199 over six chunks.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(x), inplace=True))


class TemporalMix(nn.Module):
    """Depthwise conv across the frame axis; the only path carrying time."""

    def __init__(self, ch, taps=3):
        super().__init__()
        self.taps = taps
        self.dw = nn.Conv3d(ch, ch, (taps, 1, 1), padding=(taps // 2, 0, 0), groups=ch)
        nn.init.zeros_(self.dw.weight); nn.init.zeros_(self.dw.bias)

    def forward(self, x, t):
        if self.taps == 1:
            return x
        bt, c, h, w = x.shape
        y = x.unflatten(0, (bt // t, t)).permute(0, 2, 1, 3, 4)
        return x + self.dw(y).permute(0, 2, 1, 3, 4).flatten(0, 1)


class LatentSR(nn.Module):
    def __init__(self, width=256, lr_blocks=8, hr_blocks=6, lat_ch=24, scale=3,
                 taps=3):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(lat_ch, width, 3, padding=1)
        self.lr_body = nn.ModuleList(ResBlock(width) for _ in range(lr_blocks))
        self.tmix = TemporalMix(width, taps)
        self.up = nn.Conv2d(width, width * scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)
        # Keyframe branch lives at the HR latent grid, where the keyframe is
        # natively sized -- no resampling of a signal that resampling destroys.
        self.kenc = nn.Sequential(nn.Conv2d(lat_ch, width, 3, padding=1),
                                  nn.ReLU(inplace=True), ResBlock(width))
        self.kdist = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, width * 2))
        self.kfuse = nn.Conv2d(width * 2, width, 1)
        self.hr_body = nn.ModuleList(ResBlock(width) for _ in range(hr_blocks))
        self.tail = nn.Conv2d(width, lat_ch, 3, padding=1)
        for m in (self.kfuse, self.tail):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        nn.init.zeros_(self.kdist[-1].weight); nn.init.zeros_(self.kdist[-1].bias)

    def encode_key(self, kf):
        """kf: [B,24,H,W] on the HR latent grid. Once per keyframe interval."""
        return self.kenc(kf)

    def forward(self, lr, frames, kfeat, kf_latent, dist):
        """lr: [B*T,24,h,w]; kfeat/kf_latent: [B,...]; dist: [B,1] frames back."""
        h = self.head(lr)
        for b in self.lr_body:
            h = b(h)
        h = self.tmix(h, frames)
        h = self.shuffle(self.up(h))                      # [B*T,W,3h,3w]
        g = self.kdist(dist)
        sc, sh = g.chunk(2, dim=1)
        r = kfeat * (1 + sc[:, :, None, None]) + sh[:, :, None, None]
        rep = h.shape[0] // r.shape[0]
        if rep > 1:
            r = r.repeat_interleave(rep, 0)
        h = h + self.kfuse(torch.cat([h, r], dim=1))
        for b in self.hr_body:
            h = b(h)
        base = kf_latent
        if base.shape[0] != h.shape[0]:
            base = base.repeat_interleave(h.shape[0] // base.shape[0], 0)
        return base + self.tail(h)
