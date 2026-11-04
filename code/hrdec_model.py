"""A small student for the H3 HR VAE decoder.

The teacher is a 36-layer 2.42B ViT that costs 5847 ms for one 5-latent chunk
at 1344x768 -- 8.3x the entire 708 ms chunk budget. Its attention is already
windowed into 256x256 tiles (1008 identical 1797-token attentions at HR vs 72
at LR), so sparsity has at most 1.75x in it and cannot open that gap. Only a
much smaller network can.

Shape contract, measured rather than assumed (hrdec_align.py):
    latents[5k:5k+5]  [24,5,48,84]   ->   frames[17k:17k+17]  [17,3,768,1344]
17 = 5*4 - 3 is the VAE's own accounting: 4 pixel frames per latent, minus
`token_drop=3` charged once per clip. So the temporal stage expands 4x to 20
and drops 3, mirroring the teacher instead of inventing a ratio.

Cost is controlled the same way the refiner's is: every convolution runs at the
lowest resolution that can still carry the information, and resolution is bought
one PixelShuffle at a time with the channel count halving each step, which keeps
per-stage FLOPs constant instead of exploding at the top.
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


class UpStage(nn.Module):
    """2x spatial via PixelShuffle, then `blocks` residual blocks at the new size."""

    def __init__(self, cin, cout, blocks):
        super().__init__()
        self.proj = nn.Conv2d(cin, cout * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.body = nn.Sequential(*[ResBlock(cout) for _ in range(blocks)])

    def forward(self, x):
        return self.body(self.shuffle(self.proj(x)))


class HRDecoder(nn.Module):
    def __init__(self, width=256, stage_blocks=(2, 2, 2, 1), latent_ch=24,
                 n_lat=5, n_frames=17, drop=3, scale=16):
        super().__init__()
        # One PixelShuffle(2) per stage, so the stage count IS the scale. A
        # five-entry tuple silently built a 32x model whose output was twice
        # the target in each axis; assert instead of letting that pass.
        assert 2 ** len(stage_blocks) == scale, \
            f"{len(stage_blocks)} stages give {2**len(stage_blocks)}x, need {scale}x"
        self.n_lat, self.n_frames, self.drop = n_lat, n_frames, drop
        self.stem = nn.Conv3d(latent_ch, width, 3, padding=1)
        # Temporal 4x. Kernel 4 stride 4 keeps each latent's four frames
        # separable; the following 3-tap conv is what lets a frame see its
        # neighbours, which is where temporal consistency has to come from.
        self.tup = nn.ConvTranspose3d(width, width, (4, 1, 1), stride=(4, 1, 1))
        self.tmix = nn.Conv3d(width, width, (3, 1, 1), padding=(1, 0, 0))
        chans = [width >> i for i in range(len(stage_blocks) + 1)]
        self.stages = nn.ModuleList(
            UpStage(chans[i], chans[i + 1], b) for i, b in enumerate(stage_blocks))
        self.out = nn.Conv2d(chans[-1], 3, 3, padding=1)

    def forward(self, z):
        """z: [B,24,n_lat,h,w] -> [B*n_frames,3,h*16,w*16] in [-1,1]."""
        B = z.shape[0]
        x = F.relu(self.stem(z), inplace=True)
        x = self.tup(x)
        x = x + self.tmix(x)
        # Mirror the teacher: it drops `frame_pre_padding` frames off the front
        # of every decoded chunk.
        x = x[:, :, self.drop:self.drop + self.n_frames]
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)      # [B*T,C,h,w]
        for s in self.stages:
            x = s(x)
        return torch.tanh(self.out(x))
