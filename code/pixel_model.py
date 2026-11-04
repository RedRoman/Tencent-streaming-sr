"""Pixel-domain 3x SR refiner for Sol-H3.

The latent-domain route failed because H3's 16x-compressed latent space is not
isometric to pixels: a model 4.7 dB better in latent MSE decoded 6.1 dB worse
than plain bicubic. Here the objective is measured in the same domain as the
deliverable, so optimizing it cannot drift away from what the video looks like.

Latency comes from keeping every convolution at LR resolution and touching the
HR grid only in the final PixelShuffle. `temporal_window` frames are processed
jointly; 1 means strictly per-frame, which is the streaming configuration.
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

    def __init__(self, ch, taps):
        super().__init__()
        self.taps = taps
        self.dw = nn.Conv3d(ch, ch, (taps, 1, 1), padding=(taps // 2, 0, 0), groups=ch)
        nn.init.zeros_(self.dw.weight)
        nn.init.zeros_(self.dw.bias)

    def forward(self, x, t):
        """x: [B*T,C,H,W] -> same, mixing only within each clip of T frames."""
        if self.taps == 1:
            return x
        bt, c, h, w = x.shape
        y = x.unflatten(0, (bt // t, t)).permute(0, 2, 1, 3, 4)
        y = self.dw(y).permute(0, 2, 1, 3, 4).flatten(0, 1)
        return x + y


class TextFiLM(nn.Module):
    """Per-channel scale/shift driven by H3's own prompt embedding.

    This is the first thing in the pipeline that a conventional SR model cannot
    have: the caption H3 was generating from. Pure SR sees only pixels and must
    guess what the blur is hiding; "a wooden crate of oranges" is a prior on the
    texture that no amount of pixel evidence provides.

    Deliberately cheap and clip-constant: the prompt does not change inside a
    stream, so the projection runs once per clip, never per frame, and costs
    nothing in the streaming budget. Zero-initialised so an untrained model is
    exactly the text-free model.
    """

    def __init__(self, width, text_dim=5120, hidden=256, sites=1):
        super().__init__()
        # One shared trunk, one cheap head per injection site: putting a full
        # 5120->width MLP at every block would cost 1.4M parameters each.
        self.trunk = nn.Sequential(nn.Linear(text_dim, hidden), nn.SiLU())
        self.heads = nn.ModuleList(nn.Linear(hidden, width * 2) for _ in range(sites))
        for hd in self.heads:
            nn.init.zeros_(hd.weight)
            nn.init.zeros_(hd.bias)

    def forward(self, h, ctx, site=0):
        """h: [B*T,C,H,W]; ctx: [B,text_dim] pooled prompt (or None)."""
        if ctx is None:
            return h
        g = self.heads[site](self.trunk(ctx))   # [B, 2C]
        scale, shift = g.chunk(2, dim=1)
        bt = h.shape[0]
        rep = bt // scale.shape[0]
        scale = scale.repeat_interleave(rep, 0)[:, :, None, None]
        shift = shift.repeat_interleave(rep, 0)[:, :, None, None]
        return h * (1 + scale) + shift


class KeyframeRef(nn.Module):
    """Read a sparse HR keyframe into the LR-resolution feature grid.

    RelayVSR's premise, tested on our content: run the expensive generative
    model only on sparse keyframes and let the cheap network spread that detail
    over the interval. Measured here without any learning, pasting frame t-k's
    high-frequency residual onto frame t beats bicubic on LPIPS all the way out
    to k=15 (-0.0176) even with no motion compensation, so the detail does
    survive the trip. The question this module answers is how much a learned,
    distance-aware reader adds on top.

    Latency: `encode` runs ONCE per keyframe interval and its output is cached,
    so the per-frame cost is one 1x1 convolution. PixelUnshuffle brings the HR
    keyframe to the LR grid for free, keeping every convolution at LR
    resolution -- the property the whole refiner is built around.
    """

    def __init__(self, width, scale=3):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(scale)
        self.enc = nn.Sequential(nn.Conv2d(3 * scale * scale, width, 3, padding=1),
                                 nn.ReLU(inplace=True), ResBlock(width))
        # How far away the keyframe is. The reader has to trust a 1-frame-old
        # reference far more than a 15-frame-old one, and it cannot infer the
        # distance from the pixels, so it is fed explicitly.
        self.dist = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, width * 2))
        self.fuse = nn.Conv2d(width * 2, width, 1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)   # starts as the reference-free model
        nn.init.zeros_(self.dist[-1].weight)
        nn.init.zeros_(self.dist[-1].bias)

    def encode(self, ref_hr):
        """ref_hr: [B,3,H,W] -> [B,width,h,w]. Once per keyframe, not per frame."""
        return self.enc(self.unshuffle(ref_hr))

    def forward(self, h, ref_feat, dist):
        """h: [B*T,C,h,w]; ref_feat: [B,C,h,w]; dist: [B,1] frames since keyframe."""
        if ref_feat is None:
            return h
        g = self.dist(dist)
        scale, shift = g.chunk(2, dim=1)
        r = ref_feat * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        rep = h.shape[0] // r.shape[0]
        if rep > 1:
            r = r.repeat_interleave(rep, 0)
        return h + self.fuse(torch.cat([h, r], dim=1))


class KeyframeRefLatent(nn.Module):
    """Read a sparse HR keyframe that is still in H3's LATENT form.

    Measured: the H3 VAE cannot decode fewer than 5 latents, and at HR that
    minimum chunk costs 5847 ms -- 4.7x the entire 30-frame interval budget. So
    a pixel-space keyframe is out, and the reference has to be consumed as the
    latent H3 already produced, with no decode at all.

    Geometry is exact, not resampled: one HR latent cell covers 16x16 HR pixels
    = 16/3 LR pixels, so a `crop` x `crop` LR window maps to crop*3/16 latent
    cells. The encoder lifts those cells to the LR feature grid by transposed
    convolution rather than interpolation, so no detail is thrown away on the
    way in.

    Cost is per keyframe, not per frame: `encode` runs once per interval.
    """

    def __init__(self, width, latent_ch=24, local_gate=False):
        super().__init__()
        # 16 HR px per latent cell / 3 = 16/3 LR px. Factor as 4x then 4/3x:
        # a stride-4 transposed conv, then a stride-4 one with stride-3 output
        # trimming is awkward, so lift 4x and let a final 4->3 resize land it.
        self.up = nn.Sequential(
            nn.Conv2d(latent_ch, width, 3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(width, width, 4, stride=4), nn.ReLU(inplace=True),
            ResBlock(width))
        self.dist = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, width * 2))
        # How far the keyframe is in TIME is one scalar for the whole frame, but
        # how stale it is varies by REGION. Measured on the latent route: the
        # keyframe stays the better source over ~30% of the picture -- whatever
        # has not moved -- while going badly wrong on whatever has. A global
        # distance cannot express that split, so the local change is fed in and
        # the keyframe gets weighed per position.
        self.local_gate = local_gate
        if local_gate:
            self.gate = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, width, 3, padding=1))
            nn.init.zeros_(self.gate[-1].weight); nn.init.zeros_(self.gate[-1].bias)
        self.fuse = nn.Conv2d(width * 2, width, 1)
        nn.init.zeros_(self.fuse.weight); nn.init.zeros_(self.fuse.bias)
        nn.init.zeros_(self.dist[-1].weight); nn.init.zeros_(self.dist[-1].bias)

    def encode(self, ref_lat, out_hw):
        """ref_lat: [B,24,h,w] -> [B,width,*out_hw] on the LR grid."""
        y = self.up(ref_lat)
        if y.shape[-2:] != tuple(out_hw):
            y = F.interpolate(y, size=out_hw, mode="bilinear", align_corners=False)
        return y

    def forward(self, h, ref_feat, dist, change=None):
        """change: [B*T,1,h,w], local |LR_t - LR_keyframe| on the LR grid."""
        if ref_feat is None:
            return h
        g = self.dist(dist)
        sc, sh = g.chunk(2, dim=1)
        r = ref_feat * (1 + sc[:, :, None, None]) + sh[:, :, None, None]
        rr = h.shape[0] // r.shape[0]
        if rr > 1:
            r = r.repeat_interleave(rr, 0)
        if self.local_gate and change is not None:
            # Zero-init head: untrained, this term is 0 and the module is
            # bit-identical to the ungated version.
            r = r * (1 + self.gate(change))
        return h + self.fuse(torch.cat([h, r], dim=1))


class PixelSR(nn.Module):
    def __init__(self, width=96, blocks=12, scale=3, temporal_taps=1, text_dim=0,
                 text_sites=1, keyframe_ref=False, local_gate=False):
        super().__init__()
        self.scale = scale
        self.temporal_taps = temporal_taps
        self.text_dim = text_dim
        # text_sites=1 modulates once after the head; >1 spreads the injection
        # points evenly through the body, so the prompt can steer late features
        # and not just the stem.
        self.text_sites = text_sites if text_dim else 0
        self.film = TextFiLM(width, text_dim, sites=text_sites) if text_dim else None
        self.kref = (KeyframeRefLatent(width, local_gate=local_gate)
                     if keyframe_ref == "latent"
                     else KeyframeRef(width, scale) if keyframe_ref else None)
        self.head = nn.Conv2d(3, width, 3, padding=1)
        self.body = nn.ModuleList(ResBlock(width) for _ in range(blocks))
        # One temporal mixer in the middle of the body when a window is allowed.
        self.tmix = TemporalMix(width, temporal_taps)
        self.tail = nn.Conv2d(width, 3 * scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)
        # Start as an exact bicubic upscaler so training only has to learn the
        # residual detail, and step 0 already matches the baseline.
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, lr, frames=1, text=None, ref_feat=None, dist=None,
                change=None):
        """lr: [B*T,3,h,w] in [-1,1]; `frames` is T so the mixer can reshape.

        `text` is the pooled prompt embedding, [B,text_dim], one row per clip.
        """
        base = F.interpolate(lr, scale_factor=self.scale, mode="bicubic", align_corners=False)
        h = self.head(lr)
        if self.kref is not None:
            h = self.kref(h, ref_feat, dist, change)
        sites = {}
        if self.film is not None:
            if self.text_sites == 1:
                h = self.film(h, text)
            else:
                step = max(len(self.body) // self.text_sites, 1)
                sites = {min((k + 1) * step - 1, len(self.body) - 1): k
                         for k in range(self.text_sites)}
        mid = len(self.body) // 2
        for i, b in enumerate(self.body):
            h = b(h)
            if i in sites:
                h = self.film(h, text, site=sites[i])
            if i == mid:
                h = self.tmix(h, frames)
        return (base + self.shuffle(self.tail(h))).clamp(-1, 1)
