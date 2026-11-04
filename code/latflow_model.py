"""Flow matching in H3's HR latent space: the manifold-alignment stage.

The deterministic latent regressor converged at structure PSNR 28.50 against
34.06 for the pixel pipeline and 35.03 for the true latent. That gap has a name
in the literature -- manifold deviation (LSRNA, CVPR 2025): upsampling pushes
the latent off the distribution the VAE decoder was trained on, and an L1/LPIPS
regressor cannot fix it because the conditional mean it converges to is itself
off-manifold. Regression is the disease, not the cure.

So this stage is generative. It transports the regressor's estimate onto the HR
latent manifold with a rectified-flow velocity field, conditioned on everything
that carries truth about this frame:

    z_t   the point being transported, between the stage-1 estimate and the HR latent
    lr_up the LR latent for THIS frame, noise-augmented (SimpleGVR lists noise
          augmentation of the LR condition as one of two factors that decide
          whether cascaded SR works)
    kf    the HR keyframe latent, the only on-manifold HR signal available
    t, d  flow time and frames-since-keyframe

Source is the stage-1 estimate rather than Gaussian noise: this project already
measured a bridge objective beating a noise objective by 7.4 dB in this latent
space, and the bridge's starting point is far better now than it was then.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with FiLM from the (t, distance) embedding."""

    def __init__(self, ch, emb):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.emb = nn.Linear(emb, ch * 2)
        nn.init.zeros_(self.emb.weight); nn.init.zeros_(self.emb.bias)

    def forward(self, x, e):
        s, b = self.emb(e).chunk(2, dim=1)
        rep = x.shape[0] // s.shape[0]
        if rep > 1:
            s = s.repeat_interleave(rep, 0); b = b.repeat_interleave(rep, 0)
        h = F.relu(self.c1(x) * (1 + s[:, :, None, None]) + b[:, :, None, None], True)
        return x + self.c2(h)


def sinusoidal(x, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / half)
    a = x[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], dim=-1)


class LatentFlow(nn.Module):
    def __init__(self, width=192, blocks=10, lat_ch=24, emb=256):
        super().__init__()
        self.emb_dim = emb
        self.tproj = nn.Sequential(nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        # z_t, the LR latent lifted to this grid, the keyframe, AND the stage-1
        # estimate as its own channel group. The first version used the stage-1
        # output only as the ODE's starting point, which left this stage with
        # exactly the conditioning stage 1 already had -- no new information to
        # exploit, so the best it could do was re-derive the same conditional
        # mean, worse. Passing it explicitly lets this stage correct an answer
        # rather than recompute one.
        self.inp = nn.Conv2d(lat_ch * 4, width, 3, padding=1)
        self.body = nn.ModuleList(ResBlock(width, emb) for _ in range(blocks))
        self.out = nn.Conv2d(width, lat_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, z_t, lr_up, kf, t, dist, z1=None):
        """z_t/lr_up/z1: [B*T,24,H,W]; kf: [B,24,H,W]; t: [B*T]; dist: [B,1]."""
        e = self.tproj(sinusoidal(t, self.emb_dim))
        d = sinusoidal(dist.squeeze(1), self.emb_dim)
        rep = z_t.shape[0] // d.shape[0]
        if rep > 1:
            d = d.repeat_interleave(rep, 0)
        e = e + d
        k = kf
        if k.shape[0] != z_t.shape[0]:
            k = k.repeat_interleave(z_t.shape[0] // k.shape[0], 0)
        if z1 is None:
            z1 = torch.zeros_like(z_t)
        h = self.inp(torch.cat([z_t, lr_up, k, z1], dim=1))
        for b in self.body:
            h = b(h, e)
        # Zero-init head: at step 0 the velocity is 0, so the ODE is the identity
        # and the output is exactly the stage-1 estimate. Any gain is real.
        return self.out(h)

    @torch.no_grad()
    def sample(self, z0, lr_up, kf, dist, nfe=2, z1=None, sigma=0.0, generator=None):
        """Integrate from the stage-1 estimate z0 to the HR latent manifold.

        z_t = (1-t)*z_hr + t*z_src, so dz/dt = z_src - z_hr and reaching t=0
        from t=1 means stepping DOWN in t: z <- z - v/nfe. The first version
        added the velocity instead, which walked the estimate away from the
        target -- it showed up as a gain that was slightly negative, perfectly
        flat from step 1000 to 20000, and identical for a 7.9M and a 28.5M
        model, i.e. a constant artefact rather than a training outcome.
        """
        # Perturb the start. Without this the whole chain -- deterministic
        # regressor, deterministic start, deterministic ODE -- is still a single
        # deterministic map computing a conditional mean, which is the thing
        # flow matching exists to avoid and the reason the first attempt lost to
        # its own starting point. This is LSRNA's noise addition, placed where
        # the ODE can actually transport it back onto the manifold.
        z = z0
        if sigma > 0:
            n = torch.randn(z.shape, device=z.device, dtype=z.dtype,
                            generator=generator)
            z = z + sigma * n
        for i in range(nfe):
            t = torch.full((z.shape[0],), 1.0 - i / nfe, device=z.device)
            v = self(z, lr_up, kf, t, dist, z1)
            z = z - v / nfe
        return z
