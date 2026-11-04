"""Adversarial pieces for the pixel refiner.

Reconstruction losses cannot produce sharp texture: 3x downsampling destroys 86%
of GT's high-frequency energy (measured), so the detail is not recoverable and
must be *generated*. L1 answers that uncertainty with the conditional mean --
blur. A discriminator instead rewards output that looks like real texture, which
is why GAN-based SR looks sharp. PSNR drops in exchange; that trade is the point.

The discriminator is a small spectral-normalized PatchGAN so it stays cheap and
stable, and it never runs at inference -- streaming latency is unaffected.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def _c(i, o, s):
    return spectral_norm(nn.Conv2d(i, o, 4, stride=s, padding=1))


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN. Judges local texture rather than global layout, which is
    what matters for super-resolution."""

    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            _c(3, ch, 2), nn.LeakyReLU(0.2, True),
            _c(ch, ch * 2, 2), nn.LeakyReLU(0.2, True),
            _c(ch * 2, ch * 4, 2), nn.LeakyReLU(0.2, True),
            _c(ch * 4, ch * 4, 1), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ch * 4, 1, 4, stride=1, padding=1)),
        )

    def forward(self, x):
        return self.net(x)


def d_loss(disc, real, fake):
    """Hinge loss: stabler than BCE and standard for SR GANs."""
    r = disc(real)
    f = disc(fake.detach())
    return (F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean()) * 0.5


def g_loss(disc, fake):
    return -disc(fake).mean()


class TemporalPatchDiscriminator(PatchDiscriminator):
    """Same PatchGAN, but it judges `window` consecutive frames at once.

    The spatial discriminator did raise high-frequency energy to 0.84x GT, yet
    LPIPS got worse and flicker rose monotonically (0.0120 -> 0.0146): it was
    rewarding per-frame noise that does not agree between neighbouring frames.
    Stacking the frames on the channel axis lets the discriminator see that
    disagreement, so inconsistent texture is no longer a winning strategy.

    The generator stays memoryless -- only the discriminator sees a window, and
    it is discarded at inference, so streaming latency is untouched.
    """

    def __init__(self, window, ch=64):
        super().__init__(ch)
        self.window = window
        first = self.net[0]
        self.net[0] = spectral_norm(nn.Conv2d(3 * window, ch, 4, stride=2, padding=1))
        del first

    def forward(self, x):
        """x: [B*T,3,H,W] with each clip's T frames contiguous -> [B,T*3,H,W]."""
        if x.dim() == 4 and x.shape[1] == 3:
            x = x.unflatten(0, (x.shape[0] // self.window, self.window)).flatten(1, 2)
        return self.net(x)
