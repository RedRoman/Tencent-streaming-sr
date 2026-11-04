"""Is trilinear(LR) actually aligned with HR in the latent domain?

A 1.69 dB PSNR between an upsampled LR latent and its HR latent is suspiciously
low for content that is visually identical. Check correlation, per-channel
scale, and whether a simple affine fit closes the gap -- i.e. whether the two
latent sets share a coordinate system at all.
"""
import os, random
import torch, torch.nn.functional as F
import torch_npu  # noqa: F401
from solh3_train import Pairs, upsample_lr, HIGH, VAL_IDS

dev = "npu:0"
ids = sorted(f[:-8] for f in os.listdir(HIGH) if f.endswith(".pt.part"))
random.Random(1234).shuffle(ids)

for src in ("h3", "degraded"):
    ds = Pairs(ids[:VAL_IDS], src)
    lo, hi = ds.get(0)
    lo = lo.unsqueeze(0).to(dev); hi = hi.unsqueeze(0).to(dev)
    up = upsample_lr(lo)
    a, b = up.flatten(), hi.flatten()
    corr = float(F.cosine_similarity((a-a.mean()).unsqueeze(0), (b-b.mean()).unsqueeze(0)))
    psnr = float(-10*torch.log10(F.mse_loss(up, hi)+1e-12))
    # best global affine
    A = torch.stack([a, torch.ones_like(a)], 1)
    sol = torch.linalg.lstsq(A.float(), b.float().unsqueeze(1)).solution.squeeze()
    fit = a*sol[0] + sol[1]
    psnr_fit = float(-10*torch.log10(F.mse_loss(fit, b)+1e-12))
    print("%-9s PSNR %6.3f  corr %6.3f  affine(scale %.3f bias %+.3f) -> %6.3f dB"
          % (src, psnr, corr, sol[0], sol[1], psnr_fit), flush=True)
    # per-channel correlation spread
    cs = []
    for c in range(24):
        x, y = up[0,c].flatten(), hi[0,c].flatten()
        cs.append(float(F.cosine_similarity((x-x.mean()).unsqueeze(0), (y-y.mean()).unsqueeze(0))))
    cs.sort()
    print("          per-channel corr: min %.3f  median %.3f  max %.3f"
          % (cs[0], cs[12], cs[-1]), flush=True)
os._exit(0)
