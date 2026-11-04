"""Check whether the low/ and high/ latent sets are the same content at two
resolutions (a usable SR pair) or two unrelated generations.

Latent grids are [24,37,16,28] and [24,37,48,84] -- exactly 3x spatially, same
temporal length -- so average-pooling HR by 3x lands on the LR grid. A true pair
should correlate far above a deliberately mismatched pairing.
"""
import os
import torch
import torch.nn.functional as F

LOW = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/low/video"
HIGH = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/high/video"


def load(d, i):
    return torch.load(f"{d}/{i}.pt.part", map_location="cpu", weights_only=False).float()


def corr(a, b):
    a, b = a.flatten(), b.flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-9))


def to_lr_grid(hr):
    return F.avg_pool3d(hr.unsqueeze(0), kernel_size=(1, 3, 3)).squeeze(0)


ids = sorted(f[:-8] for f in os.listdir(LOW) if f.endswith(".pt.part"))[:10]
lo = {i: load(LOW, i) for i in ids}
hi = {i: load(HIGH, i) for i in ids}

print(f"{'id':<22}{'corr(LR, pooled HR)':>22}{'corr vs next id':>20}")
matched, shuffled = [], []
for n, i in enumerate(ids):
    other = ids[(n + 1) % len(ids)]
    m = corr(lo[i], to_lr_grid(hi[i]))
    s = corr(lo[i], to_lr_grid(hi[other]))
    matched.append(m)
    shuffled.append(s)
    print(f"{i:<22}{m:>22.4f}{s:>20.4f}")

print()
print(f"mean matched  {sum(matched)/len(matched):.4f}")
print(f"mean shuffled {sum(shuffled)/len(shuffled):.4f}")
print()
print("per-channel corr, first pair:")
i = ids[0]
d = to_lr_grid(hi[i])
cs = [corr(lo[i][c], d[c]) for c in range(24)]
print("  min %.3f  median %.3f  max %.3f" % (
    min(cs), sorted(cs)[12], max(cs)))

os._exit(0)
