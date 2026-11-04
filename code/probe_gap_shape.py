"""Characterize how our H3 LR differs from a clean downsample of HR.

If the difference is mostly a blur/sharpness mismatch or a shift, it can be
simulated when building training pairs. If it is high-frequency noise or
genuine content difference, it cannot -- and the ceiling stands.
"""
import json, os, random
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
PIX = ROOT / "data/pixels"


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def hf(x):
    g = x.mean(1, keepdim=True)
    return float(((g - F.avg_pool2d(g, 3, 1, 1)) ** 2).mean() / ((g ** 2).mean() + 1e-9))


def main():
    dev = "npu:0"
    ids = sorted(f[:-4] for f in os.listdir(f"{PIX}/hr") if f.endswith(".npy"))
    random.Random(1234).shuffle(ids)
    rows = []
    for vid in ids[:4]:
        hr = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")
        lr = np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")
        n = 16
        g = torch.from_numpy(np.asarray(hr[:n])).permute(0, 3, 1, 2).to(dev).float() / 127.5 - 1
        ours = torch.from_numpy(np.asarray(lr[:n])).permute(0, 3, 1, 2).to(dev).float() / 127.5 - 1

        best = None
        for mode, aa in (("bicubic_aa", True), ("bicubic_noaa", False),
                         ("area", None), ("bilinear_aa", True)):
            if mode == "area":
                c = F.interpolate(g, scale_factor=1/3, mode="area")
            else:
                m = "bilinear" if "bilinear" in mode else "bicubic"
                c = F.interpolate(g, scale_factor=1/3, mode=m, align_corners=False, antialias=aa)
            p = psnr(c.clamp(-1, 1), ours)
            if best is None or p > best[1]:
                best = (mode, p)

        clean = F.interpolate(g, scale_factor=1/3, mode="bicubic",
                              align_corners=False, antialias=True).clamp(-1, 1)
        # Is the residual broadband noise or structured (blur) difference?
        resid = ours - clean
        lowfreq_resid = F.avg_pool2d(resid, 5, 1, 2)
        rows.append(dict(clip=vid,
            best_downsample=best[0], best_psnr=round(best[1], 2),
            hf_ours=round(hf(ours), 5), hf_clean=round(hf(clean), 5),
            sharper="ours" if hf(ours) > hf(clean) else "clean",
            resid_energy=round(float((resid ** 2).mean()), 5),
            resid_lowfreq_share=round(float((lowfreq_resid ** 2).mean()
                                            / ((resid ** 2).mean() + 1e-9)), 3),
            mean_shift=round(float(resid.mean()), 4),
            std_ratio=round(float(ours.std() / clean.std()), 3)))
        print(json.dumps(rows[-1]), flush=True)
        torch.npu.empty_cache()

    a = lambda k: round(sum(r[k] for r in rows) / len(rows), 4) if isinstance(rows[0][k], (int, float)) else None
    print("\nMEAN " + json.dumps({k: a(k) for k in rows[0]
                                  if isinstance(rows[0][k], (int, float))}), flush=True)
    os._exit(0)


main()
