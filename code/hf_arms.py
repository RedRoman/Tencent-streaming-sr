"""High-frequency energy of several arms on one fixed clip set.

LPIPS says whether an output looks right; this says whether it is *generating*
detail at all. GT is 1.00x by definition; bicubic sits at 0.15x because 3x
downsampling destroyed 86% of the high-frequency energy. An arm that scores
near 1.0 here but badly on LPIPS is inventing the wrong texture; an arm near
bicubic is simply blurry.
"""
import json, os, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch_npu  # noqa: F401
from pixel_model import PixelSR
from pixel_train import PIX, to_t
import random

dev = "npu:0"
CLIPS = 4


def hf_energy(x):
    g = x.mean(1, keepdim=True)
    lo = F.avg_pool2d(g, 3, stride=1, padding=1)
    return float(((g - lo) ** 2).mean() / ((g ** 2).mean() + 1e-9))


def main():
    arms = [a.split("=") for a in sys.argv[1:]]
    ids = sorted(f[:-4] for f in os.listdir(f"{PIX}/hr") if f.endswith(".npy"))
    random.Random(1234).shuffle(ids)
    ids = ids[:CLIPS]

    gt_e, bic_e, out_e = [], [], {n: [] for n, _ in arms}
    models = {}
    for n, ck in arms:
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        m = PixelSR(blob["args"]["width"], blob["args"]["blocks"], 3, blob["taps"]).to(dev).eval()
        m.load_state_dict(blob["model"])
        models[n] = (m, blob["args"]["window"])

    for vid in ids:
        lr_np = np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")
        hr_np = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")
        T = min(24, lr_np.shape[0])
        lr = to_t(np.asarray(lr_np[:T]), dev)
        gt = to_t(np.asarray(hr_np[:T]), dev)
        bic = F.interpolate(lr, scale_factor=3, mode="bicubic", align_corners=False).clamp(-1, 1)
        gt_e.append(hf_energy(gt)); bic_e.append(hf_energy(bic))
        for n, (m, W) in models.items():
            with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                o = m(lr, frames=W if T % W == 0 else 1).float()
            out_e[n].append(hf_energy(o))

    g = sum(gt_e) / len(gt_e)
    print(f"{'arm':12s}{'hf_energy':>12s}{'rel_GT':>9s}")
    print(f"{'GT':12s}{g:12.5f}{1.0:9.2f}")
    b = sum(bic_e) / len(bic_e)
    print(f"{'bicubic':12s}{b:12.5f}{b/g:9.2f}")
    res = dict(gt=g, bicubic=b)
    for n in out_e:
        v = sum(out_e[n]) / len(out_e[n])
        res[n] = v
        print(f"{n:12s}{v:12.5f}{v/g:9.2f}")
    json.dump(res, open("results/hf_arms.json", "w"), indent=2)
    os._exit(0)


main()
