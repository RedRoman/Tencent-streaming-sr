"""How much detail is actually recoverable, and is LPIPS doing anything?

Three questions:
 1. frequency content -- how much of GT's high-frequency energy survives in LR
 2. an oracle -- GT downsampled then upsampled, i.e. the best any 3x SR can do
    if it perfectly inverts the resampling but invents nothing
 3. whether our outputs differ from bicubic at all, measured not guessed
"""
import os, random, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch_npu  # noqa: F401
from pixel_model import PixelSR
from pixel_train import PIX, to_t

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
RUN = Path(open(f"{ROOT}/results/CURRENT_SHARP").read().strip())
dev = "npu:0"


def hf_energy(x):
    """Fraction of total energy above a 1/3 Nyquist cut, per frame."""
    g = x.mean(1, keepdim=True)
    lo = F.avg_pool2d(g, 3, stride=1, padding=1)
    return float(((g - lo) ** 2).mean() / ((g ** 2).mean() + 1e-9))


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) / 4 + 1e-12))


ids = sorted(f[:-4] for f in os.listdir(f"{PIX}/hr") if f.endswith(".npy"))
random.Random(1234).shuffle(ids)
out = []
for vid in ids[:3]:
    hr = to_t(np.asarray(np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")[60:61]), dev)
    lr = to_t(np.asarray(np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")[60:61]), dev)
    bic = F.interpolate(lr, scale_factor=3, mode="bicubic", align_corners=False).clamp(-1, 1)
    # Oracle: GT resampled down and back up -- no invention, perfect inversion
    down = F.interpolate(hr, scale_factor=1/3, mode="bicubic", align_corners=False, antialias=True)
    oracle = F.interpolate(down, scale_factor=3, mode="bicubic", align_corners=False).clamp(-1, 1)

    row = dict(clip=vid,
               hf_gt=round(hf_energy(hr), 5),
               hf_lr_upsampled=round(hf_energy(bic), 5),
               hf_oracle=round(hf_energy(oracle), 5),
               psnr_oracle=round(psnr(oracle, hr), 2),
               psnr_bicubic=round(psnr(bic, hr), 2))
    for n in ("w160_l1", "w160_lpips"):
        ck = RUN / n / "best.pt"
        if not ck.exists():
            continue
        b = torch.load(ck, map_location="cpu", weights_only=False); a = b["args"]
        m = PixelSR(a["width"], a["blocks"], 3, b["taps"]).to(dev).eval(); m.load_state_dict(b["model"])
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            p = m(lr, frames=1).float()
        row[f"hf_{n}"] = round(hf_energy(p), 5)
        row[f"psnr_{n}"] = round(psnr(p, hr), 2)
        # how far the model moved away from its own bicubic starting point
        row[f"delta_from_bicubic_{n}"] = round(float((p - bic).abs().mean()), 5)
        del m; torch.npu.empty_cache()
    out.append(row); print(json.dumps(row), flush=True)

k = out[0]
print("\n-- high-frequency energy, relative to GT = 1.00")
for key in [x for x in k if x.startswith("hf_")]:
    v = sum(r[key] for r in out) / len(out)
    g = sum(r["hf_gt"] for r in out) / len(out)
    print(f"   {key:<22} {v:.5f}   ({v/g:.2f}x GT)")
print("\n-- PSNR")
for key in [x for x in k if x.startswith("psnr_")]:
    print(f"   {key:<22} {sum(r[key] for r in out)/len(out):.2f}")
print("\n-- mean |model - bicubic| (0 = indistinguishable from the baseline)")
for key in [x for x in k if x.startswith("delta_")]:
    print(f"   {key:<34} {sum(r[key] for r in out)/len(out):.5f}")
os._exit(0)
