"""Render bicubic / no-GAN / GAN arms / GT on a high-detail crop.

PSNR ranks these backwards by construction, so the deciding evidence is the
pixels: does the adversarial term actually synthesize texture, and at which
weight does it start inventing artifacts instead.
"""
import os, sys, random
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch_npu  # noqa: F401
from PIL import Image, ImageDraw
from pixel_model import PixelSR
from pixel_train import PIX, to_t

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
GAN = Path(open(f"{ROOT}/results/CURRENT_GAN").read().strip())
SHARP = ROOT / "results/sharp_0101"
dev = "npu:0"

ARMS = [("L1", SHARP / "w160_l1"), ("LPIPS", GAN / "lpips1_nogan"),
        ("GAN .02", GAN / "gan002"), ("GAN .05", GAN / "gan005"),
        ("GAN .15", GAN / "gan015")]


def hf(x):
    g = x.mean(1, keepdim=True)
    return float(((g - F.avg_pool2d(g, 3, 1, 1)) ** 2).mean() / ((g ** 2).mean() + 1e-9))


ids = sorted(f[:-4] for f in os.listdir(f"{PIX}/hr") if f.endswith(".npy"))
random.Random(1234).shuffle(ids)
vid = ids[0]; FR, S = 60, 150
hr = to_t(np.asarray(np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")[FR:FR + 1]), dev)
lr = to_t(np.asarray(np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")[FR:FR + 1]), dev)
bic = F.interpolate(lr, scale_factor=3, mode="bicubic", align_corners=False).clamp(-1, 1)

# busiest window in GT, so softness is visible
g = hr.mean(1, keepdim=True); best = (-1, 0, 0)
for y in range(0, hr.shape[-2] - S, 50):
    for x in range(0, hr.shape[-1] - S, 50):
        c = g[..., y:y + S, x:x + S]
        e = float((c[..., 1:] - c[..., :-1]).abs().mean() + (c[..., 1:, :] - c[..., :-1, :]).abs().mean())
        if e > best[0]:
            best = (e, y, x)
_, by, bx = best
print(f"clip {vid} frame {FR} crop y={by} x={bx}", flush=True)

panels = [("bicubic", bic)]
for name, d in ARMS:
    ck = d / "latest.pt"
    if not ck.exists():
        print(f"  skip {name}", flush=True); continue
    b = torch.load(ck, map_location="cpu", weights_only=False); a = b["args"]
    m = PixelSR(a["width"], a["blocks"], 3, b["taps"]).to(dev).eval(); m.load_state_dict(b["model"])
    with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
        panels.append((name, m(lr, frames=1).float()))
    del m; torch.npu.empty_cache()
panels.append(("GT", hr))

print("\nhigh-frequency energy (GT = 1.00):", flush=True)
ghf = hf(hr)
for n, t in panels:
    print(f"   {n:<10} {hf(t):.5f}  ({hf(t)/ghf:.2f}x)", flush=True)

tiles, LBL = [], 34
for n, t in panels:
    c = t[0, :, by:by + S, bx:bx + S]
    a = ((c.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
    a = np.asarray(Image.fromarray(a).resize((S * 3, S * 3), Image.NEAREST))
    lab = Image.new("RGB", (S * 3, LBL), (15, 15, 15))
    ImageDraw.Draw(lab).text((8, 10), n, fill=(240, 240, 240))
    tiles.append(np.concatenate([np.asarray(lab), a], axis=0))
Image.fromarray(np.concatenate(tiles, axis=1)).save("results/gan_compare.png")
print("\nwrote results/gan_compare.png", flush=True)
os._exit(0)
