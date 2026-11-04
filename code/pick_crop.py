"""Find a crop with real high-frequency detail, then render the four panels there.

The previous crop was mostly flat black, where every method looks identical.
Score candidate windows by the GT's local gradient energy and take the busiest.
"""
import os, sys, random
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch_npu  # noqa: F401
from PIL import Image
from pixel_model import PixelSR
from pixel_train import PIX, to_t

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
RUN = Path(sys.argv[1]); dev = "npu:0"

ids = sorted(f[:-4] for f in os.listdir(f"{PIX}/hr") if f.endswith(".npy"))
random.Random(1234).shuffle(ids)
vid = ids[0]
hr_np = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")
lr_np = np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")
FR, S = 60, 180
gt_full = to_t(np.asarray(hr_np[FR:FR + 1]), dev)

# gradient energy per candidate window
g = gt_full.mean(1, keepdim=True)
gx = (g[..., 1:] - g[..., :-1]).abs().mean()
best, by, bx = -1, 0, 0
H, W = gt_full.shape[-2:]
for y in range(0, H - S, 60):
    for x in range(0, W - S, 60):
        c = g[..., y:y + S, x:x + S]
        e = float((c[..., 1:] - c[..., :-1]).abs().mean() + (c[..., 1:, :] - c[..., :-1, :]).abs().mean())
        if e > best:
            best, by, bx = e, y, x
print(f"clip {vid} frame {FR}: busiest {S}x{S} crop at y={by} x={bx} (energy {best:.4f})", flush=True)

lr = to_t(np.asarray(lr_np[FR:FR + 1]), dev)
bic = F.interpolate(lr, scale_factor=3, mode="bicubic", align_corners=False).clamp(-1, 1)
panels = [("bicubic", bic)]
for n in ("w160_l1", "w160_lpips"):
    b = torch.load(RUN / n / "best.pt", map_location="cpu", weights_only=False)
    a = b["args"]
    m = PixelSR(a["width"], a["blocks"], 3, b["taps"]).to(dev).eval(); m.load_state_dict(b["model"])
    with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
        panels.append((f"{n}@{b['step']}", m(lr, frames=1).float()))
    del m; torch.npu.empty_cache()
panels.append(("GT", gt_full))

tiles = []
for name, t in panels:
    c = t[0, :, by:by + S, bx:bx + S]
    img = ((c.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
    tiles.append(np.asarray(Image.fromarray(img).resize((S * 3, S * 3), Image.NEAREST)))
Image.fromarray(np.concatenate(tiles, axis=1)).save("results/arm_crop.png")

# also a full frame of the L1 arm for overall impression
full = ((panels[1][1][0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
Image.fromarray(full).resize((full.shape[1] // 2, full.shape[0] // 2), Image.LANCZOS).save("results/arm_full.png")
print("wrote results/arm_crop.png  order: " + " | ".join(n for n, _ in panels), flush=True)
print("wrote results/arm_full.png (half-size full frame of " + panels[1][0] + ")", flush=True)
os._exit(0)
