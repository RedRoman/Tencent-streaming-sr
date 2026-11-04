"""Put bicubic / L1 / LPIPS / GT side by side on the same crop.

PSNR cannot answer "does it look sharper"; this renders the actual pixels so the
difference between an L1-trained and a perceptually-trained refiner is visible.
"""
import os, sys, random
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch_npu  # noqa: F401
from PIL import Image
from pixel_model import PixelSR
from pixel_train import PIX, to_t

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
RUN = Path(sys.argv[1])
dev = "npu:0"


def load(name):
    b = torch.load(RUN / name / "best.pt", map_location="cpu", weights_only=False)
    a = b["args"]
    m = PixelSR(a["width"], a["blocks"], 3, b["taps"]).to(dev).eval()
    m.load_state_dict(b["model"])
    return m, b["step"], a["width"], a["lpips"]


arms = {}
for n in ("w160_l1", "w160_lpips"):
    try:
        arms[n] = load(n)
        print(f"{n}: step {arms[n][1]} w{arms[n][2]} lpips={arms[n][3]}", flush=True)
    except Exception as e:
        print(f"{n}: {e}", flush=True)

ids = sorted(f[:-4] for f in os.listdir(f"{PIX}/hr") if f.endswith(".npy"))
random.Random(1234).shuffle(ids)
vid = ids[0]
lr_np = np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")
hr_np = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")
FR = 60
lr = to_t(np.asarray(lr_np[FR:FR + 1]), dev)
gt = to_t(np.asarray(hr_np[FR:FR + 1]), dev)
bic = F.interpolate(lr, scale_factor=3, mode="bicubic", align_corners=False).clamp(-1, 1)

panels = [("bicubic", bic)]
for n, (m, _, _, _) in arms.items():
    with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
        panels.append((n, m(lr, frames=1).float()))
panels.append(("GT", gt))

# A detail crop where texture matters, magnified so softness is obvious.
H, W = gt.shape[-2:]
y, x, s = int(H * 0.40), int(W * 0.35), 180
tiles = []
for name, t in panels:
    c = t[0, :, y:y + s, x:x + s]
    img = ((c.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
    img = np.asarray(Image.fromarray(img).resize((s * 3, s * 3), Image.NEAREST))
    tiles.append(img)
    print(f"  panel {name}", flush=True)
sheet = np.concatenate(tiles, axis=1)
Image.fromarray(sheet).save("results/arm_compare.png")
print(f"wrote results/arm_compare.png  order: {' | '.join(n for n,_ in panels)}", flush=True)
os._exit(0)
