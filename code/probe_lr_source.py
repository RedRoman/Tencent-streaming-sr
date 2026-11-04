"""Our LR is a separate H3 generation, not a downsample of our HR. How much does
that misalignment cost, and would training on clean pairs fix it?

Measures:
  - the misalignment itself (our LR vs a clean 3x downsample of HR)
  - whether a refiner TRAINED on our LR transfers to clean LR, and vice versa
The degraded-LR control arm trained earlier is exactly a clean-pair model, so
compare the two directly on both input types.
"""
import json, os, random, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
from pixel_model import PixelSR
PIX = ROOT / "data/pixels"


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def load(ck, dev):
    b = torch.load(ck, map_location="cpu", weights_only=False)
    a = b["args"]
    m = PixelSR(a["width"], a["blocks"], 3, b["taps"]).to(dev).eval()
    m.load_state_dict(b["model"])
    return m, a.get("lr_source", "h3")


def main():
    import lpips as lp
    dev = "npu:0"
    L = lp.LPIPS(net="vgg").to(dev).eval()
    for q in L.parameters():
        q.requires_grad_(False)

    models = {}
    for tag, ck in (("trained_on_h3LR", sys.argv[1]), ("trained_on_cleanLR", sys.argv[2])):
        if os.path.exists(ck):
            models[tag] = load(ck, dev)[0]
            print(f"loaded {tag}", flush=True)

    ids = sorted(f[:-4] for f in os.listdir(f"{PIX}/hr") if f.endswith(".npy"))
    random.Random(1234).shuffle(ids)
    acc = {}
    for vid in ids[:4]:
        hr = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")
        lr = np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")
        n = 24
        g = torch.from_numpy(np.asarray(hr[:n])).permute(0, 3, 1, 2).to(dev).float() / 127.5 - 1
        h3lr = torch.from_numpy(np.asarray(lr[:n])).permute(0, 3, 1, 2).to(dev).float() / 127.5 - 1
        clean = F.interpolate(g, scale_factor=1/3, mode="bicubic",
                              align_corners=False, antialias=True).clamp(-1, 1)

        def LP(x):
            return float(torch.cat([L(x[i:i+8], g[i:i+8]).flatten()
                                    for i in range(0, n, 8)]).mean())

        for mtag, m in models.items():
            for itag, inp in (("h3LR", h3lr), ("cleanLR", clean)):
                with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                    o = torch.cat([m(inp[i:i+1], frames=1).float() for i in range(n)])
                k = f"{mtag}|{itag}"
                acc.setdefault(k, {"psnr": [], "lpips": []})
                acc[k]["psnr"].append(psnr(o, g))
                acc[k]["lpips"].append(LP(o))
        torch.npu.empty_cache()

    print()
    print("%-34s%9s%9s" % ("model | input", "PSNR", "LPIPS"))
    for k, v in acc.items():
        print("%-34s%9.2f%9.4f" % (k, sum(v["psnr"])/len(v["psnr"]),
                                   sum(v["lpips"])/len(v["lpips"])), flush=True)
    os._exit(0)


main()
