"""Where is the quality ceiling, and what is actually missing?

Three questions the answer depends on:
  1. How much of the gap to GT is irrecoverable (lost to 3x downsampling)?
  2. How much is recoverable but we are not getting it (model capacity/loss)?
  3. Is the LR we receive itself degraded relative to what H3 could give us?

(3) matters because our LR comes from decoding a stored H3 latent. If that
decode is lossy, part of our ceiling is lost before the refiner ever runs.
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


def hf(x):
    g = x.mean(1, keepdim=True)
    return float(((g - F.avg_pool2d(g, 3, 1, 1)) ** 2).mean() / ((g ** 2).mean() + 1e-9))


def main():
    import lpips as lp
    dev = "npu:0"
    L = lp.LPIPS(net="vgg").to(dev).eval()
    for q in L.parameters():
        q.requires_grad_(False)

    b = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    a = b["args"]
    sr = PixelSR(a["width"], a["blocks"], 3, b["taps"]).to(dev).eval()
    sr.load_state_dict(b["model"])

    ids = sorted(f[:-4] for f in os.listdir(f"{PIX}/hr") if f.endswith(".npy"))
    random.Random(1234).shuffle(ids)
    rows = []
    for vid in ids[:4]:
        hr = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")
        lr = np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")
        n = 24
        g = torch.from_numpy(np.asarray(hr[:n])).permute(0, 3, 1, 2).to(dev).float() / 127.5 - 1
        l = torch.from_numpy(np.asarray(lr[:n])).permute(0, 3, 1, 2).to(dev).float() / 127.5 - 1

        # ORACLE: GT downsampled 3x then our refiner run on it. This is the best
        # the refiner could do if the LR it received were a clean 3x view of GT.
        clean_lr = F.interpolate(g, scale_factor=1/3, mode="bicubic",
                                 align_corners=False, antialias=True).clamp(-1, 1)
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            ours = torch.cat([sr(l[i:i+1], frames=1).float() for i in range(n)])
            oracle = torch.cat([sr(clean_lr[i:i+1], frames=1).float() for i in range(n)])
        bic = F.interpolate(l, scale_factor=3, mode="bicubic", align_corners=False).clamp(-1, 1)

        def LP(x):
            return float(torch.cat([L(x[i:i+8], g[i:i+8]).flatten()
                                    for i in range(0, n, 8)]).mean())

        rows.append(dict(clip=vid,
            # how different is our received LR from a clean downsample of GT?
            lr_vs_cleanlr_psnr=round(psnr(l, clean_lr), 2),
            psnr_ours=round(psnr(ours, g), 2),
            psnr_oracle_cleanlr=round(psnr(oracle, g), 2),
            psnr_bicubic=round(psnr(bic, g), 2),
            lpips_ours=round(LP(ours), 4),
            lpips_oracle=round(LP(oracle), 4),
            lpips_bicubic=round(LP(bic), 4),
            hf_gt=round(hf(g), 5), hf_ours=round(hf(ours), 5), hf_bic=round(hf(bic), 5)))
        print(json.dumps(rows[-1]), flush=True)
        torch.npu.empty_cache()

    avg = lambda k: round(sum(r[k] for r in rows) / len(rows), 4)
    m = {k: avg(k) for k in rows[0] if k != "clip"}
    m["oracle_headroom_psnr"] = round(m["psnr_oracle_cleanlr"] - m["psnr_ours"], 2)
    m["oracle_headroom_lpips"] = round(m["lpips_ours"] - m["lpips_oracle"], 4)
    print("\nMEAN " + json.dumps(m), flush=True)
    os._exit(0)


main()
