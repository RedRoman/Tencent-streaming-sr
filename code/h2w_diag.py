"""Localize the -10.7 dB: is it the mapper, or the Wan VAE round-trip itself?

Upper bound test: encode the GT HR frames with the Wan VAE and decode them back.
Whatever PSNR that reaches is the ceiling for anything that goes through Wan
latent space. If the ceiling is already low, the route is capped regardless of
how good the mapper is.
"""
import os, random, sys, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
PIX = ROOT / "data/pixels"
WAN_PTH = "/temp/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def main():
    dev = "npu:0"
    from vendor.wan_vae import WanVAE
    wan = WanVAE(vae_pth=WAN_PTH, dtype=torch.bfloat16, device=dev)
    wan.model.to(dtype=torch.bfloat16).eval().requires_grad_(False)

    ids = sorted(f[:-4] for f in os.listdir(f"{PIX}/hr") if f.endswith(".npy"))
    random.Random(1234).shuffle(ids)
    for vid in ids[:2]:
        hr = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")
        x = torch.from_numpy(np.asarray(hr[:17])).permute(3, 0, 1, 2).float().unsqueeze(0) / 255
        x = x.to(dev)                                  # [1,3,17,768,1344] in [0,1]
        wan.model.clear_cache()
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            z = wan.model.encode((x * 2 - 1).to(torch.bfloat16), wan.scale).float()
        wan.model.clear_cache()
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            y = wan.model.decode(z.to(torch.bfloat16), wan.scale)
        wan.model.clear_cache()
        y = (y.float().clamp(-1, 1) + 1) / 2
        print(json.dumps(dict(clip=vid, latent=list(z.shape), pixels=list(y.shape),
                              wan_roundtrip_psnr=round(psnr(y, x), 3),
                              z_mean=round(float(z.mean()), 3),
                              z_std=round(float(z.std()), 3))), flush=True)
    os._exit(0)


main()
