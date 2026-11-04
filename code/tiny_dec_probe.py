"""Can the UltraFlash tiny decoder replace the 590 ms Wan VAE decode?

It is a TAEHV built for Wan's 16-channel latent with the same 8x spatial / 4x
temporal geometry, so it drops straight into this route. Two things decide it:
speed against the 833 ms chunk budget, and how much quality is lost relative to
the full Wan decode it replaces.
"""
import json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
sys.path.insert(0, str(ROOT / "h2w/vendor/UltraFlash/inference/sr"))
TINY = "/temp/rongxiang_workspace_backup_20260921/rongxiang/UltraFlash/inference/checkpoints/tiny_decoder.pth"
WAN_PTH = "/temp/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
PAIRS = ROOT / "data/wanpairs"
PIX = ROOT / "data/pixels"


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def main():
    dev = "npu:0"
    from tiny_decoder import WanTinyDecoder
    from vendor.wan_vae import WanVAE

    tiny = WanTinyDecoder(TINY, dtype=torch.float32).to(dev).eval()
    print(f"tiny decoder: {sum(p.numel() for p in tiny.parameters())/1e6:.2f}M params", flush=True)
    wan = WanVAE(vae_pth=WAN_PTH, dtype=torch.bfloat16, device=dev)
    wan.model.to(dtype=torch.bfloat16).eval().requires_grad_(False)

    ids = sorted(f[:-4] for f in os.listdir(f"{PAIRS}/wan")
                 if f.endswith(".npy") and not f.endswith(".tmp.npy"))
    random.Random(1234).shuffle(ids)
    rows = []
    for vid in ids[:3]:
        z = torch.from_numpy(np.load(f"{PAIRS}/wan/{vid}.npy")).float()[:, :5].unsqueeze(0).to(dev)
        hrpx = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")

        def run_tiny():
            with torch.no_grad():
                return tiny.decode_latents(z)
        def run_wan():
            wan.model.clear_cache()
            with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                r = wan.model.decode(z.to(torch.bfloat16), wan.scale)
            wan.model.clear_cache()
            return r

        for _ in range(2):
            run_tiny(); run_wan()
        torch.npu.synchronize(); t0 = time.time()
        for _ in range(3):
            pt = run_tiny()
        torch.npu.synchronize(); ms_tiny = (time.time() - t0) / 3 * 1000
        t0 = time.time()
        for _ in range(3):
            pw = run_wan()
        torch.npu.synchronize(); ms_wan = (time.time() - t0) / 3 * 1000

        pt = pt.float().clamp(-1, 1); pw = pw.float().clamp(-1, 1)
        n = min(pt.shape[2], pw.shape[2])
        gt = torch.from_numpy(np.asarray(hrpx[:n])).permute(3, 0, 1, 2).unsqueeze(0).to(dev).float() / 127.5 - 1
        rows.append(dict(clip=vid, tiny_shape=list(pt.shape), wan_shape=list(pw.shape),
                         tiny_ms=round(ms_tiny, 1), wan_ms=round(ms_wan, 1),
                         speedup=round(ms_wan / ms_tiny, 1),
                         psnr_tiny_vs_wan=round(psnr(pt[:, :, :n], pw[:, :, :n]), 2),
                         psnr_tiny_vs_gt=round(psnr(pt[:, :, :n], gt[:, :, :n]), 2),
                         psnr_wan_vs_gt=round(psnr(pw[:, :, :n], gt[:, :, :n]), 2)))
        print(json.dumps(rows[-1]), flush=True)
        torch.npu.empty_cache()

    a = lambda k: round(sum(r[k] for r in rows) / len(rows), 2)
    print("\nMEAN " + json.dumps(dict(tiny_ms=a("tiny_ms"), wan_ms=a("wan_ms"),
          speedup=a("speedup"), psnr_tiny_vs_wan=a("psnr_tiny_vs_wan"),
          psnr_tiny_vs_gt=a("psnr_tiny_vs_gt"), psnr_wan_vs_gt=a("psnr_wan_vs_gt"))), flush=True)
    os._exit(0)


main()
