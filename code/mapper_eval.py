"""Final acceptance for the H3 -> Wan -> UltraFlash route.

Scores against the ORIGINAL HR pixels, the same reference the pixel-domain
baseline (bicubic 25.96 / w1 27.00) was measured on, so the three routes are
directly comparable. Three numbers are reported per clip:

  ours      mapper + frozen UltraFlash upsampler, decoded by the Wan VAE
  bicubic   LR pixels upscaled 3x -- the bar
  ceiling   Wan encode of the HR pixels, decoded back: the best this route can
            ever reach, since everything passes through Wan latent space

Also emits the LR | ours | GT triptych for the demo video.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
PIX = ROOT / "data/pixels"
PAIRS = ROOT / "data/wanpairs"
WAN_PTH = "/temp/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
SEG_PX, SEG_LAT, NSEG = 22, 7, 5


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def ssim(a, b):
    a = a.mean(1, keepdim=True); b = b.mean(1, keepdim=True)
    k = torch.ones(1, 1, 11, 11, device=a.device, dtype=a.dtype) / 121
    ma, mb = F.conv2d(a, k), F.conv2d(b, k)
    sa = F.conv2d(a * a, k) - ma ** 2; sb = F.conv2d(b * b, k) - mb ** 2
    sab = F.conv2d(a * b, k) - ma * mb
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float((((2 * ma * mb + c1) * (2 * sab + c2)) /
                  ((ma ** 2 + mb ** 2 + c1) * (sa + sb + c2))).mean())


def flicker(v):
    return float((v[1:] - v[:-1]).abs().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--save-video", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from h3wan.model import Mapper
    from vendor.wan_vae import WanVAE
    from mapper_train import load_upsampler, wan_decode, VAL_CLIPS

    dev = "npu:0"; args.out.mkdir(parents=True, exist_ok=True)
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mapper = Mapper().to(dev); mapper.load_state_dict(blob["mapper"]); mapper.eval()
    up = load_upsampler(dev, from_joint=True)
    if "upsampler" in blob:
        up.load_state_dict(blob["upsampler"]); print("using fine-tuned upsampler", flush=True)
    up.eval()
    wan = WanVAE(vae_pth=WAN_PTH, dtype=torch.bfloat16, device=dev)
    wan.model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    print(f"ckpt step={blob['step']} args={blob['args'].get('init')} "
          f"train_up={blob['args'].get('train_upsampler')}", flush=True)

    ids = sorted(f[:-4] for f in os.listdir(f"{PAIRS}/wan")
                 if f.endswith(".npy") and not f.endswith(".tmp.npy"))
    random.Random(1234).shuffle(ids)
    rows = []
    for n, vid in enumerate(ids[: args.clips]):
        h3 = torch.from_numpy(np.load(f"{PAIRS}/h3/{vid}.npy")).float()
        wtrue = torch.from_numpy(np.load(f"{PAIRS}/wan/{vid}.npy")).float()
        hrpx = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")
        lrpx = np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")

        ours, ceil, t_lat, t_dec = [], [], 0.0, 0.0
        for s in range(NSEG):
            a, b = s * SEG_LAT, (s + 1) * SEG_LAT
            hseg = h3[:, a:b].unsqueeze(0).to(dev)
            torch.npu.synchronize(); t0 = time.time()
            with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                low = mapper(hseg)
                high = torch.cat([up(low[:, :, :5], parallel=True),
                                  up(low[:, :, 5:], parallel=True)], dim=2).float()
            torch.npu.synchronize(); t_lat += time.time() - t0
            t0 = time.time()
            ours.append(wan_decode(wan, high).cpu())
            torch.npu.synchronize(); t_dec += time.time() - t0
            ceil.append(wan_decode(wan, wtrue[:, a:b].unsqueeze(0).to(dev)).cpu())
            torch.npu.empty_cache()

        pred = torch.cat(ours, dim=2)[0].permute(1, 0, 2, 3)     # [T,3,H,W] in [0,1]
        cl = torch.cat(ceil, dim=2)[0].permute(1, 0, 2, 3)
        T = pred.shape[0]
        gt = torch.from_numpy(np.asarray(hrpx[:T])).permute(0, 3, 1, 2).float() / 255
        lr_up = F.interpolate(
            torch.from_numpy(np.asarray(lrpx[:T])).permute(0, 3, 1, 2).float() / 255,
            scale_factor=3, mode="bicubic", align_corners=False).clamp(0, 1)

        row = dict(clip=vid, frames=T,
                   psnr=round(psnr(pred, gt), 3),
                   psnr_bicubic=round(psnr(lr_up, gt), 3),
                   psnr_ceiling=round(psnr(cl, gt), 3),
                   ssim=round(ssim(pred, gt), 4), ssim_bicubic=round(ssim(lr_up, gt), 4),
                   flicker=round(flicker(pred), 5), flicker_gt=round(flicker(gt), 5),
                   latent_ms_per_latent=round(t_lat / (NSEG * SEG_LAT) * 1000, 2),
                   decode_ms_per_latent=round(t_dec / (NSEG * SEG_LAT) * 1000, 1))
        row["gain"] = round(row["psnr"] - row["psnr_bicubic"], 3)
        row["headroom"] = round(row["psnr_ceiling"] - row["psnr"], 3)
        rows.append(row); print(json.dumps(row), flush=True)

        if n < args.save_video:
            def u8(x):
                return (x.permute(0, 2, 3, 1) * 255).round().clamp(0, 255).byte().numpy()
            np.save(args.out / f"{vid}_triptych.npy",
                    np.concatenate([u8(lr_up), u8(pred), u8(gt)], axis=2))

    def avg(k):
        return round(sum(r[k] for r in rows) / len(rows), 4)
    summary = dict(ckpt=str(args.ckpt), step=blob["step"], clips=len(rows),
                   **{k: avg(k) for k in rows[0] if k not in ("clip", "frames")})
    (args.out / "summary.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=2))
    print("\nSUMMARY " + json.dumps(summary), flush=True)
    os._exit(0)


main()
