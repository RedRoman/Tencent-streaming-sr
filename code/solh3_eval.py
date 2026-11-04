"""Sample a trained Sol-H3 refiner, decode to pixels, and score against HR truth.

Reports, on held-out clips:
  - latent PSNR of the sampled HR latent
  - pixel PSNR / SSIM after the real H3 VAE decode
  - temporal flicker: mean |frame_t - frame_{t-1}| difference vs the HR reference,
    which is where a too-short refiner window would show up
Baselines on the same clips and the same decode path: trilinear-upsampled LR, and
the LR latent decoded at its own resolution then bicubic-resized in pixel space.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from diffusers import AutoencoderKLMiniMaxH3
from solh3_model import SolH3Refiner
from solh3_train import LOW, HIGH, VAL_IDS, Pairs, upsample_lr

VAE_DIR = "/home/ma-user/modelarts/user-job-dir/sol_h3/ckpts/h3-base/vae"


@torch.no_grad()
def sample(model, lr_cond, steps, device, seed=0, objective="noise", bridge_start=None):
    """Euler integration of the learned velocity field down to t=0.

    The t=1 endpoint must match training: pure noise for the "noise" objective,
    the upsampled LR itself for the "bridge" objective (where few-step sampling
    therefore starts already near the answer).
    """
    if objective == "bridge":
        # The trajectory lives on the HR grid even when `lr_cond` is a raw LR
        # that the model re-grids itself.
        x = (bridge_start if bridge_start is not None else lr_cond).clone()
    else:
        g = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(lr_cond.shape, generator=g).to(device)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t = ts[i].expand(x.shape[0])
        with torch.autocast("npu", dtype=torch.bfloat16):
            v = model(x, lr_cond, t).float()
        x = x + (ts[i + 1] - ts[i]) * v
    return x


@torch.no_grad()
def decode(vae, z, mean, std):
    z = (z.to(torch.float32) * std + mean)
    px = vae.decode(z).sample.float().clamp(-1, 1)
    return (px + 1) / 2  # [0,1]


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def ssim(a, b):
    """Global SSIM on the luma channel, 11x11 uniform window."""
    a = a.mean(1, keepdim=True)
    b = b.mean(1, keepdim=True)
    k = torch.ones(1, 1, 11, 11, device=a.device) / 121
    mu_a, mu_b = F.conv2d(a, k), F.conv2d(b, k)
    sa = F.conv2d(a * a, k) - mu_a ** 2
    sb = F.conv2d(b * b, k) - mu_b ** 2
    sab = F.conv2d(a * b, k) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (sa + sb + c2))
    return float(s.mean())


def flicker(v):
    """Mean absolute temporal difference; v is [T,3,H,W] in [0,1]."""
    return float((v[1:] - v[:-1]).abs().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=8, help="sampling NFE")
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--frames", type=int, default=37, help="latent frames to refine per clip")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-video", type=int, default=2)
    args = ap.parse_args()

    device = "npu:0"
    args.out.mkdir(parents=True, exist_ok=True)

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    targs = blob["args"]
    model = SolH3Refiner(width=targs["width"], time_taps=targs["time_taps"],
                         cond_mode=targs["cond"],
                         learned_upsample=targs.get("learned_upsample", False)).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    chunk = targs["chunk"]
    print(f"loaded {args.ckpt} step={blob['step']} chunk={chunk} cond={targs['cond']}", flush=True)

    vae = AutoencoderKLMiniMaxH3.from_pretrained(VAE_DIR, torch_dtype=torch.float32).to(device).eval()
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)

    ids = sorted(f[:-8] for f in os.listdir(HIGH) if f.endswith(".pt.part"))
    rng_split = random.Random(1234)
    rng_split.shuffle(ids)
    val = Pairs(ids[:VAL_IDS], targs.get("lr_source", "h3"))

    rows = []
    for n in range(args.clips):
        lo, hi = val.get(n)
        T = min(args.frames, hi.shape[1])
        lo, hi = lo[:, :T].unsqueeze(0).to(device), hi[:, :T].unsqueeze(0).to(device)
        lr_up = upsample_lr(lo)
        raw_lr = targs.get("learned_upsample", False)
        cond = lo if raw_lr else lr_up

        # Refine in consecutive chunks of the trained window, concatenated.
        # Time the per-chunk cost: this is the latency the window size buys.
        pieces, chunk_ms = [], []
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            torch.npu.synchronize()
            t0 = time.time()
            pieces.append(sample(model, cond[:, :, s:e], args.steps, device, seed=1000 + n,
                                 objective=targs.get("objective", "noise"),
                                 bridge_start=lr_up[:, :, s:e]))
            torch.npu.synchronize()
            chunk_ms.append((time.time() - t0) * 1000)
        pred = torch.cat(pieces, dim=2)
        # One latent frame carries 4 pixel frames = 166.7 ms of 24 fps video.
        ms_per_latent = sum(chunk_ms) / T

        lat_psnr = psnr(pred, hi)
        lat_psnr_base = psnr(lr_up, hi)

        gt_px = decode(vae, hi, mean, std)[0].permute(1, 0, 2, 3)      # [T,3,H,W]
        pr_px = decode(vae, pred, mean, std)[0].permute(1, 0, 2, 3)
        lr_px = decode(vae, lo, mean, std)[0].permute(1, 0, 2, 3)
        lr_px_up = F.interpolate(lr_px, size=gt_px.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)

        row = dict(clip=ids[n], frames=T,
                   latent_psnr=round(lat_psnr, 3),
                   latent_psnr_trilinear=round(lat_psnr_base, 3),
                   pixel_psnr=round(psnr(pr_px, gt_px), 3),
                   pixel_psnr_bicubic=round(psnr(lr_px_up, gt_px), 3),
                   ssim=round(ssim(pr_px, gt_px), 4),
                   ssim_bicubic=round(ssim(lr_px_up, gt_px), 4),
                   flicker_pred=round(flicker(pr_px), 5),
                   flicker_gt=round(flicker(gt_px), 5),
                   flicker_bicubic=round(flicker(lr_px_up), 5),
                   refine_ms_per_latent=round(ms_per_latent, 1),
                   refine_realtime_x=round(166.7 / ms_per_latent, 2),
                   ttff_ms=round((chunk - 1) * 166.7 + chunk_ms[0], 1))
        rows.append(row)
        print(json.dumps(row), flush=True)

        if n < args.save_video:
            trip = torch.cat([lr_px_up, pr_px, gt_px], dim=3)  # side by side
            np.save(args.out / f"{ids[n]}_triptych.npy",
                    (trip.permute(0, 2, 3, 1) * 255).round().byte().cpu().numpy())

    def avg(k):
        return round(sum(r[k] for r in rows) / len(rows), 4)

    summary = dict(ckpt=str(args.ckpt), step=blob["step"], chunk=chunk, cond=targs["cond"],
                   objective=targs.get("objective", "noise"),
                   lr_source=targs.get("lr_source") or "h3",
                   nfe=args.steps, clips=len(rows),
                   **{k: avg(k) for k in rows[0] if k not in ("clip", "frames")})
    (args.out / "summary.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=2))
    print("\nSUMMARY " + json.dumps(summary), flush=True)
    os._exit(0)


main()
