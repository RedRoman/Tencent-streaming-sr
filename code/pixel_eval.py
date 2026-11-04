"""Evaluate the pixel-domain refiner on full-resolution held-out clips and emit
the side-by-side video.

Everything is measured in the delivery domain: PSNR/SSIM against the HR decode,
temporal flicker, and the wall time of the refiner itself. bicubic is carried
through every row as the bar that has to be cleared.
"""
from __future__ import annotations

import argparse, json, os, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from pixel_model import PixelSR
from pixel_train import (PIX, VAL_CLIPS, TEXT_DIM, to_t, psnr_from_mse, pooled_text,
                         hr_latent, LAT_PER_LRPX, FRAMES_PER_LAT)


def ssim(a, b):
    """Global SSIM on luma, 11x11 uniform window; inputs in [-1,1]."""
    a = (a.mean(1, keepdim=True) + 1) / 2
    b = (b.mean(1, keepdim=True) + 1) / 2
    k = torch.ones(1, 1, 11, 11, device=a.device, dtype=a.dtype) / 121
    ma, mb = F.conv2d(a, k), F.conv2d(b, k)
    sa = F.conv2d(a * a, k) - ma ** 2
    sb = F.conv2d(b * b, k) - mb ** 2
    sab = F.conv2d(a * b, k) - ma * mb
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float((((2 * ma * mb + c1) * (2 * sab + c2)) /
                  ((ma ** 2 + mb ** 2 + c1) * (sa + sb + c2))).mean())


_LPIPS = None


def lpips_score(a, b, device):
    """Mean LPIPS over frames; inputs [T,3,H,W] in [-1,1]."""
    global _LPIPS
    if _LPIPS is None:
        import lpips as pkg
        _LPIPS = pkg.LPIPS(net="vgg").to(device).eval()
        for q in _LPIPS.parameters():
            q.requires_grad_(False)
    vals = []
    with torch.no_grad():
        for i in range(0, a.shape[0], 8):
            vals.append(float(_LPIPS(a[i:i + 8].to(device), b[i:i + 8].to(device)).mean()))
    return sum(vals) / len(vals)


def flicker(v):
    """Mean |frame_t - frame_{t-1}| over a [T,3,H,W] clip."""
    return float((v[1:] - v[:-1]).abs().mean() / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--save-video", type=int, default=2)
    ap.add_argument("--chunk-frames", type=int, default=16, help="frames per forward, memory only")
    ap.add_argument("--kf-interval", type=int, default=30,
                    help="frames between HR keyframes (the H3 side pays once per "
                         "interval); only used when the checkpoint has a reference")
    ap.add_argument("--text-clip", default=None,
                    help="borrow another clip's prompt: the wrong-caption control")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dev = "npu:0"
    args.out.mkdir(parents=True, exist_ok=True)
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a, taps = blob["args"], blob["taps"]
    kf = ("latent" if a.get("ref_latent") else bool(a.get("ref_max")))
    gate = bool(a.get("local_gate"))
    model = PixelSR(a["width"], a["blocks"], 3, taps,
                    text_dim=TEXT_DIM if a.get("text") else 0,
                    text_sites=max(a.get("text") or 1, 1),
                    keyframe_ref=kf, local_gate=gate).to(dev).eval()
    model.load_state_dict(blob["model"])
    W = a["window"]
    print(f"loaded {args.ckpt} step={blob['step']} window={W} taps={taps} "
          f"lr_source={a['lr_source']}", flush=True)

    ids = sorted(f[:-4] for f in os.listdir(f"{PIX}/hr") if f.endswith(".npy"))
    random.Random(1234).shuffle(ids)
    rows = []
    for n, vid in enumerate(ids[: args.clips]):
        lr_np = np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")
        hr_np = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")
        T = (lr_np.shape[0] // W) * W

        # One pooled prompt row for the clip; broadcast over every frame.
        txt = (pooled_text(args.text_clip or vid)[None].to(dev)
               if model.film is not None else None)
        # Keyframe schedule: a fresh HR latent every `--kf-interval` frames, each
        # one re-encoded once and then reused across the interval -- exactly what
        # the streaming driver does, so the metric matches the deployment.
        zref = hr_latent(vid) if model.kref is not None else None
        outs, bics, ms = [], [], 0.0
        # Advance by a whole number of windows. Stepping by `chunk_frames` and
        # then trimming the chunk end to a window multiple silently DROPPED the
        # frames in between (with chunk 16 and W=3, frame 15 vanished), so the
        # predictions were no longer contiguous while the ground truth still was
        # -- every W>1 arm was scored against misaligned frames and lost ~6 dB.
        step = args.chunk_frames - args.chunk_frames % W
        for s in range(0, T, step):
            e = min(s + step, T)
            e -= (e - s) % W
            if e <= s:
                continue
            lr = to_t(np.asarray(lr_np[s:e]), dev)
            rf = dist = chg = None
            if zref is not None:
                # The most recent keyframe at or before this chunk's first frame.
                kfr = (s // args.kf_interval) * args.kf_interval
                tl = min(int(round(kfr / FRAMES_PER_LAT)), zref.shape[0] - 1)
                rf_in = torch.from_numpy(
                    np.asarray(zref[tl], dtype=np.float32))[None].to(dev)
                dist = torch.tensor([[float(s - kfr)]], device=dev)
                if gate:
                    # Local change against the LR frame at the keyframe's time,
                    # exactly as training measured it.
                    kf_lr = to_t(np.asarray(lr_np[kfr:kfr + 1]), dev)
                    chg = (lr - kf_lr.expand_as(lr)).abs().mean(1, keepdim=True)
                with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                    rf = model.kref.encode(rf_in, lr.shape[-2:])
            torch.npu.synchronize(); t0 = time.time()
            with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                o = model(lr, frames=W, text=txt, ref_feat=rf, dist=dist,
                          change=chg).float()
            torch.npu.synchronize(); ms += (time.time() - t0) * 1000
            outs.append(o.cpu())
            bics.append(F.interpolate(lr, scale_factor=3, mode="bicubic",
                                      align_corners=False).clamp(-1, 1).cpu())
        pred = torch.cat(outs); bic = torch.cat(bics)
        gt = to_t(np.asarray(hr_np[: pred.shape[0]]), "cpu")

        row = dict(clip=vid, frames=int(pred.shape[0]),
                   psnr=round(psnr_from_mse(F.mse_loss(pred, gt)), 3),
                   psnr_bicubic=round(psnr_from_mse(F.mse_loss(bic, gt)), 3),
                   ssim=round(ssim(pred, gt), 4),
                   lpips=round(lpips_score(pred, gt, dev), 4),
                   lpips_bicubic=round(lpips_score(bic, gt, dev), 4),
                   ssim_bicubic=round(ssim(bic, gt), 4),
                   flicker=round(flicker(pred), 5),
                   flicker_gt=round(flicker(gt), 5),
                   flicker_bicubic=round(flicker(bic), 5),
                   ms_per_frame=round(ms / pred.shape[0], 2),
                   realtime_x=round((1000 / 24) / (ms / pred.shape[0]), 2))
        row["gain_psnr"] = round(row["psnr"] - row["psnr_bicubic"], 3)
        rows.append(row)
        print(json.dumps(row), flush=True)

        if n < args.save_video:
            def u8(x):
                return ((x.permute(0, 2, 3, 1) + 1) * 127.5).round().clamp(0, 255).byte().numpy()
            trip = np.concatenate([u8(bic), u8(pred), u8(gt)], axis=2)
            np.save(args.out / f"{vid}_triptych.npy", trip)

    def avg(k):
        return round(sum(r[k] for r in rows) / len(rows), 4)
    summary = dict(ckpt=str(args.ckpt), step=blob["step"], window=W, taps=taps,
                   lr_source=a["lr_source"], params=a.get("params"), clips=len(rows),
                   **{k: avg(k) for k in rows[0] if k not in ("clip", "frames")})
    (args.out / "summary.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=2))
    print("\nSUMMARY " + json.dumps(summary), flush=True)
    os._exit(0)


main()
