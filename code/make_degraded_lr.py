"""Build the degraded-LR control set: the ordinary-streaming-SR condition.

The H3-generated LR latents in `low/` are a coherent generative trajectory. The
control replaces them with what a conventional SR system actually gets -- the HR
video bicubic-downsampled to the same 448x256 and encoded by the same VAE -- so
that a later W ablation isolates *where the temporal consistency comes from*
rather than the resolution change.

Everything else (resolution, latent geometry, VAE, normalization, clip ids) is
held identical to the H3-LR set.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from diffusers import AutoencoderKLMiniMaxH3

VAE_DIR = "/home/ma-user/modelarts/user-job-dir/sol_h3/ckpts/h3-base/vae"
VIDEOS = "/temp/yunfeng/datasets/h3_selected_800/videos"
HIGH = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/high/video"
OUT = "/home/ma-user/modelarts/user-job-dir/sol_h3/data/degraded_lr"

SRC_W, SRC_H = 1344, 768
LR_W, LR_H = 448, 256


def read_mp4(path):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-"], capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.decode()[-300:])
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, SRC_H, SRC_W, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    dev = "npu:0"
    vae = AutoencoderKLMiniMaxH3.from_pretrained(VAE_DIR, torch_dtype=torch.float32).to(dev).eval()
    mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)

    ids = sorted(f[:-8] for f in os.listdir(HIGH) if f.endswith(".pt.part"))
    ids = ids[args.shard :: args.shards]
    print(f"shard {args.shard}/{args.shards}: {len(ids)} clips", flush=True)

    t0 = time.time()
    done = 0
    for n, vid in enumerate(ids):
        dst = f"{OUT}/{vid}.pt"
        if os.path.exists(dst):
            done += 1
            continue
        try:
            frames = read_mp4(f"{VIDEOS}/{vid}.mp4")
        except Exception as e:
            print(f"SKIP {vid}: {e}", file=sys.stderr, flush=True)
            continue

        x = torch.from_numpy(frames.copy()).to(dev).float().div_(127.5).sub_(1.0)
        x = x.permute(3, 0, 1, 2).unsqueeze(0)  # [1,3,T,H,W]
        # Bicubic to the LR grid, exactly the conventional-SR degradation.
        t = x.shape[2]
        lr = F.interpolate(x.squeeze(0).permute(1, 0, 2, 3), size=(LR_H, LR_W),
                           mode="bicubic", align_corners=False).clamp(-1, 1)
        lr = lr.permute(1, 0, 2, 3).unsqueeze(0)  # [1,3,T,LR_H,LR_W]

        with torch.no_grad():
            z = vae.encode(lr).latent_dist.mode()
        z = ((z - mean) / std)[0].to(torch.bfloat16).cpu()

        torch.save(z, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        done += 1
        if done % 10 == 0 or n == 0:
            el = time.time() - t0
            print(f"[{args.shard}] {done}/{len(ids)} {vid} {tuple(z.shape)} "
                  f"{el/max(done,1):.1f}s/clip eta {(len(ids)-done)*el/max(done,1)/60:.0f}min",
                  flush=True)

    print(f"shard {args.shard} done: {done} clips in {(time.time()-t0)/60:.1f} min", flush=True)
    os._exit(0)


main()
