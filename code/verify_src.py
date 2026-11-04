"""Confirm the stored HR latents really are H3-VAE encodings of these MP4s.

If they are, the native decode of the latent should reconstruct the MP4 closely
(VAE round-trip error only). If they came from somewhere else, PSNR collapses
and the MP4s cannot be used as distillation targets.
"""
import os
import subprocess

import numpy as np
import torch
import torch_npu  # noqa: F401
from diffusers import AutoencoderKLMiniMaxH3

VAE_DIR = "/home/ma-user/modelarts/user-job-dir/sol_h3/ckpts/h3-base/vae"
HIGH = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/high/video"
VIDEOS = "/temp/yunfeng/datasets/h3_selected_800/videos"


def read_mp4(path, w=1344, h=768):
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"], capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.decode()[-400:])
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w, 3)


def main():
    dev = "npu:0"
    vae = AutoencoderKLMiniMaxH3.from_pretrained(VAE_DIR, torch_dtype=torch.float32).to(dev).eval()
    mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)

    ids = ["H3O20261001_0001", "H3O20261001_0002", "H3O20261001_0003"]
    for i, vid in enumerate(ids):
        z = torch.load(f"{HIGH}/{vid}.pt.part", map_location="cpu", weights_only=False)
        z = (z.unsqueeze(0).to(dev, torch.float32) * std + mean)
        with torch.no_grad():
            px = vae.decode(z).sample.float().clamp(-1, 1)
        dec = ((px[0].permute(1, 2, 3, 0) + 1) / 2).cpu().numpy()  # [T,H,W,3] in [0,1]

        src = read_mp4(f"{VIDEOS}/{vid}.mp4").astype(np.float32) / 255.0
        n = min(len(src), len(dec))
        a, b = dec[:n], src[:n]
        mse = float(((a - b) ** 2).mean())
        psnr = -10 * np.log10(mse + 1e-12)

        # Mismatched control: compare against a different video.
        other = read_mp4(f"{VIDEOS}/{ids[(i+1) % len(ids)]}.mp4").astype(np.float32) / 255.0
        m = min(n, len(other))
        psnr_x = -10 * np.log10(float(((a[:m] - other[:m]) ** 2).mean()) + 1e-12)

        print(f"{vid}  frames dec={len(dec)} src={len(src)}  "
              f"PSNR(decode vs its own mp4) = {psnr:6.2f} dB   "
              f"vs other mp4 = {psnr_x:5.2f} dB", flush=True)

    os._exit(0)


main()
