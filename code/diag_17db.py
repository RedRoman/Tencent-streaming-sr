"""Why is decode(latent) only ~17 dB from the source MP4?

A VAE round-trip should land near 30 dB, so 17 dB means either the MP4 is not
the latent's source, or the two are misaligned. Test the cheap explanations:
frame offset, channel order, and limited/full colour range.
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
VID = "H3O20261001_0001"


def read_mp4(path, extra=()):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24",
           *extra, "-"]
    p = subprocess.run(cmd, capture_output=True)
    p.check_returncode()
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, 768, 1344, 3).astype(np.float32) / 255


def psnr(a, b):
    return -10 * np.log10(float(((a - b) ** 2).mean()) + 1e-12)


def main():
    dev = "npu:0"
    vae = AutoencoderKLMiniMaxH3.from_pretrained(VAE_DIR, torch_dtype=torch.float32).to(dev).eval()
    mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)

    z = torch.load(f"{HIGH}/{VID}.pt.part", map_location="cpu", weights_only=False)
    z = (z.unsqueeze(0).to(dev, torch.float32) * std + mean)
    with torch.no_grad():
        px = vae.decode(z).sample.float().clamp(-1, 1)
    dec = ((px[0].permute(1, 2, 3, 0) + 1) / 2).cpu().numpy()

    src = read_mp4(f"{VIDEOS}/{VID}.mp4")
    print(f"dec {dec.shape}  src {src.shape}")
    print(f"dec mean {dec.mean():.4f} std {dec.std():.4f} | src mean {src.mean():.4f} std {src.std():.4f}")

    print("\n-- frame offset sweep (dec[t+o] vs src[t])")
    for o in range(-4, 5):
        if o >= 0:
            a, b = dec[o:], src[: len(dec) - o]
        else:
            a, b = dec[: o], src[-o:]
        n = min(len(a), len(b))
        print(f"   offset {o:+d}: {psnr(a[:n], b[:n]):6.2f} dB")

    print("\n-- channel permutation (full overlap)")
    n = min(len(dec), len(src))
    for name, idx in (("RGB", [0, 1, 2]), ("BGR", [2, 1, 0]), ("GRB", [1, 0, 2])):
        print(f"   {name}: {psnr(dec[:n][..., idx], src[:n]):6.2f} dB")

    print("\n-- per-frame PSNR, first and last few")
    pf = [psnr(dec[i], src[i]) for i in range(n)]
    print("   first5", [f"{v:.1f}" for v in pf[:5]])
    print("   last5 ", [f"{v:.1f}" for v in pf[-5:]])
    print(f"   min {min(pf):.1f} @frame {int(np.argmin(pf))}  max {max(pf):.1f} @frame {int(np.argmax(pf))}")

    print("\n-- best affine rescale of dec onto src (fixes range/gamma mismatch)")
    a, b = dec[:n].reshape(-1), src[:n].reshape(-1)
    A = np.stack([a, np.ones_like(a)], 1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    print(f"   scale {sol[0]:.4f} bias {sol[1]:+.4f} -> {psnr(a*sol[0]+sol[1], b):6.2f} dB")

    os._exit(0)


main()
