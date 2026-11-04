"""Can the native H3 decoder be driven chunk by chunk?

The decoder is a non-causal ViT over the whole clip, so a streaming system would
have to feed it windows. Measure what that costs in quality: decode the full 37
latent frames at once (the reference), then in windows of 5/9/13 with varying
overlap, and compare the overlapping-region pixels.

If small windows reproduce the full decode closely, streaming pixel output is
possible with the native decoder and a tiny decoder only needs to beat its
speed. If they do not, a causal/tiny decoder is mandatory, not an optimization.
"""
import os

import torch
import torch_npu  # noqa: F401
from diffusers import AutoencoderKLMiniMaxH3

VAE_DIR = "/home/ma-user/modelarts/user-job-dir/sol_h3/ckpts/h3-base/vae"
HIGH = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/high/video"
VID = "H3O20261001_0001"


def psnr(a, b):
    return float(-10 * torch.log10(((a - b) ** 2).mean() + 1e-12))


def main():
    dev = "npu:0"
    vae = AutoencoderKLMiniMaxH3.from_pretrained(VAE_DIR, torch_dtype=torch.float32).to(dev).eval()
    mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)

    z = torch.load(f"{HIGH}/{VID}.pt.part", map_location="cpu", weights_only=False)
    z = (z.unsqueeze(0).to(dev, torch.float32) * std + mean)
    T = z.shape[2]

    with torch.no_grad():
        full = vae.decode(z).sample.float().clamp(-1, 1)
    print(f"full decode: latents {T} -> pixels {full.shape[2]}", flush=True)
    px_per_lat = full.shape[2] / T

    for win in (5, 9, 13):
        for ov in (0, 2, 4):
            if ov >= win:
                continue
            step = win - ov
            pieces, ok = [], True
            with torch.no_grad():
                for s in range(0, T, step):
                    e = min(s + win, T)
                    if e - s < 5:  # decoder needs >=5 latent frames
                        break
                    try:
                        out = vae.decode(z[:, :, s:e]).sample.float().clamp(-1, 1)
                    except Exception as ex:
                        print(f"  win={win} ov={ov}: FAILED {str(ex)[:80]}", flush=True)
                        ok = False
                        break
                    # Drop the pixels that correspond to the overlapped latents.
                    drop = 0 if s == 0 else int(ov * px_per_lat)
                    pieces.append(out[:, :, drop:])
                    if e >= T:
                        break
            if not ok or not pieces:
                continue
            joined = torch.cat(pieces, dim=2)
            n = min(joined.shape[2], full.shape[2])
            print(f"  win={win:2d} ov={ov}: pixels {joined.shape[2]:3d} vs {full.shape[2]} "
                  f"PSNR vs full = {psnr(joined[:, :, :n], full[:, :, :n]):6.2f} dB", flush=True)

    os._exit(0)


main()
