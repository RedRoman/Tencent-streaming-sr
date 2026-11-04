"""Which cached pixel frames does chunk k of latents correspond to?

Chunk 0 matched frames 0..16 at 41.9 dB. That does not prove chunk k matches
frames 17k..17k+16: the decoder applies `frame_pre_padding` and a token_drop
tail, which can treat the first chunk specially. Training on a wrong mapping is
the exact failure that killed the TAE attempt, so measure it instead of
assuming: decode latents [5k:5k+5] and score it against several candidate
offsets, then report the best.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from diffusers import AutoencoderKLMiniMaxH3

dev = "npu:0"
HIGH = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/high/video"


def main():
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        "ckpts/h3-base/vae", torch_dtype=torch.float32).to(dev).eval()
    mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
    vid = "H3O20261001_0001"
    z = torch.load(f"{HIGH}/{vid}.pt.part", map_location="cpu", weights_only=False)
    pix = np.load(f"data/pixels/hr/{vid}.npy", mmap_mode="r")

    for k in (0, 1, 2, 4):
        with torch.no_grad():
            w = z[:, 5 * k:5 * k + 5].unsqueeze(0).to(dev, torch.float32) * std + mean
            out = vae.decode(w).sample[0].permute(1, 0, 2, 3).float().clamp(-1, 1).cpu()
        T = out.shape[0]
        best = (None, -1e9)
        for off in range(max(17 * k - 6, 0), 17 * k + 7):
            if off + T > pix.shape[0]:
                continue
            ref = torch.from_numpy(np.asarray(pix[off:off + T])).permute(0, 3, 1, 2)
            ref = ref.float().div(127.5).sub(1.0)
            p = float(-10 * torch.log10(F.mse_loss(out, ref) / 4 + 1e-12))
            if p > best[1]:
                best = (off, p)
        print(f"chunk {k}: latents[{5*k}:{5*k+5}] -> {T} frames, "
              f"best offset {best[0]} (expected {17*k})  PSNR {best[1]:.2f} dB", flush=True)
        torch.npu.empty_cache()
    os._exit(0)


main()
