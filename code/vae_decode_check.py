"""Decode one stored HR latent through the real H3 VAE on NPU and write an MP4.

This is the acceptance path for Sol-H3: if the stored [24,37,48,84] latents do
not decode into the original 1344x768x124 video, every downstream pixel metric
would be measured against the wrong reference.

The decoder is a non-causal ViT, so this decodes the whole clip at once. Nothing
here claims a streaming decode.
"""
import os
import sys
import time

import torch

sys.path.insert(0, "/home/ma-user/modelarts/user-job-dir/sol_h3/.venv/lib/python3.12/site-packages")

import torch_npu  # noqa: F401
from diffusers import AutoencoderKLMiniMaxH3

VAE_DIR = "/home/ma-user/modelarts/user-job-dir/sol_h3/ckpts/h3-base/vae"
HIGH = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/high/video"
LOW = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/low/video"
OUT = "/home/ma-user/modelarts/user-job-dir/sol_h3/results/vae_decode_check"
SAMPLE = "H3O20261001_0001"


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = "npu:0"

    t0 = time.time()
    vae = AutoencoderKLMiniMaxH3.from_pretrained(VAE_DIR, torch_dtype=torch.float32)
    vae = vae.to(dev).eval()
    print(f"VAE loaded in {time.time()-t0:.1f}s", flush=True)
    print(f"  latent_channels={vae.config.latent_channels} "
          f"clip_length={vae.config.clip_length} token_drop={vae.config.token_drop}", flush=True)

    mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)

    for tag, src in (("high", HIGH), ("low", LOW)):
        z = torch.load(f"{src}/{SAMPLE}.pt.part", map_location="cpu", weights_only=False)
        print(f"\n[{tag}] stored normalized latent {tuple(z.shape)} {z.dtype}", flush=True)

        # Stored tensors are normalized; undo it for the VAE.
        z = z.unsqueeze(0).to(dev, torch.float32)
        z = (z * std + mean).to(torch.float32)

        t0 = time.time()
        with torch.no_grad():
            px = vae.decode(z).sample
        torch.npu.synchronize()
        dt = time.time() - t0

        px = px.float().clamp(-1, 1)
        print(f"[{tag}] decoded {tuple(px.shape)} in {dt:.1f}s "
              f"range[{px.min():.2f},{px.max():.2f}] peak={torch.npu.max_memory_allocated()/2**30:.1f}GiB",
              flush=True)

        frames = ((px[0].permute(1, 2, 3, 0) + 1) * 127.5).round().byte().cpu().numpy()
        import numpy as np
        np.save(f"{OUT}/{SAMPLE}_{tag}.npy", frames)
        print(f"[{tag}] saved frames {frames.shape} -> {OUT}/{SAMPLE}_{tag}.npy", flush=True)

    os._exit(0)


main()
