"""What does one HR keyframe actually cost?

The relay's gain was measured with a GT keyframe. Deploying it means H3 has to
produce that keyframe for real, and the whole scheme lives or dies on whether
one keyframe fits in the interval's spare budget. Two components:

  1. HR VAE decode -- 1176 ms/latent was measured on a long clip. A keyframe is
     ONE latent, and the decoder works in chunks, so the minimum may be much
     less (or much more, if the chunk size forces a fixed cost).
  2. the H3 transformer forward at HR -- measured separately; needs the 62 GB
     model, so it is reported as feasible/not rather than guessed at.

Budget: at a 30-frame interval the keyframe must be ready within 1250 ms of
video, minus what decode and refine already consume on their own cards.
"""
import json, os, time
import torch
import torch_npu  # noqa: F401
from diffusers import AutoencoderKLMiniMaxH3

VAE_DIR = "ckpts/h3-base/vae"
LOW = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/low/video"
HIGH = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/high/video"
dev = "npu:0"


def tm(fn, warm=1, it=3):
    for _ in range(warm):
        fn()
    torch.npu.synchronize(); t = time.time()
    for _ in range(it):
        fn()
    torch.npu.synchronize()
    return (time.time() - t) / it * 1000


def main():
    vae = AutoencoderKLMiniMaxH3.from_pretrained(VAE_DIR, torch_dtype=torch.float32).to(dev).eval()
    mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
    vid = "H3O20261001_0001"
    zl = torch.load(f"{LOW}/{vid}.pt.part", map_location="cpu", weights_only=False)
    zh = torch.load(f"{HIGH}/{vid}.pt.part", map_location="cpu", weights_only=False)
    print(f"LR latent {tuple(zl.shape)}   HR latent {tuple(zh.shape)}", flush=True)

    rows = []
    for tag, z in (("LR 448x256", zl), ("HR 1344x768", zh)):
        for n in (1, 2, 5):
            try:
                w = z[:, :n].unsqueeze(0).to(dev, torch.float32) * std + mean
                with torch.no_grad():
                    frames = vae.decode(w).sample.shape[2]
                    ms = tm(lambda: vae.decode(w))
                rows.append(dict(grid=tag, latents=n, frames=frames,
                                 ms=round(ms, 1), ms_per_latent=round(ms / n, 1)))
                print(f"{tag:14s} {n} latent -> {frames:3d} frames  {ms:8.1f} ms"
                      f"  ({ms/n:.1f} ms/latent)", flush=True)
                torch.npu.empty_cache()
            except Exception as e:
                print(f"{tag:14s} {n} latent  FAILED {str(e)[:80]}", flush=True)
                torch.npu.empty_cache()
    json.dump(rows, open("results/keyframe_cost.json", "w"), indent=2)
    os._exit(0)


main()
