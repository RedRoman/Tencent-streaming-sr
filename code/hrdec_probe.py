"""Frame contract of the HR decoder, measured before anything is trained.

The TAE attempt failed once because the latent->frame correspondence was
assumed rather than measured: it trained 5 latents -> 5 frames while the real
decoder emits 4T-18 frames in 20-frame chunks. A student that maps the wrong
latents to the wrong frames can score well in training and be useless.

So: decode the real thing at several latent counts, and for each pixel frame
find which latent index it belongs to by checking the alignment directly.
"""
import json, os
import numpy as np
import torch
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
    print(f"latent {tuple(z.shape)}", flush=True)

    for n in (5, 10, 37):
        with torch.no_grad():
            w = z[:, :n].unsqueeze(0).to(dev, torch.float32) * std + mean
            f = vae.decode(w).sample.shape[2]
        print(f"  {n:2d} latents -> {f:3d} frames   ratio {f/n:.4f}", flush=True)
        torch.npu.empty_cache()

    # The cached pixels came from decoding ALL 37 latents at once. Check that a
    # 5-latent decode reproduces the first frames of it: if chunked decoding is
    # not equal to whole-clip decoding, the student's target is ill-defined.
    pix = np.load(f"data/pixels/hr/{vid}.npy", mmap_mode="r")
    with torch.no_grad():
        w = z[:, :5].unsqueeze(0).to(dev, torch.float32) * std + mean
        part = vae.decode(w).sample[0].permute(1, 0, 2, 3).float().clamp(-1, 1).cpu()
    ref = torch.from_numpy(np.asarray(pix[:part.shape[0]])).permute(0, 3, 1, 2).float()
    ref = ref.div(127.5).sub(1.0)
    mse = torch.nn.functional.mse_loss(part, ref)
    print(f"\nchunked(5) vs cached full decode: first {part.shape[0]} frames "
          f"PSNR {-10*torch.log10(mse/4):.2f} dB", flush=True)
    json.dump(dict(ok=True), open("results/hrdec_probe.json", "w"), indent=2)
    os._exit(0)


main()
