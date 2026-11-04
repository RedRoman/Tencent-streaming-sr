"""Where do the 5847 ms of an HR decode go, and how much of it is prunable?

Any latent-domain SR route ends with an HR latent that still has to become
pixels, and that decode is 8.3x over the whole chunk budget. So it is the gate:
no point improving latent SR until this is either cheap or proven cheap-able.

Two questions, in order:
  1. module-level time -- is it concentrated (prunable) or spread (not)?
  2. how the cost scales with spatial size and with latent count, which says
     whether the 5-latent minimum is the problem or the resolution is.
"""
import json, os, time
from collections import defaultdict
import torch
import torch_npu  # noqa: F401
from diffusers import AutoencoderKLMiniMaxH3

VAE_DIR = "ckpts/h3-base/vae"
HIGH = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/high/video"
LOW = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/low/video"
dev = "npu:0"


def main():
    vae = AutoencoderKLMiniMaxH3.from_pretrained(VAE_DIR, torch_dtype=torch.float32).to(dev).eval()
    mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
    dec = vae.decoder
    n = sum(p.numel() for p in dec.parameters())
    print(f"decoder params {n/1e6:.1f}M   total vae {sum(p.numel() for p in vae.parameters())/1e6:.1f}M",
          flush=True)

    zh = torch.load(f"{HIGH}/H3O20261001_0001.pt.part", map_location="cpu", weights_only=False)
    zl = torch.load(f"{LOW}/H3O20261001_0001.pt.part", map_location="cpu", weights_only=False)

    # ---- 1. module-level timing, by immediate child of the decoder ----
    acc = defaultdict(float)
    hooks = []
    stack = []

    def pre(m, i):
        torch.npu.synchronize(); stack.append(time.time())

    def post(m, i, o):
        torch.npu.synchronize(); acc[m._prof_name] += time.time() - stack.pop()

    for name, m in dec.named_children():
        m._prof_name = name
        hooks.append(m.register_forward_pre_hook(pre))
        hooks.append(m.register_forward_hook(post))

    w = zh[:, :5].unsqueeze(0).to(dev, torch.float32) * std + mean
    with torch.no_grad():
        vae.decode(w)                      # warm
    acc.clear()
    t0 = time.time()
    with torch.no_grad():
        vae.decode(w)
    torch.npu.synchronize()
    total = (time.time() - t0) * 1000
    for h in hooks:
        h.remove()
    print(f"\nHR 5-latent decode total {total:.0f} ms", flush=True)
    rows = sorted(acc.items(), key=lambda kv: -kv[1])
    for k, v in rows:
        print(f"  {k:28s} {v*1000:8.1f} ms  {v*1000/total*100:5.1f}%", flush=True)
    covered = sum(acc.values()) * 1000
    print(f"  {'(unattributed)':28s} {total-covered:8.1f} ms", flush=True)

    # ---- 2. how it scales ----
    print("\nscaling:", flush=True)
    out = []
    for tag, z in (("LR 16x28", zl), ("HR 48x84", zh)):
        for nl in (5, 10):
            if nl > z.shape[1]:
                continue
            w = z[:, :nl].unsqueeze(0).to(dev, torch.float32) * std + mean
            with torch.no_grad():
                vae.decode(w)
                torch.npu.synchronize(); t = time.time()
                f = vae.decode(w).sample.shape[2]
                torch.npu.synchronize()
            ms = (time.time() - t) * 1000
            out.append(dict(grid=tag, latents=nl, frames=f, ms=round(ms, 1)))
            print(f"  {tag}  {nl:2d} latent -> {f:3d} frames  {ms:8.1f} ms"
                  f"  ({ms/nl:7.1f} ms/latent)", flush=True)
            torch.npu.empty_cache()
    json.dump(dict(decoder_params_m=round(n/1e6, 1),
                   modules={k: round(v*1000, 1) for k, v in rows},
                   total_ms=round(total, 1), scaling=out),
              open("results/vae_profile.json", "w"), indent=2)
    os._exit(0)


main()
