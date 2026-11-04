"""Latency budget for streaming Sol-H3: what does a short refiner window buy?

Latency, not quality, is what the short-window design is for, so measure it
directly and independently of training (wall time does not depend on weights).

Reference budget: H3 emits 24 fps and its VAE compresses time 4x, so one latent
frame carries 4 pixel frames = 166.7 ms of video. A stage keeps up with realtime
iff it produces one latent frame in under that.

Reported per stage:
  chunk_ms        wall time to produce one chunk
  ms_per_latent   chunk_ms / W, the sustained cost
  realtime_x      166.7 / ms_per_latent; >1 means faster than playback
  ttff_ms         time to first output: a W-frame refiner cannot emit anything
                  until W latent frames of LR exist, so waiting for them costs
                  (W-1)*166.7 ms on top of compute
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch_npu  # noqa: F401

from diffusers import AutoencoderKLMiniMaxH3
from solh3_model import SolH3Refiner

VAE_DIR = "/home/ma-user/modelarts/user-job-dir/sol_h3/ckpts/h3-base/vae"
MS_PER_LATENT_FRAME = 1000.0 * 4 / 24  # 166.7 ms of video per latent frame


def timeit(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / iters * 1000


@torch.no_grad()
def refiner_latency(device, windows, nfes, width=512):
    model = SolH3Refiner(width=width, time_taps=5, cond_mode="concat").to(device).eval()
    rows = []
    for w in windows:
        hr = torch.randn(1, 24, w, 48, 84, device=device)
        lr = torch.randn(1, 24, w, 48, 84, device=device)
        for nfe in nfes:
            def run():
                x = hr
                for i in range(nfe):
                    t = torch.full((1,), 1.0 - i / nfe, device=device)
                    with torch.autocast("npu", dtype=torch.bfloat16):
                        v = model(x, lr, t).float()
                    x = x - v / nfe
                return x
            ms = timeit(run)
            per_latent = ms / w
            rows.append(dict(stage="refiner", window=w, nfe=nfe,
                             chunk_ms=round(ms, 1),
                             ms_per_latent=round(per_latent, 1),
                             realtime_x=round(MS_PER_LATENT_FRAME / per_latent, 2),
                             ttff_ms=round((w - 1) * MS_PER_LATENT_FRAME + ms, 1)))
            print(json.dumps(rows[-1]), flush=True)
    del model
    torch.npu.empty_cache()
    return rows


@torch.no_grad()
def decoder_latency(device, windows):
    """The native H3 decoder is a non-causal ViT over the whole clip; measure
    what it costs at each chunk length and whether short chunks are even valid."""
    vae = AutoencoderKLMiniMaxH3.from_pretrained(VAE_DIR, torch_dtype=torch.float32).to(device).eval()
    rows = []
    for w in windows:
        z = torch.randn(1, 24, w, 48, 84, device=device)
        try:
            out = vae.decode(z).sample
            n_px = out.shape[2]
            ms = timeit(lambda: vae.decode(z).sample, warmup=1, iters=3)
            per_latent = ms / w
            rows.append(dict(stage="vae_decode", window=w, pixel_frames=n_px,
                             chunk_ms=round(ms, 1),
                             ms_per_latent=round(per_latent, 1),
                             realtime_x=round(MS_PER_LATENT_FRAME / per_latent, 2),
                             peak_gib=round(torch.npu.max_memory_allocated() / 2**30, 2)))
        except Exception as e:
            rows.append(dict(stage="vae_decode", window=w, error=str(e)[:140]))
        print(json.dumps(rows[-1]), flush=True)
        torch.npu.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 5, 13, 37])
    ap.add_argument("--nfes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--out", default="results/latency")
    args = ap.parse_args()

    device = "npu:0"
    os.makedirs(args.out, exist_ok=True)
    print(f"realtime budget: {MS_PER_LATENT_FRAME:.1f} ms per latent frame "
          f"(4 pixel frames at 24 fps)\n", flush=True)

    print("== refiner", flush=True)
    rows = refiner_latency(device, args.windows, args.nfes)
    print("\n== native VAE decoder", flush=True)
    rows += decoder_latency(device, args.windows)

    with open(f"{args.out}/latency.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {args.out}/latency.json", flush=True)
    os._exit(0)


main()
