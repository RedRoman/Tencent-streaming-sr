"""Run Sol-H3 as an actual stream and measure what a viewer would experience.

Everything so far was measured as isolated stages on precomputed tensors. This
drives the real chain under a clock: LR latents arrive at playback rate, and the
pipeline must decode, refine and emit frames fast enough to keep up.

  LR latent arrives (1 per 166.7 ms of video)
    -> buffer until the H3 decoder's minimum chunk of 5 latents
    -> H3 VAE decode   5 latents -> 17 LR frames   (non-causal WITHIN the chunk)
    -> pixel refiner   per frame, W=1               (strictly causal)
    -> emit

Reported: time to first frame, sustained throughput, and whether the output
clock ever falls behind the input clock (an underrun = visible stall).
"""
from __future__ import annotations

import argparse, json, os, sys, time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from diffusers import AutoencoderKLMiniMaxH3

from pixel_model import PixelSR

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
LOW = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/low/video"
VAE_DIR = str(ROOT / "ckpts/h3-base/vae")
FPS = 24
MS_PER_LATENT = 1000 * 4 / FPS      # one latent frame carries 4 pixel frames
CHUNK = 5                            # H3 decoder minimum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--clip", default=None)
    ap.add_argument("--latents", type=int, default=35)
    ap.add_argument("--realtime", action="store_true",
                    help="pace the producer at playback rate instead of free-running")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    dev = "npu:0"
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = blob["args"]
    sr = PixelSR(a["width"], a["blocks"], 3, blob["taps"]).to(dev).eval()
    sr.load_state_dict(blob["model"])
    vae = AutoencoderKLMiniMaxH3.from_pretrained(VAE_DIR, torch_dtype=torch.float32).to(dev).eval()
    mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
    print(f"refiner window={a['window']} params={sum(p.numel() for p in sr.parameters())/1e6:.2f}M",
          flush=True)

    vid = args.clip or sorted(f[:-8] for f in os.listdir(LOW) if f.endswith(".pt.part"))[0]
    z_all = torch.load(f"{LOW}/{vid}.pt.part", map_location="cpu", weights_only=False)
    n_lat = min(args.latents, z_all.shape[1])
    print(f"clip {vid}  {n_lat} latent frames = {n_lat*4/FPS:.1f} s of video", flush=True)

    # Warm the kernels so the first chunk is not dominated by compilation.
    with torch.no_grad():
        warm = (z_all[:, :CHUNK].unsqueeze(0).to(dev, torch.float32) * std + mean)
        px = vae.decode(warm).sample
        with torch.autocast("npu", dtype=torch.bfloat16):
            sr(px[0].permute(1, 0, 2, 3)[:1].float().clamp(-1, 1), frames=1)
    torch.npu.synchronize()
    print("warmed up\n", flush=True)

    buf, emitted, events = deque(), 0, []
    t_start = time.time()
    t_first = None
    frames_out = []

    for i in range(n_lat):
        if args.realtime:
            # The producer cannot run ahead of playback.
            due = t_start + i * MS_PER_LATENT / 1000
            if time.time() < due:
                time.sleep(due - time.time())
        buf.append(z_all[:, i])
        if len(buf) < CHUNK:
            continue

        t0 = time.time()
        z = torch.stack(list(buf), dim=1).unsqueeze(0).to(dev, torch.float32)
        buf.clear()
        with torch.no_grad():
            lr_px = vae.decode(z * std + mean).sample.float().clamp(-1, 1)
        torch.npu.synchronize(); t_dec = time.time() - t0

        t0 = time.time()
        lr_frames = lr_px[0].permute(1, 0, 2, 3)          # [17,3,256,448]
        outs = []
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            for f in range(lr_frames.shape[0]):           # strictly per frame
                outs.append(sr(lr_frames[f:f + 1], frames=1).float())
        torch.npu.synchronize(); t_sr = time.time() - t0

        hr = torch.cat(outs)
        emitted += hr.shape[0]
        if t_first is None:
            t_first = time.time() - t_start
        if args.out is not None:
            frames_out.append(((hr.permute(0, 2, 3, 1) + 1) * 127.5)
                              .round().clamp(0, 255).byte().cpu().numpy())

        wall = time.time() - t_start
        video_t = emitted / FPS
        events.append(dict(chunk=len(events), latents=i + 1, frames=emitted,
                           decode_ms=round(t_dec * 1000, 1), refine_ms=round(t_sr * 1000, 1),
                           chunk_ms=round((t_dec + t_sr) * 1000, 1),
                           wall_s=round(wall, 2), video_s=round(video_t, 2),
                           lead_s=round(video_t - wall, 2)))
        print(json.dumps(events[-1]), flush=True)
        del lr_px, lr_frames, outs, hr
        torch.npu.empty_cache()

    total = time.time() - t_start
    per_frame = total / max(emitted, 1) * 1000
    underruns = sum(1 for e in events if e["lead_s"] < 0)
    summary = dict(clip=vid, latents=n_lat, frames_emitted=emitted,
                   ttff_ms=round(t_first * 1000, 1),
                   wall_s=round(total, 2), video_s=round(emitted / FPS, 2),
                   ms_per_frame=round(per_frame, 2),
                   realtime_x=round((1000 / FPS) / per_frame, 2),
                   mean_decode_ms=round(sum(e["decode_ms"] for e in events) / len(events), 1),
                   mean_refine_ms=round(sum(e["refine_ms"] for e in events) / len(events), 1),
                   chunks=len(events), underruns=underruns,
                   paced=bool(args.realtime))
    print("\nSTREAM " + json.dumps(summary), flush=True)

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        np.save(args.out / f"{vid}_stream.npy", np.concatenate(frames_out))
        (args.out / "stream.json").write_text(json.dumps(dict(summary=summary, events=events), indent=2))
        print(f"wrote {args.out}", flush=True)
    os._exit(0)


main()
