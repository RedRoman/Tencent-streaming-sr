"""Sol-H3 streaming with the producer and the GPU overlapped.

The serial version waits for 5 latents (833 ms of video) and only then spends
554 ms decoding+refining, so each chunk costs 1387 ms against an 833 ms budget
and the stream falls behind. The work itself only needs 66% of the budget --
what is missing is overlap.

Here a producer thread emits latents on the playback clock into a queue while
the main thread consumes whole chunks, so generation of chunk N+1 happens during
processing of chunk N. Steady-state cost per chunk becomes max(833, 554) = 833.
"""
from __future__ import annotations

import argparse, json, os, queue, threading, time
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401
from diffusers import AutoencoderKLMiniMaxH3

from pixel_model import PixelSR

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
LOW = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/low/video"
VAE_DIR = str(ROOT / "ckpts/h3-base/vae")
FPS = 24
# The H3 VAE emits 3.4 pixel frames per latent, not 4: `token_drop=3` is charged
# against every clip, and the ratio is exactly 3.40 at 5/10/15/35 latents, so
# chunked decoding loses nothing -- 4 frames/latent was simply the wrong constant.
# Using 4 made the producer feed too slowly and showed a fake -0.125 s/chunk drift,
# and it inflated the per-chunk budget from 708 ms to 833 ms.
FRAMES_PER_LATENT = 3.4
MS_PER_LATENT = 1000 * FRAMES_PER_LATENT / FPS
CHUNK = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--latents", type=int, default=35)
    ap.add_argument("--clip", default=None)
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

    vid = args.clip or sorted(f[:-8] for f in os.listdir(LOW) if f.endswith(".pt.part"))[0]
    z_all = torch.load(f"{LOW}/{vid}.pt.part", map_location="cpu", weights_only=False)
    n_lat = min(args.latents, z_all.shape[1])

    with torch.no_grad():
        w = (z_all[:, :CHUNK].unsqueeze(0).to(dev, torch.float32) * std + mean)
        p = vae.decode(w).sample
        with torch.autocast("npu", dtype=torch.bfloat16):
            sr(p[0].permute(1, 0, 2, 3)[:1].float().clamp(-1, 1), frames=1)
    torch.npu.synchronize()
    print(f"clip {vid}  {n_lat} latents = {n_lat*FRAMES_PER_LATENT/FPS:.1f}s video; pipelined\n", flush=True)

    q: queue.Queue = queue.Queue(maxsize=4)
    t_start = time.time()

    def producer():
        """Emit latents on the playback clock, batched into decoder chunks."""
        buf = []
        for i in range(n_lat):
            due = t_start + i * MS_PER_LATENT / 1000
            now = time.time()
            if now < due:
                time.sleep(due - now)
            buf.append(z_all[:, i])
            if len(buf) == CHUNK:
                q.put((torch.stack(buf, dim=1), time.time() - t_start))
                buf = []
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()

    events, emitted, t_first, frames_out = [], 0, None, []
    while True:
        item = q.get()
        if item is None:
            break
        zc, t_ready = item
        t0 = time.time()
        z = zc.unsqueeze(0).to(dev, torch.float32)
        with torch.no_grad():
            lr_px = vae.decode(z * std + mean).sample.float().clamp(-1, 1)
        torch.npu.synchronize(); t_dec = time.time() - t0

        t0 = time.time()
        lr_frames = lr_px[0].permute(1, 0, 2, 3)
        outs = []
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            for f in range(lr_frames.shape[0]):
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
        events.append(dict(chunk=len(events), frames=emitted,
                           latents_ready_s=round(t_ready, 2),
                           decode_ms=round(t_dec * 1000, 1), refine_ms=round(t_sr * 1000, 1),
                           wall_s=round(wall, 2), video_s=round(emitted / FPS, 2),
                           lead_s=round(emitted / FPS - wall, 2),
                           queue_depth=q.qsize()))
        print(json.dumps(events[-1]), flush=True)
        del lr_px, lr_frames, outs, hr
        torch.npu.empty_cache()

    total = time.time() - t_start
    per_frame = total / max(emitted, 1) * 1000
    summary = dict(clip=vid, frames=emitted, ttff_ms=round(t_first * 1000, 1),
                   wall_s=round(total, 2), video_s=round(emitted / FPS, 2),
                   ms_per_frame=round(per_frame, 2),
                   realtime_x=round((1000 / FPS) / per_frame, 2),
                   final_lead_s=events[-1]["lead_s"] if events else None,
                   underruns=sum(1 for e in events if e["lead_s"] < -0.05),
                   mean_decode_ms=round(sum(e["decode_ms"] for e in events) / len(events), 1),
                   mean_refine_ms=round(sum(e["refine_ms"] for e in events) / len(events), 1))
    print("\nSTREAM " + json.dumps(summary), flush=True)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        np.save(args.out / f"{vid}_stream.npy", np.concatenate(frames_out))
        (args.out / "stream.json").write_text(json.dumps(dict(summary=summary, events=events), indent=2))
        print(f"wrote {args.out}", flush=True)
    os._exit(0)


main()
