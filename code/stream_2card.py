"""Two-stage streaming: H3 decode and the refiner on separate NPUs.

The one-card driver runs decode and refine back to back inside each chunk, so
the chunk period is their sum (424 + refine). That is the model under which I
said w192 does not fit. But the two stages are independent -- decode of chunk
N+1 does not depend on refine of chunk N -- so they can run concurrently. On a
single NPU that buys nothing (both saturate the same cores); on two NPUs the
chunk period becomes max(424, refine) instead of the sum, which is what the
"pipelined" budget in cap_probe.py assumes.

This measures whether that is real: decode on --dec-dev, refine on --sr-dev,
connected by a bounded queue, with the producer still paced to the playback
clock. Deployment-wise this is honest -- H3 itself already occupies several
cards, so giving the refiner its own card is a provisioning choice, not a trick.
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
    ap.add_argument("--dec-dev", default="npu:4")
    ap.add_argument("--sr-dev", default="npu:5")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    vid = args.clip or sorted(f[:-8] for f in os.listdir(LOW) if f.endswith(".pt.part"))[0]
    z_all = torch.load(f"{LOW}/{vid}.pt.part", map_location="cpu", weights_only=False)
    n_lat = min(args.latents, z_all.shape[1])

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = blob["args"]

    q_lat: queue.Queue = queue.Queue(maxsize=3)
    q_px: queue.Queue = queue.Queue(maxsize=2)
    ready = threading.Barrier(3)
    t_start = 0.0
    dec_ms, sr_ms = [], []

    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        VAE_DIR, torch_dtype=torch.float32).to(args.dec_dev).eval()
    mean = torch.tensor(vae.config.latents_mean, device=args.dec_dev).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=args.dec_dev).view(1, -1, 1, 1, 1)
    sr = PixelSR(a["width"], a["blocks"], 3, blob["taps"]).to(args.sr_dev).eval()
    sr.load_state_dict(blob["model"])

    def decoder():
        torch.npu.set_device(args.dec_dev)
        with torch.no_grad():  # warm the kernels before the clock starts
            vae.decode(z_all[:, :CHUNK].unsqueeze(0).to(args.dec_dev, torch.float32) * std + mean)
        torch.npu.synchronize()
        ready.wait()
        while True:
            item = q_lat.get()
            if item is None:
                q_px.put(None); return
            t0 = time.time()
            with torch.no_grad():
                px = vae.decode(item.unsqueeze(0).to(args.dec_dev, torch.float32) * std + mean).sample
            px = px[0].permute(1, 0, 2, 3).float().clamp(-1, 1).cpu()
            torch.npu.synchronize()
            dec_ms.append((time.time() - t0) * 1000)
            q_px.put(px)

    results: list = []

    def refiner():
        torch.npu.set_device(args.sr_dev)
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            sr(torch.randn(1, 3, 256, 448, device=args.sr_dev), frames=1)
        torch.npu.synchronize()
        ready.wait()
        emitted = 0
        while True:
            px = q_px.get()
            if px is None:
                return
            t0 = time.time()
            outs = []
            with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                for f in range(px.shape[0]):
                    outs.append(sr(px[f:f + 1].to(args.sr_dev), frames=1).float())
            torch.npu.synchronize()
            sr_ms.append((time.time() - t0) * 1000)
            hr = torch.cat(outs)
            emitted += hr.shape[0]
            wall = time.time() - t_start
            results.append(dict(chunk=len(results), frames=emitted,
                                decode_ms=round(dec_ms[-1], 1),
                                refine_ms=round(sr_ms[-1], 1),
                                wall_s=round(wall, 2), video_s=round(emitted / FPS, 2),
                                lead_s=round(emitted / FPS - wall, 2),
                                q_lat=q_lat.qsize(), q_px=q_px.qsize()))
            print(json.dumps(results[-1]), flush=True)
            if args.out is not None:
                results[-1]["_px"] = ((hr.permute(0, 2, 3, 1) + 1) * 127.5
                                      ).round().clamp(0, 255).byte().cpu().numpy()
            del outs, hr
            torch.npu.empty_cache()

    td = threading.Thread(target=decoder, daemon=True)
    tr = threading.Thread(target=refiner, daemon=True)
    td.start(); tr.start()
    ready.wait()  # both stages warmed; start the playback clock now
    t_start = time.time()
    print(f"clip {vid}  {n_lat} latents = {n_lat*FRAMES_PER_LATENT/FPS:.1f}s video; "
          f"decode@{args.dec_dev} refine@{args.sr_dev} w{a['width']}x{a['blocks']}\n", flush=True)

    buf = []
    for i in range(n_lat):
        due = t_start + i * MS_PER_LATENT / 1000
        now = time.time()
        if now < due:
            time.sleep(due - now)
        buf.append(z_all[:, i])
        if len(buf) == CHUNK:
            q_lat.put(torch.stack(buf, dim=1)); buf = []
    q_lat.put(None)
    tr.join()

    total = time.time() - t_start
    emitted = results[-1]["frames"] if results else 0
    per_frame = total / max(emitted, 1) * 1000
    n = len(results)
    summary = dict(clip=vid, width=a["width"], blocks=a["blocks"],
                   frames=emitted, chunks=n,
                   wall_s=round(total, 2), video_s=round(emitted / FPS, 2),
                   ms_per_frame=round(per_frame, 2),
                   mean_decode_ms=round(sum(dec_ms) / len(dec_ms), 1),
                   mean_refine_ms=round(sum(sr_ms) / len(sr_ms), 1),
                   # steady state: skip the first two chunks (queue fill-up)
                   chunk_period_ms=round((results[-1]["wall_s"] - results[1]["wall_s"])
                                         * 1000 / max(n - 2, 1), 1) if n > 2 else None,
                   first_lead_s=results[0]["lead_s"] if results else None,
                   final_lead_s=results[-1]["lead_s"] if results else None,
                   underruns=sum(1 for r in results[2:] if r["lead_s"] < results[1]["lead_s"] - 0.1))
    print("\nSTREAM2 " + json.dumps(summary), flush=True)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        np.save(args.out / f"{vid}_stream.npy",
                np.concatenate([r.pop("_px") for r in results]))
        (args.out / "stream.json").write_text(json.dumps(dict(summary=summary, events=results), indent=2))
        print(f"wrote {args.out}", flush=True)
    os._exit(0)


main()
