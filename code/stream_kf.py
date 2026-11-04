"""End-to-end streaming for the keyframe build of Sol-H3.

The two-card driver proved decode and refine overlap. This adds the third
stage: every `--kf-interval` frames the refiner switches to a fresh HR latent
keyframe, re-encodes it once, and reuses that encoding for the whole interval.

What this measures that the offline eval cannot:
  - whether the keyframe switch costs anything at the seam (encode is 1.1 ms,
    but it lands inside one chunk's budget, not spread over the interval)
  - whether the chunk period still tracks the playback clock with the extra work
  - cold start: the first interval has a keyframe at distance 0, later ones do
    not, so the quality is not uniform across the stream

The H3-side cost of PRODUCING the keyframe latent is not simulated here -- the
latent is read from disk, as the LR stream is. That cost is measured separately
(it is a marginal forward on the H3 instance already running), and is reported
with this run rather than folded into it.
"""
from __future__ import annotations

import argparse, json, os, queue, threading, time
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401
from diffusers import AutoencoderKLMiniMaxH3

from pixel_model import PixelSR
from pixel_train import hr_latent, FRAMES_PER_LAT, TEXT_DIM

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
LOW = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/low/video"
VAE_DIR = str(ROOT / "ckpts/h3-base/vae")
FPS = 24
FRAMES_PER_LATENT = 3.4          # measured: token_drop=3 is charged per clip
MS_PER_LATENT = 1000 * FRAMES_PER_LATENT / FPS
CHUNK = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--latents", type=int, default=35)
    ap.add_argument("--clip", default=None)
    ap.add_argument("--kf-interval", type=int, default=30)
    ap.add_argument("--dec-dev", default="npu:0")
    ap.add_argument("--sr-dev", default="npu:1")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    vid = args.clip or sorted(f[:-8] for f in os.listdir(LOW) if f.endswith(".pt.part"))[0]
    z_all = torch.load(f"{LOW}/{vid}.pt.part", map_location="cpu", weights_only=False)
    n_lat = min(args.latents, z_all.shape[1])
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = blob["args"]
    has_kf = bool(a.get("ref_max"))
    zref = hr_latent(vid) if has_kf else None

    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        VAE_DIR, torch_dtype=torch.float32).to(args.dec_dev).eval()
    mean = torch.tensor(vae.config.latents_mean, device=args.dec_dev).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=args.dec_dev).view(1, -1, 1, 1, 1)
    sr = PixelSR(a["width"], a["blocks"], 3, blob["taps"],
                 text_dim=TEXT_DIM if a.get("text") else 0,
                 text_sites=max(a.get("text") or 1, 1),
                 keyframe_ref=("latent" if a.get("ref_latent") else has_kf)
                 ).to(args.sr_dev).eval()
    sr.load_state_dict(blob["model"])

    q_lat: queue.Queue = queue.Queue(maxsize=3)
    q_px: queue.Queue = queue.Queue(maxsize=2)
    ready = threading.Barrier(3)
    t_start = 0.0
    dec_ms, sr_ms, kf_ms = [], [], []
    # Absolute start/end of each stage occupancy, for the wall-clock timeline.
    dec_span, sr_span, lat_span = [], [], []

    def decoder():
        torch.npu.set_device(args.dec_dev)
        with torch.no_grad():
            vae.decode(z_all[:, :CHUNK].unsqueeze(0).to(args.dec_dev, torch.float32) * std + mean)
        torch.npu.synchronize(); ready.wait()
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
            dec_span.append((t0 - t_start, time.time() - t_start))
            q_px.put(px)

    results: list = []

    def refiner():
        torch.npu.set_device(args.sr_dev)
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            # Warm BOTH paths before the clock starts. Warming only the plain
            # forward left the keyframe encoder's kernels to compile inside the
            # first chunk: 1338 ms for an op that then costs 4 ms, which looked
            # like a real cost of the keyframe design and is not.
            if sr.kref is not None:
                wref = sr.kref.encode(
                    torch.zeros(1, 24, 48, 84, device=args.sr_dev), (256, 448))
                sr(torch.randn(1, 3, 256, 448, device=args.sr_dev), frames=1,
                   ref_feat=wref, dist=torch.zeros(1, 1, device=args.sr_dev))
            else:
                sr(torch.randn(1, 3, 256, 448, device=args.sr_dev), frames=1)
        torch.npu.synchronize(); ready.wait()
        emitted, cur_kf, rf = 0, -1, None
        while True:
            px = q_px.get()
            if px is None:
                return
            t0 = time.time()
            # Refresh the keyframe when this chunk crosses an interval boundary.
            kfms = 0.0
            if zref is not None:
                want = (emitted // args.kf_interval) * args.kf_interval
                if want != cur_kf:
                    tk = time.time()
                    tl = min(int(round(want / FRAMES_PER_LAT)), zref.shape[0] - 1)
                    rin = torch.from_numpy(np.asarray(zref[tl], dtype=np.float32)
                                           )[None].to(args.sr_dev)
                    with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                        rf = sr.kref.encode(rin, (256, 448))
                    torch.npu.synchronize()
                    kfms = (time.time() - tk) * 1000
                    cur_kf = want
            outs = []
            with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                for f in range(px.shape[0]):
                    d = (torch.tensor([[float(emitted + f - cur_kf)]], device=args.sr_dev)
                         if zref is not None else None)
                    outs.append(sr(px[f:f + 1].to(args.sr_dev), frames=1,
                                   ref_feat=rf, dist=d).float())
            torch.npu.synchronize()
            sr_ms.append((time.time() - t0) * 1000); kf_ms.append(kfms)
            sr_span.append((t0 - t_start, time.time() - t_start))
            hr = torch.cat(outs)
            emitted += hr.shape[0]
            wall = time.time() - t_start
            results.append(dict(chunk=len(results), frames=emitted,
                                decode_ms=round(dec_ms[-1], 1),
                                refine_ms=round(sr_ms[-1], 1),
                                kf_encode_ms=round(kfms, 2),
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
    ready.wait()
    t_start = time.time()
    print(f"clip {vid}  {n_lat} latents = {n_lat*FRAMES_PER_LATENT/FPS:.1f}s video; "
          f"keyframe every {args.kf_interval} frames; w{a['width']}x{a['blocks']}\n", flush=True)

    buf = []
    for i in range(n_lat):
        due = t_start + i * MS_PER_LATENT / 1000
        now = time.time()
        if now < due:
            time.sleep(due - now)
        buf.append(z_all[:, i])
        if len(buf) == CHUNK:
            lat_span.append(time.time() - t_start)   # chunk complete at H3's output
            q_lat.put(torch.stack(buf, dim=1)); buf = []
    q_lat.put(None)
    tr.join()

    total = time.time() - t_start
    emitted = results[-1]["frames"] if results else 0
    n = len(results)
    summary = dict(clip=vid, width=a["width"], blocks=a["blocks"],
                   keyframe_interval=args.kf_interval if has_kf else None,
                   frames=emitted, chunks=n,
                   wall_s=round(total, 2), video_s=round(emitted / FPS, 2),
                   ms_per_frame=round(total / max(emitted, 1) * 1000, 2),
                   mean_decode_ms=round(sum(dec_ms) / len(dec_ms), 1),
                   mean_refine_ms=round(sum(sr_ms) / len(sr_ms), 1),
                   kf_encodes=sum(1 for v in kf_ms if v > 0),
                   mean_kf_encode_ms=round(sum(kf_ms) / max(sum(1 for v in kf_ms if v > 0), 1), 2),
                   chunk_period_ms=round((results[-1]["wall_s"] - results[1]["wall_s"])
                                         * 1000 / max(n - 2, 1), 1) if n > 2 else None,
                   first_lead_s=results[0]["lead_s"] if results else None,
                   final_lead_s=results[-1]["lead_s"] if results else None,
                   drift_s=round(results[-1]["lead_s"] - results[1]["lead_s"], 3) if n > 2 else None,
                   underruns=sum(1 for r in results[2:]
                                 if r["lead_s"] < results[1]["lead_s"] - 0.1))
    print("\nSTREAMKF " + json.dumps(summary), flush=True)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        np.save(args.out / f"{vid}_stream.npy",
                np.concatenate([r.pop("_px") for r in results]))
        (args.out / "stream.json").write_text(
            json.dumps(dict(summary=summary, events=results,
                            spans=dict(latent_ready=lat_span, decode=dec_span,
                                       refine=sr_span)), indent=2))
        print(f"wrote {args.out}", flush=True)
    os._exit(0)


main()
