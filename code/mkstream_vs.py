"""Two pipelines on one wall clock: this architecture vs generating at full res.

The single-pipeline view shows that Sol-H3 keeps up. It does not show what the
alternative costs, and that is the whole reason the architecture exists. Here
both run on the same clock:

  left   Sol-H3: H3 at 448x256, LR VAE decode (427 ms/chunk, measured), refiner
  right  the same pipeline with H3 at 1344x768, so the VAE decodes at full
         resolution -- 5847 ms for its 5-latent minimum chunk, measured on this
         machine with the same code

Only the decode cost differs between the two columns, and it is a measured
number, not an estimate. The HD column is therefore OPTIMISTIC: it keeps the
cheap LR-sized transformer step and charges only the decode, so the real gap is
wider than what is drawn. Even so the right-hand side never produces a frame
inside the window the left-hand side needs to finish the clip.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys
from pathlib import Path

import numpy as np

FPS_OUT = 24
HD_DECODE_S = 5.8468          # measured, 1344x768, 5-latent chunk
CHUNK_FRAMES = 17
FPS_VIDEO = 24


def _font(sz, bold=True):
    from PIL import ImageFont
    names = ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf"] if bold else ["DejaVuSans.ttf"]
    for d in ("/usr/share/fonts/dejavu", "/usr/share/fonts/truetype/dejavu"):
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


class Canvas:
    def __init__(self, h, w):
        from PIL import Image, ImageDraw
        self.im = Image.new("RGB", (w, h), (12, 12, 16))
        self.d = ImageDraw.Draw(self.im)

    def text(self, x, y, s, col=(235, 235, 235), sz=22, bold=True):
        self.d.text((x, y), s, fill=col, font=_font(sz, bold))

    def bar(self, x, y, w, h, col):
        self.d.rectangle([x, y, x + w, y + h], fill=col)

    def paste(self, arr, x, y):
        from PIL import Image
        self.im.paste(Image.fromarray(arr), (x, y))

    def np(self):
        return np.asarray(self.im)


def encode(frames, out, fps):
    h, w = frames.shape[1:3]
    p = subprocess.run(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p", str(out)],
        input=frames.tobytes(), capture_output=True)
    if p.returncode:
        print(p.stderr.decode()[-500:], file=sys.stderr); sys.exit(p.returncode)
    print(f"wrote {out}  {frames.shape[0]}f  {w}x{h}  {os.path.getsize(out)/1e6:.1f} MB",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vid-w", type=int, default=660)
    ap.add_argument("--tail-s", type=float, default=2.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    blob = json.loads((args.run / "stream.json").read_text())
    sp, ev, summ = blob["spans"], blob["events"], blob["summary"]
    frames = np.load(next(args.run.glob("*_stream.npy")), mmap_mode="r")
    dec, ref, lat = sp["decode"], sp["refine"], sp["latent_ready"]
    n_chunk = len(ref)
    per_chunk = [ev[0]["frames"]] + [ev[i]["frames"] - ev[i - 1]["frames"]
                                     for i in range(1, n_chunk)]
    ttff = ref[0][1]
    t_end = ref[-1][1] + args.tail_s

    # HD column: same chunk cadence, decode replaced by the measured HD cost.
    # Chunk i cannot start decoding before its latents exist AND the decoder is
    # free, so the stage serialises and the backlog grows.
    hd_dec, free = [], 0.0
    for i in range(n_chunk):
        start = max(lat[i], free)
        end = start + HD_DECODE_S
        hd_dec.append((start, end)); free = end
    hd_ref = [(b, b + (ref[i][1] - ref[i][0])) for i, (a, b) in enumerate(hd_dec)]
    hd_ttff = hd_ref[0][1]

    vw = args.vid_w
    vh = int(round(frames.shape[1] * vw / frames.shape[2]))
    # head leaves room for two title lines PLUS the per-column caption line;
    # at 96 the column titles landed on top of the subtitle.
    pad, head, gap = 20, 128, 20
    gantt_h, foot = 200, 92
    W = 2 * vw + 2 * pad + gap
    H = head + vh + gantt_h + foot
    LABEL_W = 250
    span = t_end
    n_out = int(t_end * FPS_OUT)

    cols = [
        dict(x=pad, title="Sol-H3  (this work)", col=(255, 210, 60),
             sub="H3 at 448x256, decode 427 ms", dec=dec, ref=ref,
             ttff=ttff, real=True),
        dict(x=pad + vw + gap, title="direct high-res", col=(255, 120, 110),
             sub="H3 at 1344x768, decode 5847 ms", dec=hd_dec, ref=hd_ref,
             ttff=hd_ttff, real=False),
    ]

    out = np.zeros((n_out, H, W, 3), np.uint8)
    from PIL import Image
    for k in range(n_out):
        t = k / FPS_OUT
        c = Canvas(H, W)
        c.text(pad, 10, "same clip, same machine, same clock", (235, 235, 235), 27)
        c.text(pad, 48, "only the resolution H3 generates at differs; "
                        "decode costs are measured", (150, 150, 165), 20, bold=False)
        c.text(W - pad - 240, 18, f"wall clock {t:5.2f} s", (150, 220, 255), 26)

        for cd in cols:
            x = cd["x"]
            c.text(x, head - 60, cd["title"], cd["col"], 24)
            c.text(x, head - 30, cd["sub"], (150, 150, 165), 19, bold=False)
            ready = sum(per_chunk[i] for i in range(n_chunk) if cd["ref"][i][1] <= t)
            if t < cd["ttff"]:
                c.bar(x, head, vw, vh, (22, 22, 28))
                c.text(x + vw // 2 - 150, head + vh // 2 - 26,
                       "no frame yet", (140, 140, 155), 26)
                c.text(x + vw // 2 - 110, head + vh // 2 + 12,
                       f"{t:.2f} s / {cd['ttff']:.2f} s", (110, 110, 125), 22)
            else:
                shown = min(int((t - cd["ttff"]) * FPS_VIDEO), ready - 1)
                f = np.asarray(frames[max(shown, 0)])
                c.paste(np.asarray(Image.fromarray(f).resize((vw, vh))), x, head)
                if int((t - cd["ttff"]) * FPS_VIDEO) >= ready:
                    c.text(x + 14, head + 14, "STALLED", (255, 90, 90), 26)
            prod = ready / FPS_VIDEO
            c.text(x, head + vh + 8, f"produced {prod:4.2f} s of video",
                   (120, 200, 140) if prod else (120, 120, 130), 21)

        # ---- one shared gantt, both columns' stages ----
        gx0, gx1 = pad + LABEL_W, W - pad - 20

        def gx(v):
            return gx0 + (gx1 - gx0) * min(max(v, 0), span) / span

        y = head + vh + 44
        rows = [("H3 low-res", [(max(l - 0.69, 0), l) for l in lat], (90, 150, 230)),
                ("decode  427 ms", [(a, b - .02) for a, b in dec], (120, 200, 140)),
                ("decode  5847 ms", [(a, b - .02) for a, b in hd_dec], (255, 120, 110))]
        RH, GAP = 28, 48
        for i, (name, spans, col) in enumerate(rows):
            ry = y + i * GAP
            c.text(pad, ry + 3, name, col, 20)
            c.bar(gx0, ry, gx1 - gx0, RH, (30, 30, 38))
            for (a, b) in spans:
                if a >= t:
                    continue
                c.bar(gx(a), ry, max(gx(min(b, t)) - gx(a), 2), RH, col)
        nx = gx(t)
        c.bar(nx - 1, y - 6, 2, 3 * GAP + 2, (255, 255, 255))
        out[k] = c.np()

    encode(out, args.out / "stream_vs_hd.mp4", FPS_OUT)
    json.dump(dict(ours_ttff_s=round(ttff, 2), hd_ttff_s=round(hd_ttff, 2),
                   hd_decode_s=HD_DECODE_S, wall_s=round(t_end, 2),
                   note="HD column charges only the measured decode difference"),
              open(args.out / "stream_vs_hd.json", "w"), indent=2)


main()
