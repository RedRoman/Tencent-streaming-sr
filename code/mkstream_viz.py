"""Render the streaming run on a wall-clock time axis.

The comparison videos show frames that were computed offline and played back,
so nothing in them conveys that generation and upscaling happen WHILE the video
plays. This does: the output video's own time axis IS the wall clock of the
measured run, so what you watch unfolds at the speed it really unfolded.

Three things are drawn from the recorded spans, not re-simulated:
  - the picture the viewer would be seeing at that instant (black until the
    first frame is ready -- the time-to-first-frame is real latency, visible)
  - a Gantt chart of the three stages with a "now" line sweeping across, which
    is where the overlap becomes obvious: stage N+1 starts before stage N of
    the next chunk, so the chunk period is the longest stage, not the sum
  - produced vs played, and the buffer between them staying flat

Optionally a second row simulates the same pipeline at full resolution using the
measured HD decode cost, which is the thing this architecture exists to avoid.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys
from pathlib import Path

import numpy as np

FPS_OUT = 24
HD_DECODE_MS = 5846.8        # measured: 5-latent minimum chunk at 1344x768
LR_DECODE_MS = 426.6         # measured, same chunk at 448x256


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

    def bar(self, x, y, w, h, col, radius=0):
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
    ap.add_argument("--run", type=Path, required=True, help="dir from stream_kf.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--slow", type=float, default=1.0, help="2.0 = half speed")
    ap.add_argument("--vid-w", type=int, default=960)
    ap.add_argument("--tail-s", type=float, default=1.5, help="extra time after the run")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    blob = json.loads((args.run / "stream.json").read_text())
    ev, sp, summ = blob["events"], blob["spans"], blob["summary"]
    frames = np.load(next(args.run.glob("*_stream.npy")), mmap_mode="r")
    fps_video = 24

    dec, ref, lat = sp["decode"], sp["refine"], sp["latent_ready"]
    n_chunk = len(ref)
    per_chunk = [ev[0]["frames"]] + [ev[i]["frames"] - ev[i - 1]["frames"]
                                     for i in range(1, n_chunk)]
    ttff = ref[0][1]
    t_end = ref[-1][1] + args.tail_s
    n_out = int(t_end * FPS_OUT * args.slow)

    vw = args.vid_w
    vh = int(round(frames.shape[1] * vw / frames.shape[2]))
    pad, head = 24, 92            # two title lines, so the clock cannot collide
    gantt_h, foot = 240, 120
    W = vw + 2 * pad
    H = head + vh + gantt_h + foot

    # Gantt geometry: the whole run maps to the full width, so the "now" line
    # sweeping left to right is the same clock the video is playing on. The
    # label column is sized to the longest label rather than guessed, which is
    # what made the first version's text run over the bars.
    LABEL_W = 300
    gx0, gx1 = pad + LABEL_W, W - pad - 20
    span = t_end

    def gx(t):
        return gx0 + (gx1 - gx0) * min(max(t, 0), span) / span

    # A small gap at each chunk boundary so consecutive chunks stay countable;
    # H3 really does run back to back, the gap is only a separator.
    rows = [("1. H3 generates low-res", [(max(l - 0.69, 0), l) for l in lat],
             (90, 150, 230)),
            ("2. VAE decode  427 ms", [(a, b - 0.02) for a, b in dec], (120, 200, 140)),
            ("3. refiner  303 ms", [(a, b - 0.02) for a, b in ref], (240, 190, 70))]

    out = np.zeros((n_out, H, W, 3), np.uint8)
    for k in range(n_out):
        t = k / (FPS_OUT * args.slow)
        c = Canvas(H, W)

        # ---------- the picture the viewer would be seeing ----------
        ready = sum(per_chunk[i] for i in range(n_chunk) if ref[i][1] <= t)
        if t < ttff:
            c.bar(pad, head, vw, vh, (22, 22, 28))
            c.text(pad + vw // 2 - 190, head + vh // 2 - 20,
                   "waiting for the first frame", (150, 150, 160), 28)
            c.text(pad + vw // 2 - 90, head + vh // 2 + 18,
                   f"{t:.2f} s / {ttff:.2f} s", (110, 110, 125), 24)
            shown = -1
        else:
            want = int((t - ttff) * fps_video)
            shown = min(want, ready - 1)
            f = np.asarray(frames[max(shown, 0)])
            small = f[::f.shape[0] // vh + 1, ::f.shape[1] // vw + 1]
            from PIL import Image
            small = np.asarray(Image.fromarray(f).resize((vw, vh)))
            c.paste(small, pad, head)
            if want >= ready:
                c.text(pad + 16, head + 16, "BUFFERING", (255, 90, 90), 26)

        c.text(pad, 12, "Sol-H3 streaming", (235, 235, 235), 28)
        c.text(pad, 50, "all three stages run while the video is already playing",
               (150, 150, 165), 21, bold=False)
        c.text(W - pad - 250, 20, f"wall clock {t:5.2f} s", (150, 220, 255), 26)

        # ---------- stage Gantt ----------
        y = head + vh + 20
        c.text(pad, y, "pipeline  (each block is one 17-frame chunk)",
               (170, 170, 185), 20, bold=False)
        RH, GAP = 30, 54
        for i, (name, spans, col) in enumerate(rows):
            ry = y + 36 + i * GAP
            c.text(pad, ry + 4, name, col, 21)
            c.bar(gx0, ry, gx1 - gx0, RH, (30, 30, 38))
            for (a, b) in spans:
                if a >= t:
                    continue
                c.bar(gx(a), ry, max(gx(min(b, t)) - gx(a), 2), RH, col)
        nx = gx(t)
        c.bar(nx - 1, y + 30, 2, 3 * GAP + 4, (255, 255, 255))
        c.text(nx - 18, y + 30 + 3 * GAP + 4, "now", (255, 255, 255), 18)

        # ---------- produced vs played ----------
        fy = head + vh + gantt_h
        prod_s = ready / fps_video
        play_s = max(min((t - ttff), prod_s), 0) if t >= ttff else 0.0
        total_s = sum(per_chunk) / fps_video
        c.text(pad, fy, f"video produced   {prod_s:4.2f} s", (120, 200, 140), 21)
        c.text(pad, fy + 32, f"video played      {play_s:4.2f} s", (235, 235, 235), 21)
        c.text(pad, fy + 64, f"buffer {prod_s - play_s:4.2f} s"
                             f"      stalls {summ['underruns']}",
               (150, 220, 255), 21)
        bx0, bw = pad + LABEL_W, gx1 - (pad + LABEL_W)
        for j, (val, col) in enumerate(((prod_s, (120, 200, 140)),
                                        (play_s, (235, 235, 235)))):
            by = fy + 4 + j * 32
            c.bar(bx0, by, bw, 16, (30, 30, 38))
            c.bar(bx0, by, bw * min(val / total_s, 1.0), 16, col)
        out[k] = c.np()

    encode(out, args.out / "stream_wallclock.mp4", FPS_OUT)
    json.dump(dict(run=str(args.run), slow=args.slow, out_frames=int(n_out),
                   ttff_s=round(ttff, 3), wall_s=round(t_end, 2)),
              open(args.out / "stream_wallclock.json", "w"), indent=2)


main()
