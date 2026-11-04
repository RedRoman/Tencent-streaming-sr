"""Full-frame comparison video with an inset magnifier.

Keeping the whole frame is what makes the result legible as a VIDEO -- motion,
scene, temporal stability. But at 1344 px on screen the two versions are
indistinguishable, which is why the plain triptych said nothing. So: full frame
at native size, with a magnified inset of one texture-rich region pinned in a
corner, and a locator box showing where that region is.

Layout per frame, 2 panels side by side:
    [ previous version + inset ]  [ THIS version + inset ]
Ground truth is carried as a third panel when --with-gt is set; without it the
two-panel form is wider on screen per panel, which is the point.

The inset region is chosen by high-frequency energy in the ground truth rather
than by eye, so it is not the region that happens to flatter the model.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys
from pathlib import Path

import numpy as np

W_PANEL = 1344


def _font(sz):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def draw(img, items, sz=26):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f = _font(sz)
    for x, y, t, c in items:
        d.rectangle([x - 7, y - 5, x + int(len(t) * sz * 0.58) + 7, y + sz + 7],
                    fill=(0, 0, 0))
        d.text((x, y), t, fill=c, font=f)
    return np.asarray(im)


def hf_crop(gt, size):
    T, H, W, _ = gt.shape
    g = gt[::max(T // 8, 1)].astype(np.float32).mean(-1)
    lap = np.abs(4 * g[:, 1:-1, 1:-1] - g[:, :-2, 1:-1] - g[:, 2:, 1:-1]
                 - g[:, 1:-1, :-2] - g[:, 1:-1, 2:]).mean(0)
    best, by, bx = -1, 0, 0
    # Keep the crop away from the frame edge so the inset never overlaps it.
    for y in range(0, H - size, 32):
        for x in range(0, W - size, 32):
            v = lap[max(y - 1, 0):y + size, max(x - 1, 0):x + size].mean()
            if v > best:
                best, by, bx = v, y, x
    return by, bx


def psnr(a, b):
    m = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean()
    return 10 * np.log10(255.0 ** 2 / max(m, 1e-9))


def with_inset(frame, cy, cx, cs, zoom, corner, box_rgb):
    """Full frame, magnified inset pinned in a corner, locator box on the source."""
    f = frame.copy()
    H, W = f.shape[:2]
    c = f[cy:cy + cs, cx:cx + cs].copy()
    # locator on the original region
    t = 3
    f[cy:cy + cs, cx:cx + t] = box_rgb; f[cy:cy + cs, cx + cs - t:cx + cs] = box_rgb
    f[cy:cy + t, cx:cx + cs] = box_rgb; f[cy + cs - t:cy + cs, cx:cx + cs] = box_rgb
    z = np.repeat(np.repeat(c, zoom, 0), zoom, 1)
    zh, zw = z.shape[:2]
    m = 16
    y0 = m if corner[0] == "t" else H - zh - m
    x0 = m if corner[1] == "l" else W - zw - m
    # If the inset would cover its own source region, move it to the other side.
    if not (y0 + zh < cy or y0 > cy + cs or x0 + zw < cx or x0 > cx + cs):
        x0 = m if x0 > W // 2 else W - zw - m
    f[y0 - t:y0 + zh + t, x0 - t:x0 + zw + t] = box_rgb
    f[y0:y0 + zh, x0:x0 + zw] = z
    return f


def encode(frames, out, fps):
    h, w = frames.shape[1:3]
    p = subprocess.run(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p", str(out)],
        input=frames.tobytes(), capture_output=True)
    if p.returncode:
        print(p.stderr.decode()[-500:], file=sys.stderr); sys.exit(p.returncode)
    print(f"wrote {out}  {frames.shape[0]}f  {w}x{h}  {os.path.getsize(out)/1e6:.1f} MB",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--zoom", type=int, default=2)
    ap.add_argument("--with-gt", action="store_true")
    ap.add_argument("--kf-interval", type=int, default=30)
    ap.add_argument("--fps", type=int, default=24)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tn = np.load(args.new / f"{args.clip}_triptych.npy", mmap_mode="r")
    to = np.load(args.old / f"{args.clip}_triptych.npy", mmap_mode="r")
    T = min(tn.shape[0], to.shape[0])
    new = tn[:T, :, W_PANEL:2 * W_PANEL]
    old = to[:T, :, W_PANEL:2 * W_PANEL]
    gt = tn[:T, :, 2 * W_PANEL:]

    cs = args.crop
    cy, cx = hf_crop(np.asarray(gt[::8]), cs)
    print(f"{args.clip}: inset {cs}x{cs} at ({cy},{cx})", flush=True)

    panels = [("previous version", old, (210, 210, 210)),
              ("Sol-H3 keyframe (this version)", new, (255, 210, 60))]
    if args.with_gt:
        panels.append(("ground truth", gt, (120, 220, 140)))

    H, W = new.shape[1], new.shape[2]
    lab = 40
    out = np.zeros((T, H + lab, W * len(panels), 3), np.uint8)
    for t in range(T):
        gtc = np.asarray(gt[t, cy:cy + cs, cx:cx + cs])
        for i, (name, src, col) in enumerate(panels):
            f = with_inset(np.asarray(src[t]), cy, cx, cs, args.zoom, "tr", col)
            x0 = i * W
            out[t, lab:, x0:x0 + W] = f
            txt = name if name == "ground truth" else \
                f"{name}   {psnr(np.asarray(src[t, cy:cy+cs, cx:cx+cs]), gtc):.2f} dB"
            out[t, :lab, x0:x0 + W] = draw(out[t, :lab, x0:x0 + W],
                                           [(14, 7, txt, col)], 26)
        out[t] = draw(out[t], [(14, H + lab - 38,
                                f"frame {t:3d}/{T}   keyframe {t % args.kf_interval} "
                                f"frames ago   inset {args.zoom}x nearest-neighbour",
                                (150, 220, 255))], 22)
    encode(out, args.out / f"{args.clip}_full.mp4", args.fps)
    json.dump(dict(clip=args.clip, inset=[int(cy), int(cx), cs], zoom=args.zoom,
                   frames=int(T), panels=[p[0] for p in panels]),
              open(args.out / f"{args.clip}_full.json", "w"), indent=2)


main()
