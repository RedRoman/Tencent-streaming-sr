"""Four-way comparison so the structure/detail split is visible, not just tabulated.

Panels: bicubic | pixel pipeline (shipping) | latent SR | ground truth.
Each full frame keeps a magnified inset of the same texture-rich region, chosen
by high-frequency energy in the ground truth rather than by eye, and per-frame
PSNR of that region burned in.

The claim being made visible: latent SR loses 4.9 dB of STRUCTURE and only 2.0
of detail, so its failure should look like content sitting in the wrong place
rather than like softness.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np

W = 1344


def _font(sz, bold=True):
    from PIL import ImageFont
    for d in ("/usr/share/fonts/dejavu", "/usr/share/fonts/truetype/dejavu"):
        for n in (["DejaVuSans-Bold.ttf", "DejaVuSans.ttf"] if bold else ["DejaVuSans.ttf"]):
            p = os.path.join(d, n)
            if os.path.exists(p):
                return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def draw(img, items, sz=26):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img); d = ImageDraw.Draw(im); f = _font(sz)
    for x, y, t, c in items:
        d.rectangle([x - 7, y - 5, x + int(len(t) * sz * 0.58) + 7, y + sz + 7], fill=(0, 0, 0))
        d.text((x, y), t, fill=c, font=f)
    return np.asarray(im)


def hf_crop(gt, size):
    T, H, Wd, _ = gt.shape
    g = gt[::max(T // 8, 1)].astype(np.float32).mean(-1)
    lap = np.abs(4 * g[:, 1:-1, 1:-1] - g[:, :-2, 1:-1] - g[:, 2:, 1:-1]
                 - g[:, 1:-1, :-2] - g[:, 1:-1, 2:]).mean(0)
    best, by, bx = -1, 0, 0
    for y in range(0, H - size, 32):
        for x in range(0, Wd - size, 32):
            v = lap[max(y - 1, 0):y + size, max(x - 1, 0):x + size].mean()
            if v > best:
                best, by, bx = v, y, x
    return by, bx


def psnr(a, b):
    m = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean()
    return 10 * np.log10(255.0 ** 2 / max(m, 1e-9))


def inset(frame, cy, cx, cs, z, col):
    f = frame.copy(); H, Wd = f.shape[:2]
    c = f[cy:cy + cs, cx:cx + cs].copy()
    t = 3
    f[cy:cy+cs, cx:cx+t] = col; f[cy:cy+cs, cx+cs-t:cx+cs] = col
    f[cy:cy+t, cx:cx+cs] = col; f[cy+cs-t:cy+cs, cx:cx+cs] = col
    zc = np.repeat(np.repeat(c, z, 0), z, 1); zh, zw = zc.shape[:2]
    m = 16; y0, x0 = m, Wd - zw - m
    if not (y0 + zh < cy or y0 > cy + cs or x0 + zw < cx or x0 > cx + cs):
        x0 = m
    f[y0-t:y0+zh+t, x0-t:x0+zw+t] = col
    f[y0:y0+zh, x0:x0+zw] = zc
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip", type=Path, required=True, help="[bicubic|pixel|gt] npy")
    ap.add_argument("--latsr", type=Path, required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--zoom", type=int, default=2)
    ap.add_argument("--fps", type=int, default=24)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tri = np.load(args.trip, mmap_mode="r")
    ls = np.load(args.latsr, mmap_mode="r")
    T = min(tri.shape[0], ls.shape[0])
    bic, pipe, gt = tri[:T, :, :W], tri[:T, :, W:2*W], tri[:T, :, 2*W:]
    cs = args.crop
    cy, cx = hf_crop(np.asarray(gt[::8]), cs)
    print(f"{args.clip}: inset at ({cy},{cx})", flush=True)

    panels = [("bicubic", bic, (170, 170, 180)),
              ("pixel pipeline (shipping)", pipe, (255, 210, 60)),
              ("latent SR", ls, (255, 120, 110)),
              ("ground truth", gt, (120, 220, 140))]
    H = gt.shape[1]; lab = 42
    out = np.zeros((T, H + lab, W * 4, 3), np.uint8)
    from PIL import Image
    for t in range(T):
        g = np.asarray(gt[t, cy:cy+cs, cx:cx+cs])
        for i, (name, src, col) in enumerate(panels):
            f = inset(np.asarray(src[t]), cy, cx, cs, args.zoom, col)
            x0 = i * W
            out[t, lab:, x0:x0+W] = f
            txt = name if i == 3 else f"{name}  {psnr(np.asarray(src[t, cy:cy+cs, cx:cx+cs]), g):.2f} dB"
            out[t, :lab, x0:x0+W] = draw(out[t, :lab, x0:x0+W], [(14, 8, txt, col)], 28)
        out[t] = draw(out[t], [(14, H + lab - 40, f"frame {t:3d}/{T}", (150, 220, 255))], 22)
    h, w = out.shape[1:3]
    o = args.out / f"{args.clip}_4way.mp4"
    p = subprocess.run(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{w}x{h}", "-r", str(args.fps), "-i", "-",
                        "-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p", str(o)],
                       input=out.tobytes(), capture_output=True)
    if p.returncode:
        print(p.stderr.decode()[-400:], file=sys.stderr); sys.exit(p.returncode)
    print(f"wrote {o}  {w}x{h}  {os.path.getsize(o)/1e6:.1f} MB", flush=True)


main()
