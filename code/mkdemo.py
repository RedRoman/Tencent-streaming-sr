"""Build comparison videos that actually show what changed.

The triptych was unreadable: three 1344-wide panels in one 4032-wide frame
means every panel is shrunk on screen, and the whole point of this model is
high-frequency detail that disappears at that scale. Two videos instead:

  zoom  a 2x2 grid of ONE texture-rich crop, magnified with nearest-neighbour
        so the real pixels are visible and no interpolation flatters anyone:
        bicubic | previous delivery | this version | ground truth.
        A locator shows where the crop sits, and per-frame crop PSNR is burned
        in so the claim is checkable frame by frame.

  wipe  previous vs this version on the SAME frame, divider sweeping across.
        Side-by-side invites "they look the same"; a seam does not.

The crop is chosen by high-frequency energy in the ground truth, not by eye --
picking the region by hand would be picking the region that flatters the model.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys
from pathlib import Path

import numpy as np

W_PANEL = 1344
FONT = None


def _font(sz):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def draw(img, items, sz=22):
    """items: list of (x, y, text, rgb)."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f = _font(sz)
    for x, y, t, c in items:
        d.rectangle([x - 6, y - 4, x + int(len(t) * sz * 0.58) + 6, y + sz + 6],
                    fill=(0, 0, 0))
        d.text((x, y), t, fill=c, font=f)
    return np.asarray(im)


def hf_crop(gt, size):
    """Pick the crop with the most high-frequency energy in the ground truth.

    Averaged over several frames so a single motion-blurred frame cannot decide
    it, and on a coarse stride so neighbouring candidates stay distinct.
    """
    T, H, W, _ = gt.shape
    g = gt[::max(T // 8, 1)].astype(np.float32).mean(-1)
    # |Laplacian| as the high-frequency proxy; cheap and scale-free enough here.
    lap = np.abs(4 * g[:, 1:-1, 1:-1] - g[:, :-2, 1:-1] - g[:, 2:, 1:-1]
                 - g[:, 1:-1, :-2] - g[:, 1:-1, 2:]).mean(0)
    best, by, bx = -1, 0, 0
    for y in range(0, H - size, 64):
        for x in range(0, W - size, 64):
            v = lap[max(y - 1, 0):y + size, max(x - 1, 0):x + size].mean()
            if v > best:
                best, by, bx = v, y, x
    return by, bx


def psnr(a, b):
    m = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean()
    return 10 * np.log10(255.0 ** 2 / max(m, 1e-9))


def encode(frames, out, fps):
    h, w = frames.shape[1:3]
    p = subprocess.run(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p", str(out)],
        input=frames.tobytes(), capture_output=True)
    if p.returncode:
        print(p.stderr.decode()[-500:], file=sys.stderr); sys.exit(p.returncode)
    print(f"wrote {out}  {frames.shape[0]}f  {frames.shape[2]}x{frames.shape[1]}"
          f"  {os.path.getsize(out)/1e6:.1f} MB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=Path, required=True, help="dir with this version's triptychs")
    ap.add_argument("--old", type=Path, required=True, help="dir with the previous version's")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--crop", type=int, default=256, help="HR crop size before zoom")
    ap.add_argument("--zoom", type=int, default=2)
    ap.add_argument("--kf-interval", type=int, default=30)
    ap.add_argument("--fps", type=int, default=24)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tn = np.load(args.new / f"{args.clip}_triptych.npy", mmap_mode="r")
    to = np.load(args.old / f"{args.clip}_triptych.npy", mmap_mode="r")
    T = min(tn.shape[0], to.shape[0])
    # triptych layout is [bicubic | model | ground truth]
    bic = tn[:T, :, :W_PANEL]
    new = tn[:T, :, W_PANEL:2 * W_PANEL]
    old = to[:T, :, W_PANEL:2 * W_PANEL]
    gt = tn[:T, :, 2 * W_PANEL:]

    cs = args.crop
    cy, cx = hf_crop(np.asarray(gt[::8]), cs)
    print(f"{args.clip}: crop {cs}x{cs} at ({cy},{cx}) by HF energy", flush=True)

    z = args.zoom
    panels = (("LR bicubic", bic), ("previous version", old),
              ("THIS version (keyframe)", new), ("ground truth", gt))
    pw, ph = cs * z, cs * z
    lab = 34
    grid = np.zeros((T, 2 * (ph + lab), 2 * pw, 3), np.uint8)

    for t in range(T):
        gtc = np.asarray(gt[t, cy:cy + cs, cx:cx + cs])
        for i, (name, src) in enumerate(panels):
            c = np.asarray(src[t, cy:cy + cs, cx:cx + cs])
            # nearest-neighbour: show the pixels that exist, do not invent any
            zc = np.repeat(np.repeat(c, z, 0), z, 1)
            r, col = divmod(i, 2)
            y0 = r * (ph + lab)
            x0 = col * pw
            grid[t, y0 + lab:y0 + lab + ph, x0:x0 + pw] = zc
            txt = name if i == 3 else f"{name}   {psnr(c, gtc):.2f} dB"
            grid[t, y0:y0 + lab, x0:x0 + pw] = draw(
                grid[t, y0:y0 + lab, x0:x0 + pw],
                [(10, 6, txt, (255, 230, 120) if i == 2 else (230, 230, 230))], 22)
        d = t % args.kf_interval
        grid[t] = draw(grid[t], [(10, 2 * (ph + lab) - 30,
                                  f"frame {t:3d}   keyframe {d} frames ago",
                                  (150, 220, 255))], 20)
    encode(grid, args.out / f"{args.clip}_zoom.mp4", args.fps)

    # ---- wipe, on the ZOOMED crop ----
    # A full-resolution wipe showed nothing: at 1344 px on screen the two
    # versions are indistinguishable, which is the same reason the triptych
    # failed. The seam only means something where the pixels are visible.
    wipe = np.zeros((T, ph + lab, pw, 3), np.uint8)
    for t in range(T):
        split = int(pw * (0.5 + 0.40 * np.sin(2 * np.pi * t / T)))
        a = np.asarray(new[t, cy:cy + cs, cx:cx + cs])
        b = np.asarray(old[t, cy:cy + cs, cx:cx + cs])
        za = np.repeat(np.repeat(a, z, 0), z, 1)
        zb = np.repeat(np.repeat(b, z, 0), z, 1)
        f = za.copy()
        f[:, :split] = zb[:, :split]
        f[:, max(split - 3, 0):split + 3] = (255, 220, 60)
        wipe[t, lab:] = f
        wipe[t, :lab] = draw(wipe[t, :lab],
                             [(8, 6, "<- previous", (230, 230, 230)),
                              (pw - 290, 6, "THIS version ->", (255, 230, 120))], 22)
    encode(wipe, args.out / f"{args.clip}_wipe.mp4", args.fps)

    # ---- keyframe decay: the honest worst case, as one still ----
    # Quality is not uniform across an interval -- it is best right after a
    # keyframe and worst just before the next. Showing only d=0 would be
    # cherry-picking the best moment in every 30.
    ds = [0, 10, 20, args.kf_interval - 1]
    base = (T // 2 // args.kf_interval) * args.kf_interval
    tiles = []
    for d in ds:
        t = min(base + d, T - 1)
        c = np.asarray(new[t, cy:cy + cs, cx:cx + cs])
        g = np.asarray(gt[t, cy:cy + cs, cx:cx + cs])
        zc = np.repeat(np.repeat(c, z, 0), z, 1)
        head = draw(np.zeros((lab, pw, 3), np.uint8),
                    [(8, 6, f"keyframe {d} frames ago   {psnr(c, g):.2f} dB",
                      (255, 230, 120))], 22)
        tiles.append(np.concatenate([head, zc], 0))
    strip = np.concatenate(tiles, 1)
    from PIL import Image
    Image.fromarray(strip).save(args.out / f"{args.clip}_decay.png")
    print(f"wrote {args.out}/{args.clip}_decay.png  {strip.shape[1]}x{strip.shape[0]}",
          flush=True)

    json.dump(dict(clip=args.clip, crop=[int(cy), int(cx), cs], zoom=z, frames=int(T)),
              open(args.out / f"{args.clip}_demo.json", "w"), indent=2)


main()
