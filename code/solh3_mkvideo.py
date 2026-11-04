"""Turn saved triptych frame arrays into labelled side-by-side MP4s.

Panels, left to right: bicubic-upscaled LR / Sol-H3 refined / HR ground truth.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

LABELS = ("LR bicubic", "Sol-H3 refined", "HR ground truth")


def label_strip(width, panels, height=28):
    """Render panel captions with PIL if available, else a plain bar."""
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (width, height), (16, 16, 16))
        d = ImageDraw.Draw(img)
        pw = width // panels
        for i in range(panels):
            d.text((i * pw + 10, 8), LABELS[i], fill=(235, 235, 235))
            if i:
                d.line([(i * pw, 0), (i * pw, height)], fill=(90, 90, 90))
        return np.asarray(img)
    except Exception:
        return np.full((height, width, 3), 16, np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=24)
    args = ap.parse_args()

    files = sorted(args.dir.glob("*_triptych.npy"))
    if not files:
        print(f"no triptych arrays in {args.dir}", file=sys.stderr)
        sys.exit(1)

    for f in files:
        v = np.load(f)  # [T,H,W,3]
        strip = label_strip(v.shape[2], len(LABELS))
        v = np.concatenate([np.repeat(strip[None], v.shape[0], 0), v], axis=1)
        out = f.with_suffix("").with_suffix("")
        out = out.parent / (out.name.replace("_triptych", "") + "_compare.mp4")
        # This ffmpeg build has no libx264; mpeg4 at q=2 is the best available
        # and still plays everywhere.
        p = subprocess.run(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{v.shape[2]}x{v.shape[1]}", "-r", str(args.fps), "-i", "-",
             "-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p", str(out)],
            input=v.tobytes(), capture_output=True)
        if p.returncode:
            print(p.stderr.decode()[-600:], file=sys.stderr)
            sys.exit(p.returncode)
        print(f"wrote {out}  {v.shape[0]} frames  {os.path.getsize(out)/1e6:.1f} MB")


main()
