"""In moving regions, is latent SR STUCK at the keyframe, or does it put things
in the wrong place?

Latent interpolation between two real latents turned out to be as clean as a
pixel cross-fade, so "the space cannot express the in-between" is not the
explanation. Two candidates remain, and they are distinguishable:

  stuck        the residual is ~0 where it should not be, so the output still
               shows the keyframe's content. Then the output is CLOSER to the
               ground truth at the KEYFRAME's time than at the current time.
  mislocalised the residual fires but puts structure at the wrong coordinates.
               Then the output is far from both, and the error concentrates on
               edges, where a small displacement costs the most.

Measured per 64x64 tile, stratified by how much that tile actually moved, with
the pixel pipeline alongside as the control that does have faithful structure.
"""
import json, os, sys
import numpy as np

W, N_FRAMES = 1344, 17
TILE = 64


def psnr(a, b):
    m = ((a - b) ** 2).mean()
    return float(10 * np.log10(4.0 / max(m, 1e-12)))


def main():
    clip = sys.argv[1]
    tri = np.load(f"results/ev4_lat30_kf30/{clip}_triptych.npy", mmap_mode="r")
    ls = np.load(f"results/latsr_vis/{clip}_latsr.npy", mmap_mode="r")
    T = min(tri.shape[0], ls.shape[0])

    def n_(x):
        return np.asarray(x).astype(np.float32) / 127.5 - 1.0

    rows = []
    for t in range(T):
        kf = (t // N_FRAMES) * N_FRAMES
        if kf == t:
            continue
        gt_t = n_(tri[t, :, 2*W:])
        gt_kf = n_(tri[kf, :, 2*W:])
        la = n_(ls[t])
        pi = n_(tri[t, :, W:2*W])
        H, Wd = gt_t.shape[:2]
        for i in range(0, H - TILE + 1, TILE):
            for j in range(0, Wd - TILE + 1, TILE):
                s = (slice(i, i+TILE), slice(j, j+TILE))
                mot = float(np.abs(gt_t[s] - gt_kf[s]).mean())
                # edge density of the truth: where a displacement costs most
                g = gt_t[s].mean(-1)
                edge = float(np.abs(np.diff(g, axis=0)).mean()
                             + np.abs(np.diff(g, axis=1)).mean())
                rows.append((mot, edge,
                             psnr(la[s], gt_t[s]),    # latent vs now
                             psnr(la[s], gt_kf[s]),   # latent vs keyframe
                             psnr(pi[s], gt_t[s])))   # pixel vs now
    rows.sort(key=lambda r: r[0])
    nb = 5
    per = len(rows) // nb
    print(f"{clip}: {len(rows)} tiles, {T} frames (keyframe tiles excluded)", flush=True)
    print("%-12s %7s %7s %9s %9s %9s %9s" % (
        "motion bin", "motion", "edge", "lat|now", "lat|kf", "pix|now", "pix-lat"),
        flush=True)
    out = []
    for b in range(nb):
        g = rows[b*per:(b+1)*per] if b < nb - 1 else rows[b*per:]
        mo = sum(r[0] for r in g) / len(g)
        ed = sum(r[1] for r in g) / len(g)
        ln = sum(r[2] for r in g) / len(g)
        lk = sum(r[3] for r in g) / len(g)
        pn = sum(r[4] for r in g) / len(g)
        lab = ["least", "low", "mid", "high", "most"][b]
        out.append(dict(bin=lab, motion=round(mo, 4), edge=round(ed, 4),
                        lat_vs_now=round(ln, 2), lat_vs_kf=round(lk, 2),
                        pix_vs_now=round(pn, 2), gap=round(pn - ln, 2)))
        print("%-12s %7.4f %7.4f %9.2f %9.2f %9.2f %9.2f" % (
            lab, mo, ed, ln, lk, pn, pn - ln), flush=True)
    json.dump(out, open(f"results/motion_err_{clip}.json", "w"), indent=2)


main()
