"""The inset says latent SR wins; the aggregate says it loses by 3.4 dB. Both
cannot be read as "this is what latent SR looks like", so find out where the
error actually is.

Per-frame full-frame PSNR for both, plus a spatial error map. If the claim
"structure drift" is right, latent SR's error should be spread over large
regions rather than concentrated in texture, and it should grow with distance
from the keyframe.
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

W = 1344
N_FRAMES = 17


def psnr(a, b):
    m = ((a - b) ** 2).mean()
    return float(10 * np.log10(4.0 / max(m, 1e-12)))


def main():
    clip = sys.argv[1]
    tri = np.load(f"results/ev4_lat30_kf30/{clip}_triptych.npy", mmap_mode="r")
    ls = np.load(f"results/latsr_vis/{clip}_latsr.npy", mmap_mode="r")
    T = min(tri.shape[0], ls.shape[0])

    def norm(x):
        return x.astype(np.float32) / 127.5 - 1.0

    rows = []
    for t in range(T):
        gt = norm(np.asarray(tri[t, :, 2*W:]))
        pi = norm(np.asarray(tri[t, :, W:2*W]))
        la = norm(np.asarray(ls[t]))
        rows.append(dict(f=t, pixel=round(psnr(pi, gt), 2), latent=round(psnr(la, gt), 2),
                         kf_age=t % N_FRAMES))
    pm = sum(r["pixel"] for r in rows) / T
    lm = sum(r["latent"] for r in rows) / T
    print(f"{clip}: full-frame PSNR over {T} frames   pixel {pm:.2f}  latent {lm:.2f}",
          flush=True)
    win = sum(1 for r in rows if r["latent"] > r["pixel"])
    print(f"  frames where latent SR wins: {win}/{T}", flush=True)
    # does the gap track distance from the keyframe?
    by_age = {}
    for r in rows:
        by_age.setdefault(r["kf_age"], []).append(r["pixel"] - r["latent"])
    print("  gap (pixel - latent) by frames since keyframe:", flush=True)
    for age in sorted(by_age)[:17:4]:
        v = by_age[age]
        print(f"    age {age:2d}: {sum(v)/len(v):+.2f} dB", flush=True)

    # spatial distribution of the error, coarse grid
    t0 = T // 2
    gt = norm(np.asarray(tri[t0, :, 2*W:]))
    pi = norm(np.asarray(tri[t0, :, W:2*W]))
    la = norm(np.asarray(ls[t0]))
    gh, gw = 6, 10
    ch, cw = gt.shape[0] // gh, gt.shape[1] // gw
    print(f"\n  per-tile PSNR at frame {t0} (pixel / latent):", flush=True)
    for i in range(gh):
        line = []
        for j in range(gw):
            s = (slice(i*ch, (i+1)*ch), slice(j*cw, (j+1)*cw))
            line.append(f"{psnr(pi[s], gt[s]):4.1f}/{psnr(la[s], gt[s]):4.1f}")
        print("    " + " ".join(line), flush=True)
    json.dump(dict(clip=clip, pixel_mean=round(pm, 2), latent_mean=round(lm, 2),
                   latent_wins=win, frames=T),
              open(f"results/where_err_{clip}.json", "w"), indent=2)
    os._exit(0)


main()
