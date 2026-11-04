"""Can H3's latent space express MOTION as a residual? The decisive test.

The latent model's output is `keyframe_latent + residual`, and the measured
failure is structural: it loses 8.4 dB of structure but only 3.0 dB of detail,
and the loss grows monotonically with distance from the keyframe. In regions
that moved, the residual has to delete an object from its old position and draw
it at the new one. Whether that is even expressible is a property of the space,
not of the model.

So walk the straight line between two REAL HR latents of the same clip, k chunks
apart, and compare it with the straight line between their decoded pixels:

    latent path   decode( (1-a)*z_t + a*z_k )
    pixel  path   (1-a)*decode(z_t) + a*decode(z_k)

A pixel cross-fade is a legitimate image at every a -- double-exposed, but every
structure in it is real. If the latent path decodes to something FARTHER from
every real frame than the pixel cross-fade is, and its high-frequency content
spikes instead of staying between the endpoints, then the straight line in
latent space leaves the manifold. That is distortion rather than ghosting, and
it is exactly the artefact being asked about: the model's output sits partway
along that line whenever something moved.

Both paths are scored against the same two endpoints, so no interpolation
oracle is needed.
"""
import json, os, random, sys
import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from hrdec_model import HRDecoder
from hrdec_train import LAT, PIX, N_LAT, N_FRAMES, psnr_from_mse

dev = "npu:0"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
GAPS = (1, 2, 6)        # chunks apart = 17, 34, 102 frames


def hf(x):
    """Fraction of energy above a 1/3-Nyquist cut; distortion shows up here."""
    g = x.mean(1, keepdim=True)
    lo = F.avg_pool2d(g, 3, stride=1, padding=1)
    return float(((g - lo) ** 2).mean() / ((g ** 2).mean() + 1e-9))


def main():
    db = torch.load("results/hrdec_1200/s768/best.pt", map_location="cpu",
                    weights_only=False)
    dec = HRDecoder(db["args"]["width"], tuple(db["blocks"])).to(dev).eval()
    dec.load_state_dict(db["model"])
    with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
        for _ in range(3):
            dec(torch.zeros(1, 24, N_LAT, 48, 84, device=dev))
    torch.npu.synchronize()

    ids = sorted({f[:-4] for f in os.listdir(LAT) if f.endswith(".npy")} &
                 {f[:-4] for f in os.listdir(PIX) if f.endswith(".npy")})
    random.Random(1234).shuffle(ids)
    clips = ids[:4]

    def D(z):
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            return dec(z.permute(1, 0, 2, 3)[None].contiguous()).float()

    out = {}
    for gap in GAPS:
        acc = {a: {"lat_d": [], "pix_d": [], "lat_hf": [], "pix_hf": []} for a in ALPHAS}
        gt_hf = []
        for vid in clips:
            lat = np.load(f"{LAT}/{vid}.npy", mmap_mode="r")
            n = lat.shape[0] // N_LAT
            for k in range(0, max(n - gap, 1), max((n - gap) // 3, 1)):
                if k + gap >= n:
                    continue
                za = torch.from_numpy(np.asarray(lat[N_LAT*k:N_LAT*k+N_LAT],
                                                 dtype=np.float32)).to(dev)
                zb = torch.from_numpy(np.asarray(lat[N_LAT*(k+gap):N_LAT*(k+gap)+N_LAT],
                                                 dtype=np.float32)).to(dev)
                pa, pb = D(za), D(zb)
                gt_hf.append((hf(pa) + hf(pb)) / 2)
                for a in ALPHAS:
                    lp = D((1 - a) * za + a * zb)          # latent-space line
                    pp = (1 - a) * pa + a * pb             # pixel-space line
                    # distance to the NEARER endpoint: both paths are judged by
                    # the same yardstick, so the comparison needs no oracle.
                    acc[a]["lat_d"].append(max(psnr_from_mse(F.mse_loss(lp, pa)),
                                               psnr_from_mse(F.mse_loss(lp, pb))))
                    acc[a]["pix_d"].append(max(psnr_from_mse(F.mse_loss(pp, pa)),
                                               psnr_from_mse(F.mse_loss(pp, pb))))
                    acc[a]["lat_hf"].append(hf(lp))
                    acc[a]["pix_hf"].append(hf(pp))
                    del lp, pp
                del za, zb, pa, pb
                torch.npu.empty_cache()
        g = sum(gt_hf) / len(gt_hf)
        print(f"\ngap {gap} chunks = {gap*N_FRAMES} frames   (endpoint hf {g:.5f})",
              flush=True)
        print("%6s %11s %11s %11s %11s" % ("alpha", "lat->end", "pix->end",
                                           "lat hf", "pix hf"), flush=True)
        row = {}
        for a in ALPHAS:
            v = acc[a]
            ld = sum(v["lat_d"]) / len(v["lat_d"])
            pd = sum(v["pix_d"]) / len(v["pix_d"])
            lh = sum(v["lat_hf"]) / len(v["lat_hf"])
            ph = sum(v["pix_hf"]) / len(v["pix_hf"])
            row[str(a)] = dict(lat_psnr=round(ld, 2), pix_psnr=round(pd, 2),
                               lat_hf=round(lh, 5), pix_hf=round(ph, 5))
            print("%6.2f %11.2f %11.2f %11.5f %11.5f" % (a, ld, pd, lh, ph), flush=True)
        out[f"gap{gap}"] = dict(endpoint_hf=round(g, 5), alphas=row)
    json.dump(out, open("results/latent_motion.json", "w"), indent=2)
    os._exit(0)


main()
