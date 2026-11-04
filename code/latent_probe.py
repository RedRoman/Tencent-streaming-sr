"""What does latent SR start from, and how far does it have to travel?

The distilled decoder fixed the ceiling of the latent route: feeding it the TRUE
HR latent gives LPIPS 0.1449, against 0.1910 for the pixel pipeline we ship. So
latent SR has 24% of headroom and must not spend it.

This measures the two things latent SR would actually be handed, with no model
trained -- decode each with the same student and score against the same frames:

  copy   the HR keyframe latent from j chunks back, used verbatim as if it were
         the current latent. This is pure temporal transport with zero spatial
         work: the floor that any keyframe-conditioned latent SR must beat.
  tri    the LR latent upsampled 3x by trilinear interpolation. This is the
         no-keyframe starting point, and the thing that sank the earlier latent
         attempt -- VAE latents are not smooth, so interpolating them destroys
         the high frequency rather than blurring it.

If `copy` is already close to the ceiling, latent SR is mostly bookkeeping. If
`tri` is catastrophic and `copy` is far from the ceiling, the route needs a real
model and the 24% will not survive it.
"""
import json, os, random, sys
import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from hrdec_model import HRDecoder
from hrdec_train import LAT, PIX, N_LAT, N_FRAMES, psnr_from_mse

dev = "npu:0"
LOW = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/low/video"
_LP = None


def lp(a, b):
    global _LP
    if _LP is None:
        import lpips as pkg
        _LP = pkg.LPIPS(net="vgg").to(dev).eval()
        for q in _LP.parameters():
            q.requires_grad_(False)
    with torch.no_grad():
        return float(torch.cat([_LP(a[i:i+4], b[i:i+4]).flatten()
                                for i in range(0, a.shape[0], 4)]).mean())


def main():
    ck = sys.argv[1]
    clips = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    m = HRDecoder(blob["args"]["width"], tuple(blob["blocks"])).to(dev).eval()
    m.load_state_dict(blob["model"])
    with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
        for _ in range(3):
            m(torch.zeros(1, 24, N_LAT, 48, 84, device=dev))
    torch.npu.synchronize()

    ids = sorted({f[:-4] for f in os.listdir(LAT) if f.endswith(".npy")} &
                 {f[:-4] for f in os.listdir(PIX) if f.endswith(".npy")})
    random.Random(1234).shuffle(ids)

    def dec(z):
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            return m(z.permute(1, 0, 2, 3)[None].contiguous()).float().cpu()

    arms = ["true", "copy1", "copy2", "copy6", "tri"]
    acc = {a: {"psnr": [], "lpips": []} for a in arms}
    for vid in ids[:clips]:
        lat = np.load(f"{LAT}/{vid}.npy", mmap_mode="r")        # [T,24,48,84]
        pix = np.load(f"{PIX}/{vid}.npy", mmap_mode="r")
        zl = torch.load(f"{LOW}/{vid}.pt.part", map_location="cpu",
                        weights_only=False).permute(1, 0, 2, 3)  # [T,24,16,28]
        n_chunk = min(lat.shape[0] // N_LAT, pix.shape[0] // N_FRAMES,
                      zl.shape[0] // N_LAT)
        out = {a: [] for a in arms}
        for k in range(n_chunk):
            t = torch.from_numpy(np.asarray(lat[N_LAT*k:N_LAT*k+N_LAT],
                                            dtype=np.float32)).to(dev)
            out["true"].append(dec(t))
            for j, name in ((1, "copy1"), (2, "copy2"), (6, "copy6")):
                kk = max(k - j, 0)
                c = torch.from_numpy(np.asarray(lat[N_LAT*kk:N_LAT*kk+N_LAT],
                                                dtype=np.float32)).to(dev)
                out[name].append(dec(c))
            # trilinear 3x of the LR latent, matching the HR latent grid
            low = zl[N_LAT*k:N_LAT*k+N_LAT].to(dev).float()      # [5,24,16,28]
            up = F.interpolate(low.permute(1, 0, 2, 3)[None], size=(N_LAT, 48, 84),
                               mode="trilinear", align_corners=False)[0]
            out["tri"].append(dec(up.permute(1, 0, 2, 3).contiguous()))
            torch.npu.empty_cache()
        gt = torch.from_numpy(np.asarray(pix[:n_chunk*N_FRAMES])).permute(0, 3, 1, 2)
        gt = gt.float().div(127.5).sub(1.0)
        for a in arms:
            p = torch.cat(out[a])
            acc[a]["psnr"].append(psnr_from_mse(F.mse_loss(p, gt)))
            acc[a]["lpips"].append(lp(p.to(dev), gt.to(dev)))
        del out, gt
        torch.npu.empty_cache()

    print("%-8s %9s %9s" % ("arm", "psnr", "lpips"), flush=True)
    res = {}
    for a in arms:
        pm = sum(acc[a]["psnr"]) / len(acc[a]["psnr"])
        lm = sum(acc[a]["lpips"]) / len(acc[a]["lpips"])
        res[a] = dict(psnr=round(pm, 3), lpips=round(lm, 4))
        print("%-8s %9.3f %9.4f" % (a, pm, lm), flush=True)
    json.dump(res, open("results/latent_probe.json", "w"), indent=2)
    os._exit(0)


main()
