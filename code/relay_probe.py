"""Does a sparse HR keyframe actually carry usable detail к frames away?

RelayVSR's premise: run the expensive generative model only on sparse keyframes
and let a cheap network spread that detail over the interval. Before building
any of that, measure the premise on our own content.

Naive detail transfer, no motion compensation: take the high-frequency residual
of HR frame t-k and paste it onto the bicubic upsample of LR frame t. That is a
LOWER bound on what keyframe conditioning can buy -- a learned, motion-aware
reader can only do better -- but it is also the honest test of whether detail
survives the trip at all. If it is already worse than plain bicubic at k=1, then
detail does not transfer without alignment and the interval question is moot.

Reported against the two bars that matter: bicubic (no detail at all) and the
k=0 oracle (the frame's own detail, i.e. the ceiling of perfect transfer).
"""
import json, os, random
import numpy as np, torch, torch.nn.functional as F
import torch_npu  # noqa: F401
from pixel_train import PIX, to_t, psnr_from_mse

dev = "npu:0"
CLIPS = 6
KS = (0, 1, 2, 4, 8, 15, 30)

_LP = None


def lpips_of(a, b):
    global _LP
    if _LP is None:
        import lpips as pkg
        _LP = pkg.LPIPS(net="vgg").to(dev).eval()
        for q in _LP.parameters():
            q.requires_grad_(False)
    with torch.no_grad():
        return float(torch.cat([_LP(a[i:i + 8], b[i:i + 8]).flatten()
                                for i in range(0, a.shape[0], 8)]).mean())


def detail_of(hr):
    """HF residual: what an upsampler cannot reproduce from the LR grid."""
    lo = F.interpolate(F.interpolate(hr, scale_factor=1 / 3, mode="bicubic",
                                     align_corners=False),
                       scale_factor=3, mode="bicubic", align_corners=False)
    return hr - lo


def main():
    ids = sorted(f[:-4] for f in os.listdir(f"{PIX}/hr") if f.endswith(".npy"))
    random.Random(1234).shuffle(ids)
    acc = {k: dict(psnr=[], lpips=[]) for k in KS}
    bic_p, bic_l = [], []

    for vid in ids[:CLIPS]:
        lr_np = np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")
        hr_np = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")
        T = min(64, lr_np.shape[0])
        lr = to_t(np.asarray(lr_np[:T]), dev)
        hr = to_t(np.asarray(hr_np[:T]), dev)
        base = F.interpolate(lr, scale_factor=3, mode="bicubic", align_corners=False).clamp(-1, 1)
        det = detail_of(hr)

        for k in KS:
            if k >= T:
                continue
            # frame t takes its detail from frame t-k (t>=k so the keyframe is
            # always in the past -- streaming, no lookahead)
            tgt = hr[k:]
            out = (base[k:] + det[: T - k]).clamp(-1, 1)
            acc[k]["psnr"].append(psnr_from_mse(F.mse_loss(out, tgt)))
            acc[k]["lpips"].append(lpips_of(out, tgt))
        bic_p.append(psnr_from_mse(F.mse_loss(base, hr)))
        bic_l.append(lpips_of(base, hr))

    bp, bl = sum(bic_p) / len(bic_p), sum(bic_l) / len(bic_l)
    print(f"{'source':>22s}{'psnr':>9s}{'vs bic':>9s}{'lpips':>9s}{'vs bic':>9s}")
    print(f"{'bicubic (no detail)':>22s}{bp:9.3f}{0.0:9.3f}{bl:9.4f}{0.0:9.4f}")
    rows = dict(bicubic=dict(psnr=round(bp, 3), lpips=round(bl, 4)))
    for k in KS:
        if not acc[k]["psnr"]:
            continue
        p = sum(acc[k]["psnr"]) / len(acc[k]["psnr"])
        l = sum(acc[k]["lpips"]) / len(acc[k]["lpips"])
        tag = "k=0 (own detail)" if k == 0 else f"k={k} frames back"
        print(f"{tag:>22s}{p:9.3f}{p - bp:+9.3f}{l:9.4f}{l - bl:+9.4f}")
        rows[f"k{k}"] = dict(psnr=round(p, 3), lpips=round(l, 4))
    json.dump(rows, open("results/relay_probe.json", "w"), indent=2)
    os._exit(0)


main()
