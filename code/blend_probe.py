"""How much is there to gain from picking latent SR only where it wins?

Observation from the video, confirmed by the numbers: latent SR beats the pixel
pipeline where little has changed since the keyframe (at the keyframe itself it
wins by 1.62 dB) and degrades into generative-looking distortion where a lot
has (4.68 dB behind by frame 16 of an interval). That suggests a gate rather
than a better model.

Three numbers, in increasing realism:
  oracle     per-pixel pick of whichever is closer to GT. Nothing implementable
             beats this; if it is not much above the better of the two, the
             whole idea is dead and no gate is worth building.
  tile       the same pick at 32x32 tile granularity, i.e. a gate that has to
             commit to a region rather than a pixel.
  change     the gate we could actually ship: decide from |LR_t - LR_kf|
             measured in LR latent space, which costs nothing and needs no
             access to the ground truth. Threshold swept.
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

W, N_FRAMES, N_LAT = 1344, 17, 5
LLAT = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/low/video"
dev = "npu:0"
_LP = None


def lp(a, b):
    global _LP
    if _LP is None:
        import lpips as pkg
        _LP = pkg.LPIPS(net="vgg").to(dev).eval()
        for q in _LP.parameters():
            q.requires_grad_(False)
    with torch.no_grad():
        return float(torch.cat([_LP(a[i:i+4].to(dev), b[i:i+4].to(dev)).flatten()
                                for i in range(0, a.shape[0], 4)]).mean())


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) / 4 + 1e-12))


def main():
    clip = sys.argv[1]
    tri = np.load(f"results/ev4_lat30_kf30/{clip}_triptych.npy", mmap_mode="r")
    ls = np.load(f"results/latsr_vis/{clip}_latsr.npy", mmap_mode="r")
    T = min(tri.shape[0], ls.shape[0])

    def t_(x):
        return torch.from_numpy(np.asarray(x)).permute(0, 3, 1, 2).float().div(127.5).sub(1)

    gt = t_(tri[:T, :, 2*W:])
    pi = t_(tri[:T, :, W:2*W])
    la = t_(ls[:T])

    ep = ((pi - gt) ** 2).mean(1, keepdim=True)
    el = ((la - gt) ** 2).mean(1, keepdim=True)

    out = {}
    out["pixel"] = dict(psnr=round(psnr(pi, gt), 3), lpips=round(lp(pi, gt), 4))
    out["latent"] = dict(psnr=round(psnr(la, gt), 3), lpips=round(lp(la, gt), 4))

    m = (el < ep).float()
    orc = m * la + (1 - m) * pi
    out["oracle_pixelwise"] = dict(psnr=round(psnr(orc, gt), 3),
                                   lpips=round(lp(orc, gt), 4),
                                   latent_share=round(float(m.mean()), 3))

    k = 32
    mt = (F.avg_pool2d(el, k) < F.avg_pool2d(ep, k)).float()
    mt = F.interpolate(mt, size=gt.shape[-2:], mode="nearest")
    ot = mt * la + (1 - mt) * pi
    out["oracle_tile32"] = dict(psnr=round(psnr(ot, gt), 3), lpips=round(lp(ot, gt), 4),
                                latent_share=round(float(mt.mean()), 3))

    # the shippable gate: change since the keyframe, measured in LR latent space
    ll = torch.load(f"{LLAT}/{clip}.pt.part", map_location="cpu",
                    weights_only=False).permute(1, 0, 2, 3).float()
    ch = []
    for t in range(T):
        kchunk = t // N_FRAMES
        li = min(kchunk * N_LAT + (t % N_FRAMES) * N_LAT // N_FRAMES, ll.shape[0] - 1)
        kf_li = min(kchunk * N_LAT, ll.shape[0] - 1)
        d = (ll[li] - ll[kf_li]).abs().mean(0, keepdim=True)
        ch.append(d)
    ch = torch.stack(ch)                                   # [T,1,16,28]
    ch = F.interpolate(ch, size=gt.shape[-2:], mode="bilinear", align_corners=False)
    print("%-22s %9s %9s %9s" % ("", "psnr", "lpips", "latent%"), flush=True)
    for k2 in ("pixel", "latent", "oracle_tile32", "oracle_pixelwise"):
        v = out[k2]
        print("%-22s %9.3f %9.4f %9s" % (k2, v["psnr"], v["lpips"],
                                         v.get("latent_share", "-")), flush=True)
    print("", flush=True)
    # torch.quantile refuses tensors this large; subsample for the thresholds.
    flat = ch.flatten()
    sub = flat[torch.randperm(flat.numel())[:1_000_000]]
    qs = torch.quantile(sub, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
    for q, thr in zip((0.1, 0.25, 0.5, 0.75, 0.9), qs.tolist()):
        g = (ch < thr).float()
        b = g * la + (1 - g) * pi
        key = f"change_q{int(q*100)}"
        out[key] = dict(psnr=round(psnr(b, gt), 3), lpips=round(lp(b, gt), 4),
                        latent_share=round(float(g.mean()), 3))
        print("%-22s %9.3f %9.4f %9.3f" % (key, out[key]["psnr"], out[key]["lpips"],
                                           out[key]["latent_share"]), flush=True)
    json.dump(out, open(f"results/blend_{clip}.json", "w"), indent=2)
    os._exit(0)


main()
