"""Score latent SR on the SAME yardstick as latent_probe.py, not the crop metric.

Training logs LPIPS on 192px crops; the probe and the shipping pipeline were
measured full-frame on four specific clips. Those two numbers are not
comparable -- the smoke run read 0.2008 on crops while the ceiling measured
full-frame is 0.1449, and reading that as "nearly at the ceiling" would be
wrong. So this reproduces the probe's protocol exactly and adds the two controls
that say whether the keyframe is doing the work:

  ours        LR latent + its own keyframe
  shuf-kf     the same model with a keyframe from a DIFFERENT clip
  copy        no model: the keyframe latent used verbatim (the step-0 anchor)
"""
import json, os, random, sys
import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from latsr_model import LatentSR
from hrdec_model import HRDecoder
from latsr_train import HLAT, PIX, LLAT, N_LAT, N_FRAMES, FRAMES_PER_LAT, psnr_from_mse

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
        return float(torch.cat([_LP(a[i:i+4], b[i:i+4]).flatten()
                                for i in range(0, a.shape[0], 4)]).mean())


def main():
    ck = sys.argv[1]
    kf_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    clips = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    b = torch.load(ck, map_location="cpu", weights_only=False)
    a = b["args"]
    m = LatentSR(a["width"], a["lr_blocks"], a["hr_blocks"]).to(dev).eval()
    m.load_state_dict(b["model"])
    db = torch.load(a["decoder"], map_location="cpu", weights_only=False)
    dec = HRDecoder(db["args"]["width"], tuple(db["blocks"])).to(dev).eval()
    dec.load_state_dict(db["model"])
    with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
        for _ in range(3):
            dec(torch.zeros(1, 24, N_LAT, 48, 84, device=dev))
    torch.npu.synchronize()
    print(f"{ck} step {b['step']}  keyframe every {kf_frames} frames", flush=True)

    ids = sorted({f[:-4] for f in os.listdir(HLAT) if f.endswith(".npy")} &
                 {f[:-4] for f in os.listdir(PIX) if f.endswith(".npy")} &
                 {f[:-8] for f in os.listdir(LLAT) if f.endswith(".pt.part")})
    random.Random(1234).shuffle(ids)
    sel = ids[:clips]
    alien = np.load(f"{HLAT}/{ids[clips]}.npy", mmap_mode="r")   # for the control

    # keyframe cadence in chunks: one chunk is 17 frames
    kf_every = max(round(kf_frames / N_FRAMES), 1)
    arms = ["ours", "shuf_kf", "copy"]
    acc = {k: {"psnr": [], "lpips": []} for k in arms}
    for vid in sel:
        hl = np.load(f"{HLAT}/{vid}.npy", mmap_mode="r")
        px = np.load(f"{PIX}/{vid}.npy", mmap_mode="r")
        ll = torch.load(f"{LLAT}/{vid}.pt.part", map_location="cpu",
                        weights_only=False).permute(1, 0, 2, 3).float()
        n = min(hl.shape[0] // N_LAT, px.shape[0] // N_FRAMES, ll.shape[0] // N_LAT)
        out = {k: [] for k in arms}
        for k in range(n):
            kk = (k // kf_every) * kf_every
            d = torch.tensor([[(k - kk) * N_LAT * FRAMES_PER_LAT]], device=dev)
            lo = ll[N_LAT * k:N_LAT * k + N_LAT].to(dev)
            for name, src in (("ours", hl), ("shuf_kf", alien)):
                kf = torch.from_numpy(np.asarray(
                    src[min(N_LAT * kk, src.shape[0] - 1)], dtype=np.float32))[None].to(dev)
                with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                    z = m(lo, N_LAT, m.encode_key(kf), kf.repeat_interleave(N_LAT, 0), d)
                    o = dec(z.permute(1, 0, 2, 3)[None].contiguous()).float()
                out[name].append(o.cpu())
            kf = torch.from_numpy(np.asarray(hl[N_LAT * kk], dtype=np.float32))[None].to(dev)
            with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                o = dec(kf.repeat_interleave(N_LAT, 0).permute(1, 0, 2, 3)[None].contiguous()).float()
            out["copy"].append(o.cpu())
            torch.npu.empty_cache()
        gt = torch.from_numpy(np.asarray(px[:n * N_FRAMES])).permute(0, 3, 1, 2)
        gt = gt.float().div(127.5).sub(1.0)
        for k in arms:
            p = torch.cat(out[k])
            acc[k]["psnr"].append(psnr_from_mse(F.mse_loss(p, gt)))
            acc[k]["lpips"].append(lp(p.to(dev), gt.to(dev)))
            # A stale keyframe base can ghost: fidelity collapses while LPIPS
            # stays respectable, and in video that shows up as drift, not blur.
            # Flicker against GT's own flicker is what catches it.
            acc[k].setdefault("flicker", []).append(
                float((p[1:] - p[:-1]).abs().mean() / 2))
        acc.setdefault("_gt", {}).setdefault("flicker", []).append(
            float((gt[1:] - gt[:-1]).abs().mean() / 2))
        del out, gt
        torch.npu.empty_cache()

    print("%-10s %9s %9s" % ("arm", "psnr", "lpips"), flush=True)
    res = {}
    gf = sum(acc["_gt"]["flicker"]) / len(acc["_gt"]["flicker"])
    print("%-10s %9s %9s %9s  (GT flicker %.4f)" % ("", "", "", "flicker", gf), flush=True)
    for k in arms:
        pm = sum(acc[k]["psnr"]) / len(acc[k]["psnr"])
        lm = sum(acc[k]["lpips"]) / len(acc[k]["lpips"])
        fm = sum(acc[k]["flicker"]) / len(acc[k]["flicker"])
        res[k] = dict(psnr=round(pm, 3), lpips=round(lm, 4), flicker=round(fm, 5))
        print("%-10s %9.3f %9.4f %9.4f" % (k, pm, lm, fm), flush=True)
    res["gt_flicker"] = round(gf, 5)
    res["reference"] = dict(ceiling_true_latent=0.1449, shipping_pixel=0.1910,
                            bicubic=0.3178)
    json.dump(res, open(f"{os.path.dirname(ck)}/eval_full.json", "w"), indent=2)
    os._exit(0)


main()
