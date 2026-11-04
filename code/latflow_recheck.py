"""Re-score the trained flow checkpoints with the corrected sampler.

Training was unaffected by the sign error (the x0 used for the pixel loss
checks out algebraically), so the weights should be usable as they are. Sweep
NFE while we are here, since it is an inference-time knob and the right value
is not obvious for a bridge whose endpoints are this close together.
"""
import json, os, random, sys
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from latflow_model import LatentFlow
from latsr_model import LatentSR
from hrdec_model import HRDecoder
from latsr_train import (HLAT, PIX, LLAT, VAL_CLIPS, N_LAT, Clips, to_batch,
                         crop_latent, psnr_from_mse)

dev = "npu:0"


def main():
    ck = sys.argv[1]
    b = torch.load(ck, map_location="cpu", weights_only=False)
    a = b["args"]
    m = LatentFlow(a["width"], a["blocks"]).to(dev).eval()
    m.load_state_dict(b["model"])
    s1b = torch.load(a["stage1"], map_location="cpu", weights_only=False)
    sa = s1b["args"]
    st1 = LatentSR(sa["width"], sa["lr_blocks"], sa["hr_blocks"]).to(dev).eval()
    st1.load_state_dict(s1b["model"])
    db = torch.load(a["decoder"], map_location="cpu", weights_only=False)
    dec = HRDecoder(db["args"]["width"], tuple(db["blocks"])).to(dev).eval()
    dec.load_state_dict(db["model"])
    import lpips as _l
    perc = _l.LPIPS(net="vgg").to(dev).eval()

    ids = sorted({f[:-4] for f in os.listdir(HLAT) if f.endswith(".npy")} &
                 {f[:-4] for f in os.listdir(PIX) if f.endswith(".npy")} &
                 {f[:-8] for f in os.listdir(LLAT) if f.endswith(".pt.part")})
    random.Random(1234).shuffle(ids)
    val = Clips(ids[:VAL_CLIPS])
    pr = random.Random(20261013)
    probe = [[val.sample(pr, a["kf_chunks"], a["crop"])] for _ in range(12)]

    print(f"{ck} step {b['step']}", flush=True)
    print("%6s %10s %10s %10s" % ("nfe", "lpips", "psnr", "gain"), flush=True)
    base_l = None
    res = {}
    for nfe in (0, 1, 2, 4, 8):
        ls, ps = [], []
        for items in probe:
            lo, kf, tgt, crops, dist = to_batch(items, dev, a["crop"])
            with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                z_src = st1(lo, N_LAT, st1.encode_key(kf),
                            kf.repeat_interleave(N_LAT, 0), dist)
                lr_up = F.interpolate(lo, size=z_src.shape[-2:], mode="bilinear",
                                      align_corners=False)
                z = z_src if nfe == 0 else m.sample(z_src, lr_up, kf, dist, nfe=nfe)
                o = dec(crop_latent(z, crops, a["crop"], 1, N_LAT)).float()
            ls.append(float(perc(o.clamp(-1, 1), tgt).mean()))
            ps.append(psnr_from_mse(F.mse_loss(o, tgt)))
        lm, pm = sum(ls) / len(ls), sum(ps) / len(ps)
        if nfe == 0:
            base_l = lm
        res[f"nfe{nfe}"] = dict(lpips=round(lm, 4), psnr=round(pm, 3))
        tag = " (stage1)" if nfe == 0 else ""
        print("%6d %10.4f %10.3f %10s%s" % (
            nfe, lm, pm, f"{base_l-lm:+.4f}" if nfe else "-", tag), flush=True)
    json.dump(res, open(f"{os.path.dirname(ck)}/recheck.json", "w"), indent=2)
    os._exit(0)


main()
