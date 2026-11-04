"""How good does the keyframe have to be before the relay pays off?

The d-curve was measured with a GT keyframe -- an oracle. A deployed system has
to GENERATE that keyframe, and it will not be perfect. So sweep the reference's
quality and find the break-even point: the keyframe generator only has to beat
whatever LPIPS this curve crosses the reference-free baseline at.

Quality is dialled by blending GT's high-frequency residual back onto bicubic:
alpha=1 is the GT keyframe, alpha=0 is a keyframe with no detail at all. That
models a generator that recovers a fraction of the detail, which is exactly what
a real one does; it does not model a generator that invents WRONG detail, so
treat this as the optimistic side of the question.
"""
import json, os, random, sys
import torch, torch.nn.functional as F
import torch_npu  # noqa: F401
from pixel_model import PixelSR
from pixel_train import batch, Clips, PIX, VAL_CLIPS

dev = "npu:0"; N = 10
ALPHAS = (1.0, 0.75, 0.5, 0.25, 0.0)
DS = (4, 8, 15, 30)
_LP = None


def lp(a, b):
    global _LP
    if _LP is None:
        import lpips as pkg
        _LP = pkg.LPIPS(net="vgg").to(dev).eval()
        for q in _LP.parameters():
            q.requires_grad_(False)
    with torch.no_grad():
        return float(_LP(a.clamp(-1, 1), b).mean())


def load(ck):
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    a = blob["args"]
    m = PixelSR(a["width"], a["blocks"], 3, blob["taps"], text_dim=0, text_sites=1,
                keyframe_ref=bool(a.get("ref_max"))).to(dev).eval()
    m.load_state_dict(blob["model"])
    return m


def main():
    relay, plain = load(sys.argv[1]), load(sys.argv[2])
    ids = sorted(f[:-4] for f in os.listdir(f"{PIX}/hr") if f.endswith(".npy"))
    random.Random(1234).shuffle(ids)
    val = Clips(ids[:VAL_CLIPS], "h3")

    print("reference quality -> relay LPIPS (reference-free baseline in last row)")
    print("%7s" % "alpha" + "".join("%10s" % f"d={d}" for d in DS) + "%12s" % "ref_lpips")
    out, plain_sc = {}, []
    for alpha in ALPHAS:
        row, refq = [], []
        for d in DS:
            probe = [batch(val, random.Random(20261011 + i), 2, 1, dev,
                           want_text=False, ref_max=30, ref_d=d) for i in range(N)]
            vals = []
            for lr, hr, _t, ref, dist in probe:
                lo = F.interpolate(F.interpolate(ref, scale_factor=1 / 3, mode="bicubic",
                                                 align_corners=False),
                                   scale_factor=3, mode="bicubic", align_corners=False)
                rq = (lo + alpha * (ref - lo)).clamp(-1, 1)
                with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                    o = relay(lr, frames=1, ref_feat=relay.kref.encode(rq),
                              dist=dist).float()
                    if alpha == ALPHAS[0] and d == DS[0]:
                        plain_sc.append(lp(plain(lr, frames=1).float(), hr))
                vals.append(lp(o, hr))
                if d == DS[0]:
                    # quality of the keyframe itself, so the break-even point is
                    # stated in the units a keyframe generator is measured in
                    refq.append(lp(rq, ref))
            row.append(sum(vals) / len(vals))
        out[f"a{alpha}"] = [round(v, 4) for v in row]
        print("%7.2f" % alpha + "".join("%10.4f" % v for v in row)
              + "%12.4f" % (sum(refq) / len(refq)))
    pb = sum(plain_sc) / len(plain_sc)
    print("%7s" % "none" + "".join("%10.4f" % pb for _ in DS) + "%12s" % "-")
    out["plain"] = round(pb, 4)
    json.dump(out, open("results/relay_refq.json", "w"), indent=2)
    os._exit(0)


main()
