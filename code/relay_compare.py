"""One probe set, every contender -- the relay d-curve against honest bars.

The d-curve logged during training came from `evaluate_ref`'s probe, while the
no-reference baseline (bigbatch, 0.1884) came from `evaluate`'s probe. Different
clips, different crops: comparing them directly would repeat the protocol
mismatch that has bitten this project before. So rebuild the d-curve probe here
and score everything on it.

Bars, weakest to strongest:
  bicubic        -- no detail at all
  naive paste    -- reference's HF residual copied onto bicubic, no learning
  bigbatch       -- the current deliverable, reference-free
  relay          -- the learned distance-aware reader
and `copy` (output the reference verbatim), which exposes how much of the relay
model's score is just reproducing a nearby frame rather than doing SR.
"""
import json, os, random, sys
import torch, torch.nn.functional as F
import torch_npu  # noqa: F401
from pixel_model import PixelSR
from pixel_train import batch, Clips, PIX, VAL_CLIPS, EVAL_DS, psnr_from_mse
import numpy as np

dev = "npu:0"
N = 10

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
    m = PixelSR(a["width"], a["blocks"], 3, blob["taps"],
                text_dim=0, text_sites=1,
                keyframe_ref=bool(a.get("ref_max"))).to(dev).eval()
    m.load_state_dict(blob["model"])
    return m


def main():
    relay = load(sys.argv[1])
    plain = load(sys.argv[2])
    ids = sorted(f[:-4] for f in os.listdir(f"{PIX}/hr") if f.endswith(".npy"))
    random.Random(1234).shuffle(ids)
    val = Clips(ids[:VAL_CLIPS], "h3")

    print("%5s %9s %9s %9s %9s %9s" % ("d", "relay", "plain", "paste", "copy", "bicubic"))
    out = {}
    for d in EVAL_DS:
        probe = [batch(val, random.Random(20261011 + i), 2, 1, dev,
                       want_text=False, ref_max=max(EVAL_DS), ref_d=d)
                 for i in range(N)]
        acc = {k: [] for k in ("relay", "plain", "paste", "copy", "bicubic")}
        for lr, hr, _t, ref, dist in probe:
            with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                rf = relay.kref.encode(ref)
                o = relay(lr, frames=1, ref_feat=rf, dist=dist).float()
                p = plain(lr, frames=1).float()
            bic = F.interpolate(lr, scale_factor=3, mode="bicubic",
                                align_corners=False).clamp(-1, 1)
            lo = F.interpolate(F.interpolate(ref, scale_factor=1 / 3, mode="bicubic",
                                             align_corners=False),
                               scale_factor=3, mode="bicubic", align_corners=False)
            paste = (bic + (ref - lo)).clamp(-1, 1)
            acc["relay"].append(lp(o, hr)); acc["plain"].append(lp(p, hr))
            acc["paste"].append(lp(paste, hr)); acc["copy"].append(lp(ref, hr))
            acc["bicubic"].append(lp(bic, hr))
        r = {k: sum(v) / len(v) for k, v in acc.items()}
        out[f"d{d}"] = {k: round(v, 4) for k, v in r.items()}
        print("%5d %9.4f %9.4f %9.4f %9.4f %9.4f" % (
            d, r["relay"], r["plain"], r["paste"], r["copy"], r["bicubic"]))
    json.dump(out, open("results/relay_compare.json", "w"), indent=2)
    os._exit(0)
main()
