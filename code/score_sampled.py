"""Score arms by what they actually produce: a full sampling trajectory.

`evaluate()` measures a single-step x0 estimate from a random x_t, which is easy
whenever t is small and therefore flatters the model. The deliverable depends on
rolling the ODE from t=1 to t=0, so measure that instead, on a fixed clip set.
"""
import json, os, random, sys
import torch, torch.nn.functional as F
import torch_npu  # noqa: F401
from solh3_model import SolH3Refiner
from solh3_train import Pairs, upsample_lr, HIGH, VAL_IDS

ck, nfes = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
blob = torch.load(ck, map_location="cpu", weights_only=False)
a = blob["args"]; src_name = a.get("lr_source") or "h3"
obj = a.get("objective", "noise"); lu = a.get("learned_upsample", False)
dev = "npu:0"
m = SolH3Refiner(width=a["width"], time_taps=a["time_taps"], cond_mode=a["cond"],
                 learned_upsample=lu).to(dev)
m.load_state_dict(blob["model"]); m.eval()

ids = sorted(f[:-8] for f in os.listdir(HIGH) if f.endswith(".pt.part"))
random.Random(1234).shuffle(ids)
ds = Pairs(ids[:VAL_IDS], src_name)

W = a["chunk"]
out = dict(arm=os.path.basename(os.path.dirname(ck)), step=blob["step"], W=W,
           obj=obj, lu=lu, lr_source=src_name)
for nfe in nfes:
    tot, totbase, n = 0.0, 0.0, 0
    for ci in range(4):
        lo, hi = ds.get(ci)
        T = (hi.shape[1] // W) * W
        lo = lo[:, :T].unsqueeze(0).to(dev); hi = hi[:, :T].unsqueeze(0).to(dev)
        lrup = upsample_lr(lo)
        cond = lo if lu else lrup
        pieces = []
        g = torch.Generator(device="cpu").manual_seed(7)
        for s in range(0, T, W):
            c = cond[:, :, s:s+W]
            x = lrup[:, :, s:s+W].clone() if obj == "bridge" else \
                torch.randn(lrup[:, :, s:s+W].shape, generator=g).to(dev)
            ts = torch.linspace(1, 0, nfe + 1, device=dev)
            with torch.no_grad():
                for i in range(nfe):
                    with torch.autocast("npu", dtype=torch.bfloat16):
                        v = m(x, c, ts[i].expand(1)).float()
                    x = x + (ts[i+1] - ts[i]) * v
            pieces.append(x)
        pred = torch.cat(pieces, dim=2)
        tot += float(-10*torch.log10(F.mse_loss(pred, hi)+1e-12))
        totbase += float(-10*torch.log10(F.mse_loss(lrup, hi)+1e-12))
        n += 1
    out[f"sampled_psnr_nfe{nfe}"] = round(tot/n, 3)
    out["trilinear_psnr"] = round(totbase/n, 3)
print("RESULT " + json.dumps(out), flush=True)
sys.stdout.flush()
os._exit(0)
