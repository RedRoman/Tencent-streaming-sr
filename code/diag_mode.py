"""train() vs eval() on identical input and weights."""
import os, sys, random
import torch, torch.nn.functional as F
import torch_npu  # noqa: F401
from solh3_model import SolH3Refiner
from solh3_train import Pairs, upsample_lr, HIGH, VAL_IDS

blob = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
a = blob["args"]; obj = a.get("objective", "noise"); dev = "npu:0"
m = SolH3Refiner(width=a["width"], time_taps=a["time_taps"], cond_mode=a["cond"]).to(dev)
m.load_state_dict(blob["model"])

ids = sorted(f[:-8] for f in os.listdir(HIGH) if f.endswith(".pt.part"))
random.Random(1234).shuffle(ids)
ds = Pairs(ids[:VAL_IDS], a.get("lr_source") or "h3")
lo, hi = ds.get(0)
lo = lo[:, :5].unsqueeze(0).to(dev); hi = hi[:, :5].unsqueeze(0).to(dev)
lrup = upsample_lr(lo)
src = lrup if obj == "bridge" else torch.randn_like(hi)
vstar = src - hi
t = torch.full((1,), 0.5, device=dev)
xt = (1 - t).view(-1,1,1,1,1) * hi + t.view(-1,1,1,1,1) * src

for mode in ("train", "eval"):
    getattr(m, mode)()
    with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
        v = m(xt, lrup, t).float()
    print("%-6s pred std %8.4f  MSE %10.4f  cos %7.4f"
          % (mode, v.std(), F.mse_loss(v, vstar).item(),
             F.cosine_similarity(v.flatten(), vstar.flatten(), dim=0).item()), flush=True)

# GroupNorm is mode-independent; the only other candidate is batch size.
m.eval()
for bs in (1, 2, 4):
    xb = xt.repeat(bs,1,1,1,1); lb = lrup.repeat(bs,1,1,1,1); tb = t.repeat(bs)
    with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
        v = m(xb, lb, tb).float()[:1]
    print("eval bs=%d pred std %8.4f  MSE %10.4f"
          % (bs, v.std(), F.mse_loss(v, vstar).item()), flush=True)
os._exit(0)
