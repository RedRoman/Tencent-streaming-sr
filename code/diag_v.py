"""Is the predicted velocity itself wrong, or is the integrator misusing it?

Ground truth for this objective is v* = src - HR, available in closed form.
Compare the model's v against v* at the exact x_t the training loop would build.
"""
import os, sys, random
import torch, torch.nn.functional as F
import torch_npu  # noqa: F401
from solh3_model import SolH3Refiner
from solh3_train import Pairs, upsample_lr, HIGH, VAL_IDS

CK = sys.argv[1]


def main():
    dev = "npu:0"
    blob = torch.load(CK, map_location="cpu", weights_only=False)
    a = blob["args"]; obj = a.get("objective", "noise")
    m = SolH3Refiner(width=a["width"], time_taps=a["time_taps"],
                         cond_mode=a["cond"],
                         learned_upsample=a.get("learned_upsample", False)).to(dev)
    m.load_state_dict(blob["model"]); m.eval()
    print("step=%s objective=%s" % (blob["step"], obj), flush=True)

    ids = sorted(f[:-8] for f in os.listdir(HIGH) if f.endswith(".pt.part"))
    random.Random(1234).shuffle(ids)
    ds = Pairs(ids[:VAL_IDS], a.get("lr_source") or "h3")
    lo, hi = ds.get(0)
    lo = lo[:, :5].unsqueeze(0).to(dev); hi = hi[:, :5].unsqueeze(0).to(dev)
    lrup = upsample_lr(lo)
    src = lrup if obj == "bridge" else torch.randn_like(hi)
    vstar = src - hi
    print("target v* : mean %+.4f std %.4f" % (vstar.mean(), vstar.std()), flush=True)
    print("%6s %12s %12s %12s %10s" % ("t", "pred_mean", "pred_std", "MSE(v,v*)", "cos"), flush=True)
    for t in (1.0, 0.75, 0.5, 0.25, 0.0):
        tt = torch.full((1,), t, device=dev)
        xt = (1 - tt).view(-1,1,1,1,1) * hi + tt.view(-1,1,1,1,1) * src
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            v = m(xt, lrup, tt).float()
        cos = F.cosine_similarity(v.flatten(), vstar.flatten(), dim=0).item()
        print("%6.2f %12.4f %12.4f %12.4f %10.4f"
              % (t, v.mean(), v.std(), F.mse_loss(v, vstar).item(), cos), flush=True)
    os._exit(0)


main()
