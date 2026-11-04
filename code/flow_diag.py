"""Is the velocity field learning anything, or just predicting its own mean?

The corrected sampler still loses at every NFE, and NFE=1 already loses even
though the path is linear and one perfect Euler step would land exactly on the
target. So the velocity is badly estimated. Two very different causes:

  a) undertrained / too small -- the residual IS predictable and we failed
  b) the residual is NOT predictable from this conditioning, so the best the
     model can do is output its conditional mean, and subtracting a mean
     estimate adds error instead of removing it

Distinguish them by comparing the model's velocity MSE against two references:
the variance of the target itself (what predicting a constant zero achieves)
and the variance around the per-batch mean. If the model is barely better than
predicting the mean, it is (b), and no amount of capacity or training fixes it --
the flow has exactly the same conditioning as stage 1 and therefore no new
information to add.
"""
import json, os, random, sys
import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from latflow_model import LatentFlow
from latsr_model import LatentSR
from latsr_train import HLAT, PIX, LLAT, VAL_CLIPS, N_LAT, Clips, to_batch

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

    ids = sorted({f[:-4] for f in os.listdir(HLAT) if f.endswith(".npy")} &
                 {f[:-4] for f in os.listdir(PIX) if f.endswith(".npy")} &
                 {f[:-8] for f in os.listdir(LLAT) if f.endswith(".pt.part")})
    random.Random(1234).shuffle(ids)
    val = Clips(ids[:VAL_CLIPS])
    pr = random.Random(20261013)

    # Stratify by t. Averaging over random t was my own measurement error: at
    # small t, z_t = (1-t)z_hr + t z_src is ALREADY almost the answer, so the
    # model can read the velocity off its own input and scores high. Inference
    # starts at t=1, where z_t carries nothing about z_hr, and that is the only
    # regime a bridge from z_src actually runs in.
    TS = [1.0, 0.75, 0.5, 0.25, 0.05]
    per_t = {tv: {"mse": [], "pow": []} for tv in TS}
    mse_model, var_tgt, var_mean, rel = [], [], [], []
    for _ in range(24):
        items = [val.sample(pr, a["kf_chunks"], a["crop"])]
        lo, kf, tgt, crops, dist = to_batch(items, dev, a["crop"])
        z_hr = torch.from_numpy(np.stack([it[5] for it in items])).to(dev).flatten(0, 1)
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            z_src = st1(lo, N_LAT, st1.encode_key(kf),
                        kf.repeat_interleave(N_LAT, 0), dist).float()
            lr_up = F.interpolate(lo, size=z_src.shape[-2:], mode="bilinear",
                                  align_corners=False)
            t = torch.rand(z_src.shape[0], device=dev)
            z_t = (1 - t)[:, None, None, None] * z_hr + t[:, None, None, None] * z_src
            v = m(z_t, lr_up, kf, t, dist).float()
        target = z_src - z_hr
        for tv in TS:
            tt = torch.full((z_src.shape[0],), tv, device=dev)
            zt = (1 - tt)[:, None, None, None] * z_hr + tt[:, None, None, None] * z_src
            with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                vv = m(zt, lr_up, kf, tt, dist).float()
            per_t[tv]["mse"].append(float(F.mse_loss(vv, target)))
            per_t[tv]["pow"].append(float((target ** 2).mean()))
        mse_model.append(float(F.mse_loss(v, target)))
        var_tgt.append(float((target ** 2).mean()))
        var_mean.append(float(((target - target.mean()) ** 2).mean()))
        # how much of the stage-1 error does the model's velocity explain?
        rel.append(1 - mse_model[-1] / max(var_tgt[-1], 1e-9))

    mm = sum(mse_model) / len(mse_model)
    vt = sum(var_tgt) / len(var_tgt)
    vm = sum(var_mean) / len(var_mean)
    r = sum(rel) / len(rel)
    print(f"velocity MSE (model)        {mm:.5f}", flush=True)
    print(f"target power  (predict 0)   {vt:.5f}", flush=True)
    print(f"variance about global mean  {vm:.5f}", flush=True)
    print(f"fraction of residual explained  {r*100:5.1f}%", flush=True)
    print(f"\nstage1 residual ||z_src - z_hr||^2 per element: {vt:.5f}", flush=True)
    print("\n%6s %12s %12s" % ("t", "vel MSE", "explained"), flush=True)
    for tv in TS:
        a1 = sum(per_t[tv]["mse"]) / len(per_t[tv]["mse"])
        a2 = sum(per_t[tv]["pow"]) / len(per_t[tv]["pow"])
        print("%6.2f %12.5f %11.1f%%" % (tv, a1, (1 - a1 / a2) * 100), flush=True)
    json.dump(dict(velocity_mse=round(mm, 5), target_power=round(vt, 5),
                   var_about_mean=round(vm, 5), explained=round(r, 4)),
              open("results/flow_diag.json", "w"), indent=2)
    os._exit(0)


main()
