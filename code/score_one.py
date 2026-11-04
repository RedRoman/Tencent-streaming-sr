"""Deterministically score a single arm. One process per arm keeps a failure
(or a TBE teardown crash) from taking the whole comparison down."""
import json, os, random, sys
import torch
import torch_npu  # noqa: F401
from solh3_model import SolH3Refiner
from solh3_train import Pairs, evaluate, HIGH, VAL_IDS

ck = sys.argv[1]
blob = torch.load(ck, map_location="cpu", weights_only=False)
a = blob["args"]
src = a.get("lr_source") or "h3"
m = SolH3Refiner(width=a["width"], time_taps=a["time_taps"], cond_mode=a["cond"],
                 learned_upsample=a.get("learned_upsample", False)).to("npu:0")
m.load_state_dict(blob["model"])
ids = sorted(f[:-8] for f in os.listdir(HIGH) if f.endswith(".pt.part"))
random.Random(1234).shuffle(ids)
ds = Pairs(ids[:VAL_IDS], src)
v, p = evaluate(m, ds, a["chunk"], "npu:0", None,
                objective=a.get("objective", "noise"),
                raw_lr=a.get("learned_upsample", False))
print("RESULT " + json.dumps(dict(
    arm=os.path.basename(os.path.dirname(ck)), step=blob["step"], chunk=a["chunk"],
    cond=a["cond"], lr_source=src, objective=a.get("objective", "noise"),
    lu=bool(a.get("learned_upsample", False)),
    velocity_mse=round(v, 5), latent_psnr=round(p, 3))), flush=True)
sys.stdout.flush()
os._exit(0)
