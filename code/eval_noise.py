"""How much of the arm-to-arm gap is just evaluation noise?

`evaluate` draws random clips, crops, timesteps and noise, so repeated calls on
one unchanged checkpoint still disagree. Measure that spread: any difference
between arms smaller than it carries no information.
"""
import json
import os
import random
import statistics
import sys

import torch
import torch_npu  # noqa: F401

from solh3_model import SolH3Refiner
from solh3_train import Pairs, evaluate, HIGH, VAL_IDS

CKPT = sys.argv[1]
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 8


def main():
    dev = "npu:0"
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    targs = blob["args"]
    model = SolH3Refiner(width=targs["width"], time_taps=targs["time_taps"],
                         cond_mode=targs["cond"],
                         learned_upsample=targs.get("learned_upsample", False)).to(dev)
    model.load_state_dict(blob["model"])
    model.train()

    ids = sorted(f[:-8] for f in os.listdir(HIGH) if f.endswith(".pt.part"))
    random.Random(1234).shuffle(ids)
    val = Pairs(ids[:VAL_IDS], targs.get("lr_source", "h3"))

    psnrs, mses = [], []
    for r in range(REPEATS):
        v, p = evaluate(model, val, targs["chunk"], dev, random.Random(1000 + r))
        psnrs.append(p)
        mses.append(v)
        print(f"  repeat {r}: velocity_mse {v:.5f}  latent_psnr {p:.3f}", flush=True)

    print(json.dumps(dict(
        ckpt=CKPT, step=blob["step"], chunk=targs["chunk"], repeats=REPEATS,
        psnr_mean=round(statistics.mean(psnrs), 3),
        psnr_sd=round(statistics.stdev(psnrs), 3),
        psnr_range=round(max(psnrs) - min(psnrs), 3),
        mse_mean=round(statistics.mean(mses), 5),
        mse_sd=round(statistics.stdev(mses), 5),
    )), flush=True)
    os._exit(0)


main()
