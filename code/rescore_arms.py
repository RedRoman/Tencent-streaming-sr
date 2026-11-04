"""Re-score every arm's checkpoint under one deterministic probe set.

The sweep was launched with a stochastic `evaluate`, whose ~0.44 dB of sampling
noise is wider than the gaps between arms, so the numbers written into each
arm's metrics.jsonl cannot be compared to each other. The checkpoints are fine;
only the measurement was noisy. This re-measures them all identically.

A deterministic probe is built per (chunk, lr_source) -- the probe has to match
the arm's own window and condition source -- and each probe is seeded the same
way, so arms that share those settings see literally identical inputs.
"""
import json
import os
import random
import sys

import torch
import torch_npu  # noqa: F401

from solh3_model import SolH3Refiner
from solh3_train import Pairs, evaluate, HIGH, VAL_IDS

SWEEP = sys.argv[1] if len(sys.argv) > 1 else "results/sweep_20261009_1316"
WHICH = sys.argv[2] if len(sys.argv) > 2 else "best.pt"


def main():
    dev = "npu:0"
    ids = sorted(f[:-8] for f in os.listdir(HIGH) if f.endswith(".pt.part"))
    random.Random(1234).shuffle(ids)

    datasets = {}
    rows = []
    for arm in sorted(os.listdir(SWEEP)):
        ck = f"{SWEEP}/{arm}/{WHICH}"
        if not os.path.exists(ck):
            continue
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        a = blob["args"]
        src = a.get("lr_source") or "h3"
        key = (a["chunk"], src)  # probe must match window and LR source
        if key not in datasets:
            datasets[key] = Pairs(ids[:VAL_IDS], src)
        model = SolH3Refiner(width=a["width"], time_taps=a["time_taps"],
                         cond_mode=a["cond"],
                         learned_upsample=a.get("learned_upsample", False)).to(dev)
        model.load_state_dict(blob["model"])
        v, p = evaluate(model, datasets[key], a["chunk"], dev, None,
                        objective=a.get("objective", "noise"))
        rows.append(dict(arm=arm, step=blob["step"], chunk=a["chunk"],
                         cond=a["cond"], lr_source=src,
                         objective=a.get("objective", "noise"),
                         velocity_mse=round(v, 5), latent_psnr=round(p, 3)))
        print(json.dumps(rows[-1]), flush=True)
        del model
        torch.npu.empty_cache()

    out = f"{SWEEP}/rescored_{WHICH.replace('.pt','')}.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out}", flush=True)
    print(f"\n{'arm':<16}{'step':>6}{'W':>4}{'LR src':>10}{'vel MSE':>10}{'PSNR':>8}")
    for r in sorted(rows, key=lambda r: -r["latent_psnr"]):
        print(f"{r['arm']:<16}{r['step']:>6}{r['chunk']:>4}{r['lr_source']:>10}"
              f"{r['velocity_mse']:>10.4f}{r['latent_psnr']:>8.3f}")
    os._exit(0)


main()
