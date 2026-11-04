"""Can H3 itself produce the keyframe latent inside the budget?

The relay needs one HR latent per interval. Decoding it is already ruled out
(5847 ms), but the latent reader does not decode -- so the remaining cost is the
H3 transformer forward at HR. 33B, 50 layers, hidden 5376: the question is first
whether it even fits on one 61 GiB card, and only then how fast it is.

Reports what is true rather than what would be convenient: if the weights do not
fit, that is the answer and the number of layers it CAN run bounds the rest.
"""
import json, os, time
import torch
import torch_npu  # noqa: F401
from safetensors import safe_open

CK = "ckpts/h3-base/transformer"


def main():
    import glob
    shards = sorted(glob.glob(f"{CK}/*.safetensors"))
    tot, dt = 0, None
    for sh in shards:
        with safe_open(sh, framework="pt") as f:
            for k in f.keys():
                s = f.get_slice(k)
                n = 1
                for d in s.get_shape():
                    n *= d
                tot += n
                dt = dt or s.get_dtype()
    gib = sum(os.path.getsize(s) for s in shards) / 2**30
    print(f"shards {len(shards)}  params {tot/1e9:.2f}B  dtype {dt}  on-disk {gib:.1f} GiB",
          flush=True)
    free = 61.0
    print(f"card usable ~{free} GiB -> fits on one card: {gib < free}", flush=True)
    json.dump(dict(params_b=round(tot/1e9, 2), dtype=str(dt), disk_gib=round(gib, 1),
                   fits_single_card=bool(gib < free)),
              open("results/h3fwd_cost.json", "w"), indent=2)
    os._exit(0)


main()
