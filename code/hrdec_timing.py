"""Why did the bench say 82 ms/chunk and the eval 764 ms for the same model?

A 9x gap between a microbenchmark and real use is the signature of the
microbenchmark measuring something easier than the real thing -- a mistake this
project has already made once (a gather backend that looked 2x faster in
isolation and gave zero end-to-end). Find which of the differences accounts for
it before trusting either number.

Differences between the two call sites, tested one at a time:
  - bench reused ONE input tensor across iterations; eval builds a fresh one
  - bench did not call .float() on the output; eval does
  - bench ran alone; the eval ran three models on three cards at once
"""
import os, time
import numpy as np
import torch
import torch_npu  # noqa: F401
from hrdec_model import HRDecoder

dev = "npu:0"
LAT = os.environ["HOME"] + "/modelarts/user-job-dir/sol_h3/data/hrlat"


def tm(fn, w=3, it=8):
    for _ in range(w):
        fn()
    torch.npu.synchronize(); t = time.time()
    for _ in range(it):
        fn()
    torch.npu.synchronize()
    return (time.time() - t) / it * 1000


def main():
    blob = torch.load("results/hrdec_1200/s384/best.pt", map_location="cpu",
                      weights_only=False)
    m = HRDecoder(blob["args"]["width"], tuple(blob["blocks"])).to(dev).eval()
    m.load_state_dict(blob["model"])
    lat = np.load(f"{LAT}/H3O20261001_0001.npy", mmap_mode="r")

    z_fixed = torch.randn(1, 24, 5, 48, 84, device=dev)
    real = torch.from_numpy(np.asarray(lat[:5], dtype=np.float32)).to(dev)
    real = real.permute(1, 0, 2, 3)[None]

    with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
        print("A random input, no .float()   %8.1f ms" % tm(lambda: m(z_fixed)), flush=True)
        print("B real latent,  no .float()   %8.1f ms" % tm(lambda: m(real)), flush=True)
        print("C real latent,  with .float() %8.1f ms" % tm(lambda: m(real).float()), flush=True)

        def fresh():
            z = torch.from_numpy(np.asarray(lat[:5], dtype=np.float32)).to(dev)
            return m(z.permute(1, 0, 2, 3)[None]).float()
        print("D fresh host copy each call   %8.1f ms" % tm(fresh), flush=True)

        def fresh_cpu():
            z = torch.from_numpy(np.asarray(lat[:5], dtype=np.float32)).to(dev)
            o = m(z.permute(1, 0, 2, 3)[None]).float()
            return o.cpu()
        print("E + .cpu() of the output      %8.1f ms" % tm(fresh_cpu), flush=True)
    os._exit(0)


main()
