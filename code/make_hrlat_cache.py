"""Mirror the HR latents to local fp16 .npy so training can memory-map them.

Reading the .pt.part files straight off the shared /temp mount cost 0.61 s/step
against 0.22 s/step for the same recipe without a reference -- 48 clips per step
means 48 torch.load calls over NFS. The latents are only 5.3 GiB in fp16 in
total, so a local mmap removes that entirely.

Layout is [T,24,h,w] (time first) so a single frame is one contiguous slice.
"""
import os, sys, time
import numpy as np
import torch

SRC = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/high/video"
DST = os.environ["HOME"] + "/modelarts/user-job-dir/sol_h3/data/hrlat"


def main():
    os.makedirs(DST, exist_ok=True)
    ids = sorted(f[:-8] for f in os.listdir(SRC) if f.endswith(".pt.part"))
    shard, nshard = int(sys.argv[1]), int(sys.argv[2])
    mine = ids[shard::nshard]
    t0, done = time.time(), 0
    for vid in mine:
        out = f"{DST}/{vid}.npy"
        if os.path.exists(out):
            done += 1
            continue
        z = torch.load(f"{SRC}/{vid}.pt.part", map_location="cpu", weights_only=False)
        a = z.permute(1, 0, 2, 3).contiguous().to(torch.float16).numpy()
        tmp = out + ".tmp.npy"          # never let training see a partial file
        np.save(tmp, a)
        os.replace(tmp, out)
        done += 1
        if done % 25 == 0:
            print(f"[{shard}] {done}/{len(mine)}  {time.time()-t0:.0f}s", flush=True)
    print(f"[{shard}] done {done}/{len(mine)} in {time.time()-t0:.0f}s", flush=True)


main()
