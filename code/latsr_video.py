"""Render latent SR full-frame so the structure gap can be seen, not just read.

The numbers say latent SR loses 4.9 dB of STRUCTURE (29.16 vs 34.06) while
losing only 2.0 dB of detail. That distinction decides the whole route, and it
is the kind of claim that should be checkable by eye: structural error looks
like content in the wrong place, detail error looks like softness.

Writes [T,768,1344,3] uint8 for one clip, at the keyframe cadence the
checkpoint was trained for.
"""
import os, random, sys
import numpy as np
import torch
import torch_npu  # noqa: F401
from latsr_model import LatentSR
from hrdec_model import HRDecoder
from latsr_train import HLAT, PIX, LLAT, N_LAT, N_FRAMES, FRAMES_PER_LAT

dev = "npu:0"


def main():
    ck, vid, out = sys.argv[1], sys.argv[2], sys.argv[3]
    b = torch.load(ck, map_location="cpu", weights_only=False)
    a = b["args"]
    m = LatentSR(a["width"], a["lr_blocks"], a["hr_blocks"]).to(dev).eval()
    m.load_state_dict(b["model"])
    db = torch.load(a["decoder"], map_location="cpu", weights_only=False)
    dec = HRDecoder(db["args"]["width"], tuple(db["blocks"])).to(dev).eval()
    dec.load_state_dict(db["model"])
    with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
        for _ in range(3):
            dec(torch.zeros(1, 24, N_LAT, 48, 84, device=dev))
    torch.npu.synchronize()
    kf_every = a.get("kf_chunks", 2) + 1
    print(f"{vid}: keyframe every {kf_every} chunks = {kf_every*N_FRAMES} frames",
          flush=True)

    hl = np.load(f"{HLAT}/{vid}.npy", mmap_mode="r")
    px = np.load(f"{PIX}/{vid}.npy", mmap_mode="r")
    ll = torch.load(f"{LLAT}/{vid}.pt.part", map_location="cpu",
                    weights_only=False).permute(1, 0, 2, 3).float()
    n = min(hl.shape[0] // N_LAT, px.shape[0] // N_FRAMES, ll.shape[0] // N_LAT)
    frames = []
    for k in range(n):
        kk = (k // kf_every) * kf_every
        d = torch.tensor([[(k - kk) * N_LAT * FRAMES_PER_LAT]], device=dev)
        lo = ll[N_LAT * k:N_LAT * k + N_LAT].to(dev)
        kf = torch.from_numpy(np.asarray(hl[N_LAT * kk], dtype=np.float32))[None].to(dev)
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            z = m(lo, N_LAT, m.encode_key(kf), kf.repeat_interleave(N_LAT, 0), d)
            o = dec(z.permute(1, 0, 2, 3)[None].contiguous()).float()
        frames.append(((o.permute(0, 2, 3, 1) + 1) * 127.5)
                      .round().clamp(0, 255).byte().cpu().numpy())
        torch.npu.empty_cache()
    arr = np.concatenate(frames)
    np.save(out, arr)
    print(f"wrote {out}  {arr.shape}", flush=True)
    os._exit(0)


main()
