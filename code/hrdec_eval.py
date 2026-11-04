"""Does the distilled decoder hold up end to end, and against what bar?

Training scored 192px crops. Three things that cannot be read off that number:
  1. full-frame quality -- crops avoid the tile seams the teacher has at 1344px
  2. how it compares to the bar that matters. The student is only worth having
     if decoding an HR latent through it beats what we already ship: the pixel
     refiner working from an LR decode. That is the real competitor, not the
     teacher.
  3. temporal flicker, which a per-chunk student can break without the PSNR
     moving much.
"""
import json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from hrdec_model import HRDecoder
from hrdec_train import LAT, PIX, VAL_CLIPS, N_LAT, N_FRAMES, psnr_from_mse

dev = "npu:0"
_LP = None


def lp(a, b):
    global _LP
    if _LP is None:
        import lpips as pkg
        _LP = pkg.LPIPS(net="vgg").to(dev).eval()
        for q in _LP.parameters():
            q.requires_grad_(False)
    with torch.no_grad():
        return float(torch.cat([_LP(a[i:i+4], b[i:i+4]).flatten()
                                for i in range(0, a.shape[0], 4)]).mean())


def flicker(v):
    return float((v[1:] - v[:-1]).abs().mean() / 2)


def main():
    ck = Path(sys.argv[1])
    clips = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    a = blob["args"]
    m = HRDecoder(a["width"], tuple(blob["blocks"])).to(dev).eval()
    m.load_state_dict(blob["model"])
    print(f"{ck}  step {blob['step']}  {sum(p.numel() for p in m.parameters())/1e6:.2f}M",
          flush=True)

    # Warm the kernels before anything is timed. Without this the first chunk
    # carries the whole compile cost and, over only ~7 chunks, drags the mean
    # from 82 ms to 786 ms -- the same cold-start artefact that made the
    # keyframe encoder look like a 1338 ms operation when it costs 4 ms.
    with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
        for _ in range(3):
            m(torch.zeros(1, 24, N_LAT, 48, 84, device=dev))
    torch.npu.synchronize()

    ids = sorted({f[:-4] for f in os.listdir(LAT) if f.endswith(".npy")} &
                 {f[:-4] for f in os.listdir(PIX) if f.endswith(".npy")})
    random.Random(1234).shuffle(ids)
    rows = []
    for vid in ids[:clips]:
        lat = np.load(f"{LAT}/{vid}.npy", mmap_mode="r")
        pix = np.load(f"{PIX}/{vid}.npy", mmap_mode="r")
        n_chunk = min(lat.shape[0] // N_LAT, pix.shape[0] // N_FRAMES)
        outs, ms = [], 0.0
        for k in range(n_chunk):
            z = torch.from_numpy(np.asarray(
                lat[N_LAT*k:N_LAT*k+N_LAT], dtype=np.float32)).to(dev)
            z = z.permute(1, 0, 2, 3)[None]           # [1,24,5,h,w]
            torch.npu.synchronize(); t0 = time.time()
            with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                o = m(z).float()
            torch.npu.synchronize(); ms += (time.time() - t0) * 1000
            outs.append(o.cpu())
        pred = torch.cat(outs)
        gt = torch.from_numpy(np.asarray(pix[:pred.shape[0]])).permute(0, 3, 1, 2)
        gt = gt.float().div(127.5).sub(1.0)
        row = dict(clip=vid, frames=int(pred.shape[0]),
                   psnr=round(psnr_from_mse(F.mse_loss(pred, gt)), 3),
                   lpips=round(lp(pred.to(dev), gt.to(dev)), 4),
                   flicker=round(flicker(pred), 5),
                   flicker_gt=round(flicker(gt), 5),
                   ms_per_chunk=round(ms / n_chunk, 1))
        rows.append(row)
        print(json.dumps(row), flush=True)
        del pred, gt, outs
        torch.npu.empty_cache()
    agg = {k: round(sum(r[k] for r in rows) / len(rows), 4)
           for k in ("psnr", "lpips", "flicker", "flicker_gt", "ms_per_chunk")}
    print("\nMEAN " + json.dumps(agg), flush=True)
    out = ck.parent / "eval_full.json"
    out.write_text(json.dumps(dict(ckpt=str(ck), mean=agg, rows=rows), indent=2))
    os._exit(0)


main()
