"""Where does latent SR's 5.4 dB PSNR deficit live -- structure or texture?

Full-frame LPIPS is close to the pixel pipeline's (0.2378 vs 0.1910) while PSNR
is far off (22.0 vs 27.2). Those two disagree for exactly two reasons, and they
have opposite consequences:

  low-frequency error   the picture has DRIFTED -- shapes, edges and brightness
                        sit in the wrong place. LPIPS tolerates it, viewers do
                        not, and "super-resolution" stops being an honest label.
  high-frequency error  the texture is invented but plausible. That is the
                        expected, acceptable cost of generating detail that 3x
                        downsampling destroyed.

So split the error by band: downsample both by 8 and score the structure alone,
then score what is left. Compared against the shipping pixel pipeline on the
same clips and frames, since the question is relative.
"""
import json, os, random, sys
import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from latsr_model import LatentSR
from hrdec_model import HRDecoder
from latsr_train import HLAT, PIX, LLAT, N_LAT, N_FRAMES, FRAMES_PER_LAT, psnr_from_mse

dev = "npu:0"
W_PANEL = 1344


def bands(p, g):
    """(structure PSNR after 8x downsample, residual-detail PSNR)."""
    lo_p = F.avg_pool2d(p, 8)
    lo_g = F.avg_pool2d(g, 8)
    struct = psnr_from_mse(F.mse_loss(lo_p, lo_g))
    hi_p = p - F.interpolate(lo_p, size=p.shape[-2:], mode="bilinear", align_corners=False)
    hi_g = g - F.interpolate(lo_g, size=g.shape[-2:], mode="bilinear", align_corners=False)
    detail = psnr_from_mse(F.mse_loss(hi_p, hi_g))
    return struct, detail


def main():
    ck = sys.argv[1]
    clips = int(sys.argv[2]) if len(sys.argv) > 2 else 4
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

    ids = sorted({f[:-4] for f in os.listdir(HLAT) if f.endswith(".npy")} &
                 {f[:-4] for f in os.listdir(PIX) if f.endswith(".npy")} &
                 {f[:-8] for f in os.listdir(LLAT) if f.endswith(".pt.part")})
    random.Random(1234).shuffle(ids)
    # Keyframe cadence must match what the model was TRAINED for. Hardcoding
    # it meant k0 (trained only at distance 0) was fed 33-frame-old keyframes
    # it had never seen and scored 8.3 dB -- a measurement of extrapolation
    # failure, not of keyframe density. Training with kf_chunks=E-1 corresponds
    # to a keyframe every E chunks, so derive it from the checkpoint.
    kf_every = int(sys.argv[3]) if len(sys.argv) > 3 else (a.get("kf_chunks", 2) + 1)
    print(f"  keyframe every {kf_every} chunks = {kf_every*N_FRAMES} frames "
          f"(trained kf_chunks={a.get('kf_chunks')})", flush=True)
    acc = {k: {"struct": [], "detail": [], "psnr": []}
           for k in ("latent_sr", "pixel_pipeline", "latent_true")}

    for vid in ids[:clips]:
        trip = f"results/ev4_lat30_kf30/{vid}_triptych.npy"
        if not os.path.exists(trip):
            continue
        hl = np.load(f"{HLAT}/{vid}.npy", mmap_mode="r")
        px = np.load(f"{PIX}/{vid}.npy", mmap_mode="r")
        ll = torch.load(f"{LLAT}/{vid}.pt.part", map_location="cpu",
                        weights_only=False).permute(1, 0, 2, 3).float()
        tri = np.load(trip, mmap_mode="r")
        n = min(hl.shape[0] // N_LAT, px.shape[0] // N_FRAMES, ll.shape[0] // N_LAT)
        ours, true_lat = [], []
        for k in range(n):
            kk = (k // kf_every) * kf_every
            d = torch.tensor([[(k - kk) * N_LAT * FRAMES_PER_LAT]], device=dev)
            lo = ll[N_LAT * k:N_LAT * k + N_LAT].to(dev)
            kf = torch.from_numpy(np.asarray(hl[N_LAT * kk], dtype=np.float32))[None].to(dev)
            t = torch.from_numpy(np.asarray(hl[N_LAT*k:N_LAT*k+N_LAT],
                                            dtype=np.float32)).to(dev)
            with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                z = m(lo, N_LAT, m.encode_key(kf), kf.repeat_interleave(N_LAT, 0), d)
                ours.append(dec(z.permute(1, 0, 2, 3)[None].contiguous()).float().cpu())
                true_lat.append(dec(t.permute(1, 0, 2, 3)[None].contiguous()).float().cpu())
            torch.npu.empty_cache()
        T = n * N_FRAMES
        gt = torch.from_numpy(np.asarray(px[:T])).permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
        # panel 1 of the triptych is the shipping pixel model's output
        pipe = torch.from_numpy(np.asarray(tri[:T, :, W_PANEL:2*W_PANEL]))
        pipe = pipe.permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
        for name, p in (("latent_sr", torch.cat(ours)), ("pixel_pipeline", pipe),
                        ("latent_true", torch.cat(true_lat))):
            s, dt = bands(p.to(dev), gt.to(dev))
            acc[name]["struct"].append(s); acc[name]["detail"].append(dt)
            acc[name]["psnr"].append(psnr_from_mse(F.mse_loss(p, gt)))
        del ours, true_lat, gt, pipe
        torch.npu.empty_cache()

    print("%-16s %10s %10s %10s" % ("", "psnr", "structure", "detail"), flush=True)
    res = {}
    for k, v in acc.items():
        if not v["psnr"]:
            continue
        r = {m2: round(sum(v[m2]) / len(v[m2]), 3) for m2 in v}
        res[k] = r
        print("%-16s %10.3f %10.3f %10.3f" % (k, r["psnr"], r["struct"], r["detail"]),
              flush=True)
    json.dump(res, open("results/latsr_gap.json", "w"), indent=2)
    os._exit(0)


main()
