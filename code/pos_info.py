"""How much POSITIONAL information does each route start with?

The latent model moves its content -- it is not stuck on the keyframe -- but it
lands in the wrong place, and the penalty scales with how much the region moved.
Its only source of current-frame position is the LR latent, lifted 3x onto the
HR latent grid. The pixel route's is the LR frame, lifted 3x by bicubic.

So compare the two STARTING POINTS, before any network: structure fidelity of
  latent route   decode( trilinear_3x( LR latent ) )
  pixel route    bicubic_3x( LR pixels )
Structure is the low band (8x downsample), i.e. exactly where the 5-dB deficit
was localised. Whatever difference shows up here is information the latent route
never had, as opposed to information its model failed to use.

The grid arithmetic that motivates the test:
  LR latent cell = 16 LR px = 48 HR px   <- explicit position every 48 px
  LR pixel       = 3 HR px               <- explicit position every 3 px
Displacement finer than a cell has to be read out of the 24-channel code.
"""
import json, os, random
import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from hrdec_model import HRDecoder
from hrdec_train import LAT, PIX, N_LAT, N_FRAMES, psnr_from_mse

dev = "npu:0"
LLAT = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/low/video"
LRPIX = os.environ["HOME"] + "/modelarts/user-job-dir/sol_h3/data/pixels/lr"


def bands(p, g):
    lo_p, lo_g = F.avg_pool2d(p, 8), F.avg_pool2d(g, 8)
    struct = psnr_from_mse(F.mse_loss(lo_p, lo_g))
    hi_p = p - F.interpolate(lo_p, size=p.shape[-2:], mode="bilinear", align_corners=False)
    hi_g = g - F.interpolate(lo_g, size=g.shape[-2:], mode="bilinear", align_corners=False)
    return struct, psnr_from_mse(F.mse_loss(hi_p, hi_g))


def main():
    db = torch.load("results/hrdec_1200/s768/best.pt", map_location="cpu",
                    weights_only=False)
    dec = HRDecoder(db["args"]["width"], tuple(db["blocks"])).to(dev).eval()
    dec.load_state_dict(db["model"])
    with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
        for _ in range(3):
            dec(torch.zeros(1, 24, N_LAT, 48, 84, device=dev))
    torch.npu.synchronize()

    ids = sorted({f[:-4] for f in os.listdir(LAT) if f.endswith(".npy")} &
                 {f[:-4] for f in os.listdir(PIX) if f.endswith(".npy")})
    random.Random(1234).shuffle(ids)
    acc = {k: {"s": [], "d": []} for k in ("latent_tri", "pixel_bicubic", "latent_true")}
    for vid in ids[:4]:
        hl = np.load(f"{LAT}/{vid}.npy", mmap_mode="r")
        px = np.load(f"{PIX}/{vid}.npy", mmap_mode="r")
        lrp = np.load(f"{LRPIX}/{vid}.npy", mmap_mode="r")
        ll = torch.load(f"{LLAT}/{vid}.pt.part", map_location="cpu",
                        weights_only=False).permute(1, 0, 2, 3).float()
        n = min(hl.shape[0] // N_LAT, px.shape[0] // N_FRAMES, ll.shape[0] // N_LAT)
        for k in range(0, n, max(n // 3, 1)):
            gt = torch.from_numpy(np.asarray(px[N_FRAMES*k:N_FRAMES*k+N_FRAMES])
                                  ).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(dev)
            lo = ll[N_LAT*k:N_LAT*k+N_LAT].to(dev)
            up = F.interpolate(lo.permute(1, 0, 2, 3)[None], size=(N_LAT, 48, 84),
                               mode="trilinear", align_corners=False)[0]
            t = torch.from_numpy(np.asarray(hl[N_LAT*k:N_LAT*k+N_LAT],
                                            dtype=np.float32)).to(dev)
            with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                d_tri = dec(up[None].contiguous()).float()
                d_true = dec(t.permute(1, 0, 2, 3)[None].contiguous()).float()
            lrf = torch.from_numpy(np.asarray(lrp[N_FRAMES*k:N_FRAMES*k+N_FRAMES])
                                   ).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(dev)
            bic = F.interpolate(lrf, scale_factor=3, mode="bicubic",
                                align_corners=False).clamp(-1, 1)
            for name, p in (("latent_tri", d_tri), ("pixel_bicubic", bic),
                            ("latent_true", d_true)):
                s, d = bands(p[:gt.shape[0]], gt[:p.shape[0]])
                acc[name]["s"].append(s); acc[name]["d"].append(d)
            del gt, d_tri, d_true, bic, lrf
            torch.npu.empty_cache()

    print("%-18s %11s %11s" % ("starting point", "structure", "detail"), flush=True)
    out = {}
    for k, v in acc.items():
        s = sum(v["s"]) / len(v["s"]); d = sum(v["d"]) / len(v["d"])
        out[k] = dict(struct=round(s, 2), detail=round(d, 2))
        print("%-18s %11.2f %11.2f" % (k, s, d), flush=True)
    json.dump(out, open("results/pos_info.json", "w"), indent=2)
    os._exit(0)


main()
