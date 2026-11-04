"""End-to-end feasibility: H3 LR latent -> Wan latent -> 3x -> Wan decode -> PSNR.

Scores the existing h3_to_wan weights on our clips against the same HR pixel
reference the pixel-domain baseline uses, so the two routes are directly
comparable. Also times the steady state (first call includes op compilation).
"""
import importlib.util, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]
ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))

CKPT = "/temp/liusonghua/h3_to_wan_repro_32193/joint_mapper_upsampler_wan_dit_step_002000.pt"
WAN_PTH = "/temp/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
LOW = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/low/video"
PIX = ROOT / "data/pixels"


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def main():
    dev = "npu:0"
    from h3wan.model import Mapper
    from vendor.wan_vae import WanVAE

    sd = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)["model"]
    mapper = Mapper()
    mapper.load_state_dict({k[13:]: v for k, v in sd.items() if k.startswith("joint.mapper.")}, strict=True)
    mapper = mapper.to(dev).eval()

    src = ROOT / "h2w/vendor/UltraFlash/inference/sr/latent_upsampler/ultra_latent_up_v1.py"
    spec = importlib.util.spec_from_file_location("uf", src)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    up = mod.UltraLatentUpsampler(in_channels=16, out_channels=16, mid_channels=128,
        num_blocks=8, upsample_scale=3, activation="relu", residual=False,
        memory_init="replicate", zero_init_final=False, default_parallel=True)
    up.load_state_dict({k[16:]: v for k, v in sd.items() if k.startswith("joint.upsampler.")}, strict=True)
    up = up.to(dev).eval()
    del sd

    wan = WanVAE(vae_pth=WAN_PTH, dtype=torch.bfloat16, device=dev)
    wan.model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    print("all components loaded", flush=True)

    ids = sorted(f[:-4] for f in os.listdir(f"{PIX}/hr") if f.endswith(".npy"))
    random.Random(1234).shuffle(ids)

    rows = []
    for vid in ids[:3]:
        z = torch.load(f"{LOW}/{vid}.pt.part", map_location="cpu", weights_only=False)
        hr_np = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")

        outs, t_lat, t_dec = [], 0.0, 0.0
        # The decoder consumes 5 latent frames -> 17 pixel frames per call.
        for s in range(0, 10, 5):
            z5 = z[:, s:s + 5].unsqueeze(0).to(dev, torch.float32)
            torch.npu.synchronize(); t0 = time.time()
            with torch.no_grad():
                w = up(mapper(z5), parallel=True)
            torch.npu.synchronize(); t_lat += time.time() - t0

            t0 = time.time()
            wan.model.clear_cache()
            with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                rgb = wan.model.decode(w.to(torch.bfloat16), wan.scale)
            wan.model.clear_cache()
            torch.npu.synchronize(); t_dec += time.time() - t0
            outs.append(((rgb.float().clamp(-1, 1) + 1) / 2)[0])

        pred = torch.cat(outs, dim=1).permute(1, 0, 2, 3).cpu()   # [T,3,H,W] in [0,1]
        n = pred.shape[0]
        gt = torch.from_numpy(np.asarray(hr_np[:n])).permute(0, 3, 1, 2).float() / 255
        lr_up = F.interpolate(
            torch.from_numpy(np.asarray(np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")[:n]))
            .permute(0, 3, 1, 2).float() / 255,
            scale_factor=3, mode="bicubic", align_corners=False).clamp(0, 1)

        row = dict(clip=vid, frames=n, shape=list(pred.shape),
                   psnr=round(psnr(pred, gt), 3),
                   psnr_bicubic=round(psnr(lr_up, gt), 3),
                   latent_ms_per_latent=round(t_lat / 10 * 1000, 1),
                   decode_ms_per_latent=round(t_dec / 10 * 1000, 1))
        row["gain"] = round(row["psnr"] - row["psnr_bicubic"], 3)
        rows.append(row)
        print("ROW " + json.dumps(row), flush=True)

    print("\nMEAN psnr %.3f  bicubic %.3f  gain %+.3f" % (
        sum(r["psnr"] for r in rows) / len(rows),
        sum(r["psnr_bicubic"] for r in rows) / len(rows),
        sum(r["gain"] for r in rows) / len(rows)), flush=True)
    os._exit(0)


main()
