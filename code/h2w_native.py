"""Score the h3_to_wan weights at the geometry they were trained on.

Training used LR 224x384 / HR 672x1152 and the 7-frame protocol (main 5 + tail 2).
Our earlier probe fed 256x448 / 768x1344 with 5 frames -- a cross-scale
extrapolation. Rebuild the native setting from the HR pixels we already have:
resize HR to 672x1152, make the LR by antialiased bicubic to 224x384, encode LR
with the H3 VAE, and compare the route's output against the Wan encode of HR.
"""
import importlib.util, json, os, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch_npu  # noqa: F401
from diffusers import AutoencoderKLMiniMaxH3

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
CKPT = "/temp/liusonghua/h3_to_wan_repro_32193/joint_mapper_upsampler_wan_dit_step_002000.pt"
WAN_PTH = "/temp/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
H3_VAE = f"{HOME}/modelarts/user-job-dir/sol_h3/ckpts/h3-base/vae"
PIX = ROOT / "data/pixels"


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def resize(x, h, w):
    b, c, t, H, W = x.shape
    y = F.interpolate(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, H, W),
                      size=(h, w), mode="bicubic", align_corners=False, antialias=True)
    return y.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def main():
    dev = "npu:0"
    from h3wan.model import Mapper
    from vendor.wan_vae import WanVAE

    sd = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)["model"]
    mapper = Mapper(); mapper.load_state_dict(
        {k[13:]: v for k, v in sd.items() if k.startswith("joint.mapper.")}, strict=True)
    mapper = mapper.to(dev).eval()
    spec = importlib.util.spec_from_file_location("uf",
        ROOT / "h2w/vendor/UltraFlash/inference/sr/latent_upsampler/ultra_latent_up_v1.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    up = mod.UltraLatentUpsampler(in_channels=16, out_channels=16, mid_channels=128,
        num_blocks=8, upsample_scale=3, activation="relu", residual=False,
        memory_init="replicate", zero_init_final=False, default_parallel=True)
    up.load_state_dict({k[16:]: v for k, v in sd.items() if k.startswith("joint.upsampler.")}, strict=True)
    up = up.to(dev).eval(); del sd

    h3 = AutoencoderKLMiniMaxH3.from_pretrained(H3_VAE, torch_dtype=torch.float32).to(dev).eval()
    hm = torch.tensor(h3.config.latents_mean, device=dev).view(1, 24, 1, 1, 1)
    hs = torch.tensor(h3.config.latents_std, device=dev).view(1, 24, 1, 1, 1)
    wan = WanVAE(vae_pth=WAN_PTH, dtype=torch.bfloat16, device=dev)
    wan.model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    print("loaded", flush=True)

    ids = sorted(f[:-4] for f in os.listdir(f"{PIX}/hr") if f.endswith(".npy"))
    random.Random(1234).shuffle(ids)
    rows = []
    for vid in ids[:3]:
        hrpx = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")
        # 22 original frames -> native training geometry
        x = torch.from_numpy(np.asarray(hrpx[:22])).permute(3, 0, 1, 2).unsqueeze(0).float().to(dev) / 255
        hr = resize(x, 672, 1152).clamp(0, 1)
        lr = resize(hr, 224, 384).clamp(0, 1)

        with torch.no_grad():
            hz = h3.encode(lr * 2 - 1).latent_dist.mode()
            hz = (hz - hm) / hs
        print(f"{vid}: H3 LR latent {tuple(hz.shape)}", flush=True)
        if hz.shape[2] != 7:
            print("  unexpected temporal length; skipping", flush=True); continue

        torch.npu.synchronize(); t0 = time.time()
        with torch.no_grad():
            low = mapper(hz.float())
            high = torch.cat([up(low[:, :, :5], parallel=True),
                              up(low[:, :, 5:], parallel=True)], dim=2)
        torch.npu.synchronize(); t_lat = time.time() - t0

        # Reference: Wan's own encode of the HR pixels, main 17F + tail 5F
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            wan.model.clear_cache(); zm = wan.model.encode((hr[:, :, :17] * 2 - 1).to(torch.bfloat16), wan.scale).float()
            wan.model.clear_cache(); zt = wan.model.encode((hr[:, :, 17:22] * 2 - 1).to(torch.bfloat16), wan.scale).float()
            wan.model.clear_cache()
        ref = torch.cat([zm, zt], dim=2)

        # Decode both and compare in pixels
        def dec(z):
            outs = []
            for part in (z[:, :, :5], z[:, :, 5:]):
                wan.model.clear_cache()
                with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                    r = wan.model.decode(part.to(torch.bfloat16), wan.scale)
                wan.model.clear_cache()
                outs.append((r.float().clamp(-1, 1) + 1) / 2)
            return torch.cat(outs, dim=2)

        pred_px, ref_px = dec(high), dec(ref)
        n = min(pred_px.shape[2], hr.shape[2])
        lr_up = resize(lr, 672, 1152).clamp(0, 1)

        row = dict(clip=vid,
                   latent_shape=list(high.shape),
                   latent_psnr_vs_wan_encode=round(psnr(high, ref), 3),
                   pred_std=round(float(high.std()), 3), ref_std=round(float(ref.std()), 3),
                   pixel_psnr=round(psnr(pred_px[:, :, :n], hr[:, :, :n]), 3),
                   pixel_psnr_bicubic=round(psnr(lr_up[:, :, :n], hr[:, :, :n]), 3),
                   wan_ceiling=round(psnr(ref_px[:, :, :n], hr[:, :, :n]), 3),
                   latent_ms=round(t_lat * 1000, 1))
        row["gain"] = round(row["pixel_psnr"] - row["pixel_psnr_bicubic"], 3)
        rows.append(row); print("ROW " + json.dumps(row), flush=True)

    if rows:
        print("\nMEAN pixel %.3f | bicubic %.3f | gain %+.3f | wan ceiling %.3f" % (
            sum(r["pixel_psnr"] for r in rows) / len(rows),
            sum(r["pixel_psnr_bicubic"] for r in rows) / len(rows),
            sum(r["gain"] for r in rows) / len(rows),
            sum(r["wan_ceiling"] for r in rows) / len(rows)), flush=True)
    os._exit(0)


main()
