"""Re-evaluate using the repro's OWN calling convention, not my hand-rolled one.

Differences I had wrong:
  - JointX3 expects 7 latent frames and splits them main(5) + tail(2), calling
    the upsampler twice with fresh causal memory each time. I fed 5 frames in a
    single call, so the upsampler's memory state was never what it was trained on.
  - The repro's geometry is LR 224x384 / HR 672x1152 (H3 latent 14x24), not our
    256x448 / 768x1344 (16x28).

Tests both conventions so the difference is attributable.
"""
import importlib.util, json, os, random, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
sys.path.insert(0, str(ROOT / "h2w/vendor/UltraFlash/inference/sr"))
MAP = "/temp/liusonghua/h3_to_wan_repro_32193/joint_mapper_upsampler_wan_dit_step_002000.pt"
TINY = "/temp/rongxiang_workspace_backup_20260921/rongxiang/UltraFlash/inference/checkpoints/tiny_decoder.pth"
WAN_PTH = "/temp/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
H3_VAE = str(ROOT / "ckpts/h3-base/vae")
PIX = ROOT / "data/pixels"


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def resize(x, h, w):
    b, c, t, H, W = x.shape
    y = F.interpolate(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, H, W), size=(h, w),
                      mode="bicubic", align_corners=False, antialias=True)
    return y.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def main():
    from h3wan.model import Mapper
    from tiny_decoder import WanTinyDecoder
    from vendor.wan_vae import WanVAE
    from diffusers import AutoencoderKLMiniMaxH3
    dev = "npu:0"

    sd = torch.load(MAP, map_location="cpu", mmap=True, weights_only=False)["model"]
    mapper = Mapper(); mapper.load_state_dict({k[13:]: v for k, v in sd.items()
        if k.startswith("joint.mapper.")}, strict=True); mapper = mapper.to(dev).eval()
    spec = importlib.util.spec_from_file_location("uf",
        ROOT / "h2w/vendor/UltraFlash/inference/sr/latent_upsampler/ultra_latent_up_v1.py")
    um = importlib.util.module_from_spec(spec); spec.loader.exec_module(um)
    up = um.UltraLatentUpsampler(in_channels=16, out_channels=16, mid_channels=128,
        num_blocks=8, upsample_scale=3, activation="relu", residual=False,
        memory_init="replicate", zero_init_final=False, default_parallel=True)
    up.load_state_dict({k[16:]: v for k, v in sd.items()
        if k.startswith("joint.upsampler.")}, strict=True); up = up.to(dev).eval(); del sd

    h3vae = AutoencoderKLMiniMaxH3.from_pretrained(H3_VAE, torch_dtype=torch.float32).to(dev).eval()
    hm = torch.tensor(h3vae.config.latents_mean, device=dev).view(1, 24, 1, 1, 1)
    hs = torch.tensor(h3vae.config.latents_std, device=dev).view(1, 24, 1, 1, 1)
    wan = WanVAE(vae_pth=WAN_PTH, dtype=torch.bfloat16, device=dev)
    wan.model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    tiny = WanTinyDecoder(TINY, dtype=torch.float32).to(dev).eval()
    print("loaded", flush=True)

    ids = sorted(f[:-4] for f in os.listdir(f"{PIX}/hr") if f.endswith(".npy"))
    random.Random(1234).shuffle(ids)

    def wan_enc(x):
        wan.model.clear_cache()
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            z = wan.model.encode(x.to(torch.bfloat16), wan.scale).float()
        wan.model.clear_cache()
        return z

    for vid in ids[:3]:
        hrpx = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")
        x22 = torch.from_numpy(np.asarray(hrpx[:22])).permute(3, 0, 1, 2).unsqueeze(0).to(dev).float() / 127.5 - 1

        for tag, (HH, WW, lh, lw) in (("repro_geom_672x1152", (672, 1152, 224, 384)),
                                      ("our_geom_768x1344", (768, 1344, 256, 448))):
            hr = resize(x22, HH, WW).clamp(-1, 1)
            lr = resize(hr, lh, lw).clamp(-1, 1)
            with torch.no_grad():
                hz = h3vae.encode(lr).latent_dist.mode()
                hz = (hz - hm) / hs
            if hz.shape[2] != 7:
                print(json.dumps(dict(clip=vid, geom=tag, note=f"H3 gave {hz.shape[2]} latents, want 7")), flush=True)
                continue
            # true Wan HR target: main 17 -> 5, tail 5 -> 2, encoded separately
            w_hr = torch.cat([wan_enc(hr[:, :, :17]), wan_enc(hr[:, :, 17:22])], dim=2)
            with torch.no_grad():
                low = mapper(hz.float())
                # NATIVE: two upsampler calls with fresh memory
                hi_native = torch.cat([up(low[:, :, :5], parallel=True),
                                       up(low[:, :, 5:], parallel=True)], dim=2)
                # MINE: one call over all 7
                hi_single = up(low, parallel=True)
            n = 17
            gt = hr[:, :, :n]
            row = dict(clip=vid, geom=tag, h3_latent=list(hz.shape[1:]),
                       wan_hr=list(w_hr.shape[1:]),
                       latent_native=round(psnr(hi_native, w_hr), 2),
                       latent_single=round(psnr(hi_single, w_hr), 2),
                       )
            # true Wan LR, same main/tail split as the HR target
            w_lr = torch.cat([wan_enc(lr[:, :, :17]), wan_enc(lr[:, :, 17:22])], dim=2)
            row["latent_mapper_vs_trueWanLR"] = round(psnr(low, w_lr), 2)
            with torch.no_grad():
                for k, z in (("native", hi_native), ("single", hi_single), ("true", w_hr)):
                    px = tiny.decode_latents(z.float()).float().clamp(-1, 1)
                    row[f"px_{k}"] = round(psnr(px[:, :, :n], gt), 2)
            bic = resize(lr, HH, WW).clamp(-1, 1)
            row["px_bicubic"] = round(psnr(bic[:, :, :n], gt), 2)
            print(json.dumps(row), flush=True)
            torch.npu.empty_cache()
    os._exit(0)


main()
