"""Clean re-evaluation with every known calling convention corrected.

Fixed vs my earlier attempts:
  - 7-latent window (not 5), H3 token_drop=0 so 22 frames -> 7 latents
  - upsampler called twice (main 5 / tail 2) with fresh causal memory
  - tiny decoder called twice the same way, never once over 7
  - true Wan target encoded main 17->5 and tail 5->2 separately
Reports the mapper's own accuracy against the true Wan LR latent, which is the
quantity its training actually optimized.
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

    # token_drop=0 is what the repro uses for this contract (22 frames -> 7 latents)
    h3vae = AutoencoderKLMiniMaxH3.from_pretrained(
        H3_VAE, torch_dtype=torch.float32, token_drop=3).to(dev).eval()
    assert h3vae.config.token_drop == 3, h3vae.config.token_drop
    hm = torch.tensor(h3vae.config.latents_mean, device=dev).view(1, 24, 1, 1, 1)
    hs = torch.tensor(h3vae.config.latents_std, device=dev).view(1, 24, 1, 1, 1)
    wan = WanVAE(vae_pth=WAN_PTH, dtype=torch.bfloat16, device=dev)
    wan.model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    tiny = WanTinyDecoder(TINY, dtype=torch.float32).to(dev).eval()

    def wan_enc(x):
        wan.model.clear_cache()
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            z = wan.model.encode(x.to(torch.bfloat16), wan.scale).float()
        wan.model.clear_cache()
        return z

    def dec7(z):
        out = []
        for part in (z[:, :, :5], z[:, :, 5:]):
            out.append(tiny.decode_latents(part.float()).float().clamp(-1, 1))
        return torch.cat(out, dim=2)

    ids = sorted(f[:-4] for f in os.listdir(f"{PIX}/hr") if f.endswith(".npy"))
    random.Random(1234).shuffle(ids)
    rows = []
    for vid in ids[:4]:
        hrpx = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")
        hr = torch.from_numpy(np.asarray(hrpx[:22])).permute(3, 0, 1, 2).unsqueeze(0).to(dev).float() / 127.5 - 1
        lr = resize(hr, 256, 448).clamp(-1, 1)
        with torch.no_grad():
            hz = h3vae.encode(lr).latent_dist.mode()
        hz = (hz - hm) / hs
        w_hr = torch.cat([wan_enc(hr[:, :, :17]), wan_enc(hr[:, :, 17:22])], dim=2)
        w_lr = torch.cat([wan_enc(lr[:, :, :17]), wan_enc(lr[:, :, 17:22])], dim=2)
        if hz.shape[2] != 7:
            print(json.dumps(dict(clip=vid, note=f"h3 latents={hz.shape[2]}")), flush=True)
            continue
        with torch.no_grad():
            low = mapper(hz.float())
            hi = torch.cat([up(low[:, :, :5], parallel=True),
                            up(low[:, :, 5:], parallel=True)], dim=2)
            hi_true = torch.cat([up(w_lr[:, :, :5], parallel=True),
                                 up(w_lr[:, :, 5:], parallel=True)], dim=2)
        n = 22
        row = dict(clip=vid,
            mapper_vs_trueWanLR=round(psnr(low, w_lr), 2),
            mapped_up_vs_trueWanHR=round(psnr(hi, w_hr), 2),
            trueLR_up_vs_trueWanHR=round(psnr(hi_true, w_hr), 2),
            px_true_wan=round(psnr(dec7(w_hr)[:, :, :n], hr[:, :, :n]), 2),
            px_mapped=round(psnr(dec7(hi)[:, :, :n], hr[:, :, :n]), 2),
            px_trueLR_up=round(psnr(dec7(hi_true)[:, :, :n], hr[:, :, :n]), 2),
            px_bicubic=round(psnr(resize(lr, 768, 1344).clamp(-1, 1)[:, :, :n], hr[:, :, :n]), 2))
        rows.append(row); print(json.dumps(row), flush=True)
        torch.npu.empty_cache()
    if rows:
        a = lambda k: round(sum(r[k] for r in rows) / len(rows), 2)
        print("\nMEAN " + json.dumps({k: a(k) for k in rows[0] if k != "clip"}), flush=True)
    os._exit(0)


main()
