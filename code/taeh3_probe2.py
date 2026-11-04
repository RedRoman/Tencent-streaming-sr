"""Find the right latent convention and frame count for taeh3.

Two suspects for the low PSNR:
  (a) normalization -- TAEHV applies no latent scale/shift, so it may expect the
      NORMALIZED latent we store, not the de-normalized one the diffusers VAE wants
  (b) frame count -- 5 latents came back as 5 frames, but t4 upsampling should
      give 17; the temporal stages may need more input frames to engage
Sweeps both against the real VAE decode.
"""
import json, os, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "assets/taehv"))
LOW = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/low/video"
PIX = ROOT / "data/pixels"
VAE_DIR = str(ROOT / "ckpts/h3-base/vae")


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def main():
    from taehv import TAEHV
    from diffusers import AutoencoderKLMiniMaxH3
    dev = "npu:0"
    tae = TAEHV(str(ROOT / "assets/taehv/taeh3.pth")).to(dev).eval()
    vae = AutoencoderKLMiniMaxH3.from_pretrained(VAE_DIR, torch_dtype=torch.float32).to(dev).eval()
    mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)

    ids = sorted(f[:-4] for f in os.listdir(f"{PIX}/hr") if f.endswith(".npy"))
    random.Random(1234).shuffle(ids)
    vid = ids[0]
    z_all = torch.load(f"{LOW}/{vid}.pt.part", map_location="cpu", weights_only=False)
    lrpx = np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")

    for T in (5, 8, 9):
        znorm = z_all[:, :T].unsqueeze(0).to(dev, torch.float32)   # stored normalized
        zraw = znorm * std + mean                                   # de-normalized
        with torch.no_grad():
            ref = vae.decode(zraw).sample.float().clamp(-1, 1)      # [1,3,F,256,448]
        nf = ref.shape[2]
        gt = torch.from_numpy(np.asarray(lrpx[:nf])).permute(3, 0, 1, 2).unsqueeze(0).to(dev).float() / 127.5 - 1

        for tag, z in (("normalized", znorm), ("denormalized", zraw)):
            zt = z.permute(0, 2, 1, 3, 4).contiguous()              # NTCHW
            with torch.no_grad():
                o = tae.decode_video(zt, parallel=True, show_progress_bar=False)
            o = o.permute(0, 2, 1, 3, 4).float()                    # back to NCTHW, [0,1]
            o11 = (o.clamp(0, 1) * 2 - 1)
            k = min(o11.shape[2], nf)
            print(json.dumps(dict(T_latent=T, latent=tag,
                vae_frames=nf, tae_frames=int(o11.shape[2]),
                tae_vs_vae=round(psnr(o11[:, :, :k], ref[:, :, :k]), 2),
                tae_vs_gt=round(psnr(o11[:, :, :k], gt[:, :, :k]), 2),
                vae_vs_gt=round(psnr(ref[:, :, :k], gt[:, :, :k]), 2),
                out_range=[round(float(o.min()), 2), round(float(o.max()), 2)])), flush=True)
        torch.npu.empty_cache()
    os._exit(0)


main()
