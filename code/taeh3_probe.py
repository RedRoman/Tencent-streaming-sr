"""Can madebyollin's taeh3 replace the H3 VAE decode in our streaming path?

The H3 VAE decode is 424 ms of the 558 ms streaming chunk -- 76% of the budget.
taeh3 is a TAEHV trained for H3 latents, pure PyTorch (no custom CUDA ops), MIT.
Measure both correctness against the real decode and speed, on our own latents.

TAEHV conventions differ from diffusers: NTCHW ordering, image range [0,1], and
no latent scale/shift, so the stored normalized latent must be denormalized first.
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
    n = sum(p.numel() for p in tae.parameters())
    print(json.dumps(dict(taeh3_params_M=round(n / 1e6, 2),
                          time_upscale=list(getattr(tae, "decoder_time_upscale", [])),
                          space_upscale=list(getattr(tae, "decoder_space_upscale", [])))), flush=True)

    vae = AutoencoderKLMiniMaxH3.from_pretrained(VAE_DIR, torch_dtype=torch.float32).to(dev).eval()
    mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)

    ids = sorted(f[:-4] for f in os.listdir(f"{PIX}/hr") if f.endswith(".npy"))
    random.Random(1234).shuffle(ids)

    def tm(fn, warm=2, it=5):
        for _ in range(warm):
            fn()
        torch.npu.synchronize(); t0 = time.time()
        for _ in range(it):
            fn()
        torch.npu.synchronize()
        return (time.time() - t0) / it * 1000

    for vid in ids[:3]:
        z = torch.load(f"{LOW}/{vid}.pt.part", map_location="cpu", weights_only=False)
        zc = z[:, :5].unsqueeze(0).to(dev, torch.float32)
        raw = zc * std + mean                      # de-normalize for both decoders
        lrpx = np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")

        with torch.no_grad():
            ref = vae.decode(raw).sample.float().clamp(-1, 1)
            ms_vae = tm(lambda: vae.decode(raw).sample)
            # TAEHV wants NTCHW and returns [0,1]
            zt = raw.permute(0, 2, 1, 3, 4).contiguous()
            out = tae.decode_video(zt, parallel=True)
            ms_tae = tm(lambda: tae.decode_video(zt, parallel=True))

        o = out.permute(0, 2, 1, 3, 4).float().clamp(0, 1) * 2 - 1   # back to NCTHW,[-1,1]
        k = min(o.shape[2], ref.shape[2])
        gt = torch.from_numpy(np.asarray(lrpx[:k])).permute(3, 0, 1, 2).unsqueeze(0).to(dev).float() / 127.5 - 1
        print(json.dumps(dict(clip=vid,
            latent=list(zc.shape[1:]), vae_out=list(ref.shape[1:]), tae_out=list(o.shape[1:]),
            tae_vs_vae=round(psnr(o[:, :, :k], ref[:, :, :k]), 2),
            tae_vs_gt=round(psnr(o[:, :, :k], gt), 2),
            vae_vs_gt=round(psnr(ref[:, :, :k], gt), 2),
            vae_ms=round(ms_vae, 1), tae_ms=round(ms_tae, 1),
            speedup=round(ms_vae / max(ms_tae, 1e-6), 1))), flush=True)
        torch.npu.empty_cache()
    os._exit(0)


main()
