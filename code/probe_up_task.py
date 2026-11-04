"""Is Wan-latent 3x upsampling learnable at all, independent of UltraFlash?

Measures the task, not a model:
  - trilinear upsample of the true Wan LR latent vs the true Wan HR latent
    (the naive baseline any trained upsampler must beat)
  - the same comparison after decoding, so it is on the delivery scale
  - how much high-frequency energy the HR latent has that the LR one cannot
If trilinear already lands near the pixel ceiling, a small trainable upsampler
is plausible; if the gap is enormous, this subtask is itself a research problem.
"""
import json, os, random, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
sys.path.insert(0, str(ROOT / "h2w/vendor/UltraFlash/inference/sr"))
TINY = "/temp/rongxiang_workspace_backup_20260921/rongxiang/UltraFlash/inference/checkpoints/tiny_decoder.pth"
WAN_PTH = "/temp/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
PAIRS = ROOT / "data/wanpairs"; PIX = ROOT / "data/pixels"


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def up3(z):
    b, c, t, h, w = z.shape
    y = F.interpolate(z.permute(0, 2, 1, 3, 4).flatten(0, 1), scale_factor=3,
                      mode="bilinear", align_corners=False)
    return y.unflatten(0, (b, t)).permute(0, 2, 1, 3, 4).contiguous()


def main():
    from tiny_decoder import WanTinyDecoder
    from vendor.wan_vae import WanVAE
    dev = "npu:0"
    tiny = WanTinyDecoder(TINY, dtype=torch.float32).to(dev).eval()
    wan = WanVAE(vae_pth=WAN_PTH, dtype=torch.bfloat16, device=dev)
    wan.model.to(dtype=torch.bfloat16).eval().requires_grad_(False)

    ids = sorted(f[:-4] for f in os.listdir(f"{PAIRS}/wan")
                 if f.endswith(".npy") and not f.endswith(".tmp.npy"))
    random.Random(1234).shuffle(ids)
    rows = []
    for vid in ids[:4]:
        w_hr = torch.from_numpy(np.load(f"{PAIRS}/wan/{vid}.npy")).float()[:, :5].unsqueeze(0).to(dev)
        lrpx = np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")
        hrpx = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")
        lr = torch.from_numpy(np.asarray(lrpx[:22])).permute(3, 0, 1, 2).unsqueeze(0).to(dev).float() / 127.5 - 1
        wan.model.clear_cache()
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            w_lr = wan.model.encode(lr[:, :, :17].to(torch.bfloat16), wan.scale).float()
        wan.model.clear_cache()

        tri = up3(w_lr)
        n = 17
        gt = torch.from_numpy(np.asarray(hrpx[:n])).permute(3, 0, 1, 2).unsqueeze(0).to(dev).float() / 127.5 - 1
        with torch.no_grad():
            px_hr = tiny.decode_latents(w_hr).float().clamp(-1, 1)
            px_tri = tiny.decode_latents(tri).float().clamp(-1, 1)
        bic = F.interpolate(
            torch.from_numpy(np.asarray(lrpx[:n])).permute(0, 3, 1, 2).to(dev).float() / 127.5 - 1,
            scale_factor=3, mode="bicubic", align_corners=False).clamp(-1, 1).permute(1, 0, 2, 3).unsqueeze(0)

        def hf(x):
            g = x.mean(1, keepdim=True)
            return float(((g - F.avg_pool3d(g, (1, 3, 3), 1, (0, 1, 1))) ** 2).mean()
                         / ((g ** 2).mean() + 1e-9))

        rows.append(dict(clip=vid,
            latent_trilinear_vs_true=round(psnr(tri, w_hr), 2),
            px_true_hr_latent=round(psnr(px_hr[:, :, :n], gt), 2),
            px_trilinear=round(psnr(px_tri[:, :, :n], gt), 2),
            px_bicubic=round(psnr(bic, gt), 2),
            hf_true_hr_latent=round(hf(w_hr), 5),
            hf_lr_latent=round(hf(w_lr), 5),
            hf_trilinear=round(hf(tri), 5)))
        print(json.dumps(rows[-1]), flush=True)
        torch.npu.empty_cache()

    a = lambda k: round(sum(r[k] for r in rows) / len(rows), 4)
    print("\nMEAN " + json.dumps({k: a(k) for k in rows[0] if k != "clip"}), flush=True)
    os._exit(0)


main()
