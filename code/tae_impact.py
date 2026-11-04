"""Does swapping the H3 VAE for the trained TAE hurt the delivered video?

The TAE only has to be good enough that the downstream refiner cannot tell. So
measure the thing that matters: final 1344x768 quality with the real VAE decode
versus the TAE decode, same refiner, same clips.
"""
import json, os, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "assets/taehv"))
from pixel_model import PixelSR
from diffusers import AutoencoderKLMiniMaxH3

LOW = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/low/video"
PIX = ROOT / "data/pixels"
VAE_DIR = str(ROOT / "ckpts/h3-base/vae")
LAT, FRAMES = 5, 17


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def main():
    from taehv import TAEHV
    dev = "npu:0"
    tae_ck = Path(sys.argv[1])
    sr_ck = Path(sys.argv[2])

    blob = torch.load(tae_ck, map_location="cpu", weights_only=False)
    tae = TAEHV(checkpoint_path=None, arch_name="taeh3")
    tae.load_state_dict(blob["model"]); tae = tae.to(dev).eval()
    print(f"TAE step {blob['step']} val {blob.get('val_psnr')}", flush=True)

    sb = torch.load(sr_ck, map_location="cpu", weights_only=False)
    a = sb["args"]
    sr = PixelSR(a["width"], a["blocks"], 3, sb["taps"]).to(dev).eval()
    sr.load_state_dict(sb["model"])

    vae = AutoencoderKLMiniMaxH3.from_pretrained(VAE_DIR, torch_dtype=torch.float32).to(dev).eval()
    mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)

    ids = sorted(f[:-4] for f in os.listdir(f"{PIX}/hr") if f.endswith(".npy"))
    random.Random(1234).shuffle(ids)
    rows = []
    for vid in ids[:4]:
        z = torch.load(f"{LOW}/{vid}.pt.part", map_location="cpu", weights_only=False)
        zc = z[:, :LAT].unsqueeze(0).to(dev, torch.float32)
        hrpx = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")
        lrpx = np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")

        with torch.no_grad():
            torch.npu.synchronize(); t0 = time.time()
            lr_vae = vae.decode(zc * std + mean).sample.float().clamp(-1, 1)
            torch.npu.synchronize(); ms_vae = (time.time() - t0) * 1000

            zt = zc.permute(0, 2, 1, 3, 4).contiguous()
            torch.npu.synchronize(); t0 = time.time()
            o = tae.decode_video(zt, parallel=True, show_progress_bar=False)
            torch.npu.synchronize(); ms_tae = (time.time() - t0) * 1000
            lr_tae = (o.permute(0, 2, 1, 3, 4).float().clamp(0, 1) * 2 - 1)

            n = min(lr_vae.shape[2], lr_tae.shape[2], FRAMES)
            outs = {}
            for tag, lrv in (("vae", lr_vae[:, :, :n]), ("tae", lr_tae[:, :, :n])):
                fr = lrv[0].permute(1, 0, 2, 3)
                with torch.autocast("npu", dtype=torch.bfloat16):
                    outs[tag] = torch.cat([sr(fr[i:i + 1], frames=1).float()
                                           for i in range(fr.shape[0])])

        gt = torch.from_numpy(np.asarray(hrpx[:n])).permute(0, 3, 1, 2).to(dev).float() / 127.5 - 1
        lr_gt = torch.from_numpy(np.asarray(lrpx[:n])).permute(0, 3, 1, 2).to(dev).float() / 127.5 - 1
        bic = F.interpolate(lr_gt, scale_factor=3, mode="bicubic", align_corners=False).clamp(-1, 1)
        rows.append(dict(clip=vid, frames=n,
            lr_vae_vs_gt=round(psnr(lr_vae[0, :, :n].permute(1, 0, 2, 3), lr_gt), 2),
            lr_tae_vs_gt=round(psnr(lr_tae[0, :, :n].permute(1, 0, 2, 3), lr_gt), 2),
            final_via_vae=round(psnr(outs["vae"], gt), 2),
            final_via_tae=round(psnr(outs["tae"], gt), 2),
            bicubic=round(psnr(bic, gt), 2),
            decode_ms_vae=round(ms_vae, 1), decode_ms_tae=round(ms_tae, 1)))
        print(json.dumps(rows[-1]), flush=True)
        torch.npu.empty_cache()

    a_ = lambda k: round(sum(r[k] for r in rows) / len(rows), 2)
    m = {k: a_(k) for k in rows[0] if k not in ("clip", "frames")}
    m["final_loss_from_tae"] = round(m["final_via_vae"] - m["final_via_tae"], 2)
    print("\nMEAN " + json.dumps(m), flush=True)
    os._exit(0)


main()
