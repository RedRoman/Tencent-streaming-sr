"""Where does the 3 dB deficit come from: the mapper, or the frozen upsampler?

Three configurations decoded to pixels and scored against the HR reference:
  A  mapper -> frozen UltraFlash upsampler        (what we train)
  B  TRUE Wan LR latent -> frozen upsampler       (perfect mapper: upsampler ceiling)
  C  TRUE Wan HR latent                            (perfect everything: VAE ceiling)
If B is already near bicubic, the frozen upsampler is the cap and no amount of
mapper training will fix it.
"""
import importlib.util, json, os, random, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
PIX = ROOT / "data/pixels"; PAIRS = ROOT / "data/wanpairs"
WAN_PTH = "/temp/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def main():
    from h3wan.model import Mapper
    from vendor.wan_vae import WanVAE
    from mapper_train import load_upsampler, wan_decode

    dev = "npu:0"
    ck = Path(sys.argv[1])
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    mapper = Mapper().to(dev); mapper.load_state_dict(blob["mapper"]); mapper.eval()
    up = load_upsampler(dev, from_joint=True)
    if "upsampler" in blob:
        up.load_state_dict(blob["upsampler"])
    up.eval()
    wan = WanVAE(vae_pth=WAN_PTH, dtype=torch.bfloat16, device=dev)
    wan.model.to(dtype=torch.bfloat16).eval().requires_grad_(False)

    ids = sorted(f[:-4] for f in os.listdir(f"{PAIRS}/wan")
                 if f.endswith(".npy") and not f.endswith(".tmp.npy"))
    random.Random(1234).shuffle(ids)

    for vid in ids[:2]:
        h3 = torch.from_numpy(np.load(f"{PAIRS}/h3/{vid}.npy")).float()[:, :7].unsqueeze(0).to(dev)
        wtrue = torch.from_numpy(np.load(f"{PAIRS}/wan/{vid}.npy")).float()[:, :7].unsqueeze(0).to(dev)
        hrpx = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")
        lrpx = np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")

        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            low_pred = mapper(h3).float()
        # True Wan LR latent: encode the LR pixels with Wan (the mapper's ideal output)
        lr = torch.from_numpy(np.asarray(lrpx[:22])).permute(3, 0, 1, 2).unsqueeze(0).float().to(dev) / 255
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            wan.model.clear_cache()
            lm = wan.model.encode((lr[:, :, :17] * 2 - 1).to(torch.bfloat16), wan.scale).float()
            wan.model.clear_cache()
            lt = wan.model.encode((lr[:, :, 17:22] * 2 - 1).to(torch.bfloat16), wan.scale).float()
            wan.model.clear_cache()
        low_true = torch.cat([lm, lt], dim=2)

        def through_up(low):
            with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                return torch.cat([up(low[:, :, :5], parallel=True),
                                  up(low[:, :, 5:], parallel=True)], dim=2).float()

        T = 22
        gt = torch.from_numpy(np.asarray(hrpx[:T])).permute(0, 3, 1, 2).float() / 255
        bic = F.interpolate(torch.from_numpy(np.asarray(lrpx[:T])).permute(0, 3, 1, 2).float() / 255,
                            scale_factor=3, mode="bicubic", align_corners=False).clamp(0, 1)

        def score(z):
            px = wan_decode(wan, z)[0].permute(1, 0, 2, 3).cpu()[:T]
            torch.npu.empty_cache()
            return round(psnr(px, gt[:px.shape[0]]), 3)

        print(json.dumps(dict(
            clip=vid,
            A_mapper_then_up=score(through_up(low_pred)),
            B_trueWanLR_then_up=score(through_up(low_true)),
            C_trueWanHR=score(wtrue),
            bicubic=round(psnr(bic, gt), 3),
            low_pred_std=round(float(low_pred.std()), 3),
            low_true_std=round(float(low_true.std()), 3),
            low_latent_psnr=round(psnr(low_pred, low_true), 3),
        )), flush=True)
    os._exit(0)


main()
