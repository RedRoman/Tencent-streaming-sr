"""Ask the upsampler what input it wants, instead of guessing.

Freeze the upsampler and optimize its INPUT directly to match the true Wan HR
latent. If some input drives it to a good output, the module works and my
conditioning was wrong. If even a freely optimized input cannot get there, the
module cannot produce this target and the route is dead regardless of the mapper.

Also reports what that optimal input looks like relative to the true Wan LR
latent, which tells us what the mapper should have been trained to produce.
"""
import importlib.util, json, os, random, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
sys.path.insert(0, str(ROOT / "h2w/vendor/UltraFlash/inference/sr"))
MAP = "/temp/liusonghua/h3_to_wan_repro_32193/joint_mapper_upsampler_wan_dit_step_002000.pt"
UF = "/temp/rongxiang_workspace_backup_20260921/rongxiang/UltraFlash/inference/checkpoints"
TINY = f"{UF}/tiny_decoder.pth"
PAIRS = ROOT / "data/wanpairs"; PIX = ROOT / "data/pixels"


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def load_up(dev, which):
    spec = importlib.util.spec_from_file_location("uf",
        ROOT / "h2w/vendor/UltraFlash/inference/sr/latent_upsampler/ultra_latent_up_v1.py")
    um = importlib.util.module_from_spec(spec); spec.loader.exec_module(um)
    up = um.UltraLatentUpsampler(in_channels=16, out_channels=16, mid_channels=128,
        num_blocks=8, upsample_scale=3, activation="relu", residual=False,
        memory_init="replicate", zero_init_final=False, default_parallel=True)
    if which == "stage4":
        sd = torch.load(MAP, map_location="cpu", mmap=True, weights_only=False)["model"]
        up.load_state_dict({k[16:]: v for k, v in sd.items()
                            if k.startswith("joint.upsampler.")}, strict=True)
    else:
        o = torch.load(f"{UF}/2K-ultraLatentUpsampler-step_28100.pth",
                       map_location="cpu", weights_only=False)
        up.load_state_dict(o["model"] if "model" in o else o, strict=True)
    return up.to(dev).eval().requires_grad_(False)


def main():
    from tiny_decoder import WanTinyDecoder
    dev = "npu:0"
    tiny = WanTinyDecoder(TINY, dtype=torch.float32).to(dev).eval()
    ids = sorted(f[:-4] for f in os.listdir(f"{PAIRS}/wan")
                 if f.endswith(".npy") and not f.endswith(".tmp.npy"))
    random.Random(1234).shuffle(ids)
    vid = ids[0]
    w_hr = torch.from_numpy(np.load(f"{PAIRS}/wan/{vid}.npy")).float()[:, :5].unsqueeze(0).to(dev)
    hrpx = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")
    gt = torch.from_numpy(np.asarray(hrpx[:17])).permute(3, 0, 1, 2).unsqueeze(0).to(dev).float() / 127.5 - 1

    for which in ("stage4", "official2K"):
        up = load_up(dev, which)
        # free input: same grid the mapper would produce
        x = torch.zeros(1, 16, 5, 32, 56, device=dev, requires_grad=True)
        opt = torch.optim.Adam([x], lr=0.05)
        for i in range(400):
            opt.zero_grad()
            with torch.autocast("npu", dtype=torch.bfloat16):
                y = up(x, parallel=True).float()
            loss = F.mse_loss(y, w_hr)
            loss.backward(); opt.step()
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            y = up(x, parallel=True).float()
            px = tiny.decode_latents(y).float().clamp(-1, 1)
        print(json.dumps(dict(upsampler=which,
            best_latent_psnr=round(psnr(y, w_hr), 2),
            best_pixel_psnr=round(psnr(px[:, :, :17], gt), 2),
            optimized_input_std=round(float(x.std()), 3),
            final_mse=round(float(loss), 5))), flush=True)
        del up; torch.npu.empty_cache()
    os._exit(0)


main()
