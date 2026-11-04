"""What target CAN this upsampler hit?

Inverting its input could only reach 13 dB against the true Wan HR latent, so
the module is not a latent 3x super-resolver in the sense I assumed. Test the
alternatives by checking which target an optimized input can actually reach:

  T1  true Wan HR latent                 (my assumption)
  T2  trilinear 3x of the true Wan LR    (a smooth upsample, no new detail)
  T3  the upsampler's own output from the true Wan LR  (self-consistency, sanity)
A module reaches its own output trivially; whichever real target it reaches
tells us what it was trained to produce.
"""
import importlib.util, json, os, random, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
sys.path.insert(0, str(ROOT / "h2w/vendor/UltraFlash/inference/sr"))
UF = "/temp/rongxiang_workspace_backup_20260921/rongxiang/UltraFlash/inference/checkpoints"
WAN_PTH = "/temp/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
PAIRS = ROOT / "data/wanpairs"; PIX = ROOT / "data/pixels"


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def main():
    from vendor.wan_vae import WanVAE
    dev = "npu:0"
    spec = importlib.util.spec_from_file_location("uf",
        ROOT / "h2w/vendor/UltraFlash/inference/sr/latent_upsampler/ultra_latent_up_v1.py")
    um = importlib.util.module_from_spec(spec); spec.loader.exec_module(um)
    up = um.UltraLatentUpsampler(in_channels=16, out_channels=16, mid_channels=128,
        num_blocks=8, upsample_scale=3, activation="relu", residual=False,
        memory_init="replicate", zero_init_final=False, default_parallel=True)
    o = torch.load(f"{UF}/2K-ultraLatentUpsampler-step_28100.pth", map_location="cpu", weights_only=False)
    up.load_state_dict(o["model"] if "model" in o else o, strict=True)
    up = up.to(dev).eval().requires_grad_(False)
    wan = WanVAE(vae_pth=WAN_PTH, dtype=torch.bfloat16, device=dev)
    wan.model.to(dtype=torch.bfloat16).eval().requires_grad_(False)

    ids = sorted(f[:-4] for f in os.listdir(f"{PAIRS}/wan")
                 if f.endswith(".npy") and not f.endswith(".tmp.npy"))
    random.Random(1234).shuffle(ids)
    vid = ids[0]
    w_hr = torch.from_numpy(np.load(f"{PAIRS}/wan/{vid}.npy")).float()[:, :5].unsqueeze(0).to(dev)
    lrpx = np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")
    lr = torch.from_numpy(np.asarray(lrpx[:17])).permute(3, 0, 1, 2).unsqueeze(0).to(dev).float() / 127.5 - 1
    wan.model.clear_cache()
    with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
        w_lr = wan.model.encode(lr.to(torch.bfloat16), wan.scale).float()
    wan.model.clear_cache()

    b, c, t, h, wd = w_lr.shape
    tri = F.interpolate(w_lr.permute(0, 2, 1, 3, 4).flatten(0, 1), scale_factor=3,
                        mode="bilinear", align_corners=False).unflatten(0, (b, t)).permute(0, 2, 1, 3, 4).contiguous()
    with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
        self_out = up(w_lr, parallel=True).float()

    print(json.dumps(dict(w_lr=list(w_lr.shape[1:]), w_hr=list(w_hr.shape[1:]),
                          tri=list(tri.shape[1:]), self_out=list(self_out.shape[1:]),
                          natural_out_vs_trueHR=round(psnr(self_out, w_hr), 2),
                          natural_out_vs_trilinear=round(psnr(self_out, tri), 2),
                          trilinear_vs_trueHR=round(psnr(tri, w_hr), 2))), flush=True)

    for tag, target in (("trueWanHR", w_hr), ("trilinear", tri), ("self_output", self_out)):
        x = w_lr.clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=0.03)
        for _ in range(300):
            opt.zero_grad()
            with torch.autocast("npu", dtype=torch.bfloat16):
                y = up(x, parallel=True).float()
            loss = F.mse_loss(y, target)
            loss.backward(); opt.step()
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            y = up(x, parallel=True).float()
        print(json.dumps(dict(target=tag, reached_psnr=round(psnr(y, target), 2),
                              input_drift_from_wanLR=round(psnr(x.detach(), w_lr), 2))), flush=True)
        torch.npu.empty_cache()
    os._exit(0)


main()
