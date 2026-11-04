"""Redo the upsampler inversion properly before trusting the earlier verdict.

The first attempt ran 300-400 Adam steps at lr=0.03 and concluded the module
cannot reach the true Wan HR latent. But it also failed to reach a plain
trilinear target (16.6 dB), which a smooth function should hit easily -- that
smells like an unconverged optimization, not a limitation of the module.

This runs far longer, with LR decay and from several initializations, and tracks
the loss curve so convergence is visible rather than assumed.
"""
import importlib.util, json, os, random, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
sys.path.insert(0, str(ROOT / "h2w/vendor/UltraFlash/inference/sr"))
UF = "/temp/rongxiang_workspace_backup_20260921/rongxiang/UltraFlash/inference/checkpoints"
MAP = "/temp/liusonghua/h3_to_wan_repro_32193/joint_mapper_upsampler_wan_dit_step_002000.pt"
WAN_PTH = "/temp/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
PAIRS = ROOT / "data/wanpairs"; PIX = ROOT / "data/pixels"


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def main():
    from h3wan.model import Mapper
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

    ck = torch.load(MAP, map_location="cpu", mmap=True, weights_only=False)["model"]
    mapper = Mapper(); mapper.load_state_dict({k[13:]: v for k, v in ck.items()
        if k.startswith("joint.mapper.")}, strict=True); mapper = mapper.to(dev).eval(); del ck

    ids = sorted(f[:-4] for f in os.listdir(f"{PAIRS}/wan")
                 if f.endswith(".npy") and not f.endswith(".tmp.npy"))
    random.Random(1234).shuffle(ids)
    vid = ids[0]
    w_hr = torch.from_numpy(np.load(f"{PAIRS}/wan/{vid}.npy")).float()[:, :5].unsqueeze(0).to(dev)
    h3 = torch.from_numpy(np.load(f"{PAIRS}/h3/{vid}.npy")).float()[:, :5].unsqueeze(0).to(dev)
    with torch.no_grad():
        mapped = mapper(h3)

    inits = {"zeros": torch.zeros(1, 16, 5, 32, 56, device=dev),
             "mapper_out": mapped.clone(),
             "randn": torch.randn(1, 16, 5, 32, 56, device=dev) * 0.5}

    for name, x0 in inits.items():
        x = x0.clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=0.1)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4000, eta_min=1e-4)
        curve = []
        for i in range(4000):
            opt.zero_grad()
            y = up(x, parallel=True)           # fp32, no autocast, for clean gradients
            loss = F.mse_loss(y, w_hr)
            loss.backward(); opt.step(); sched.step()
            if i in (0, 99, 499, 999, 1999, 3999):
                curve.append((i + 1, round(float(loss), 5)))
        with torch.no_grad():
            y = up(x, parallel=True)
        print(json.dumps(dict(init=name, curve=curve,
                              final_latent_psnr=round(psnr(y, w_hr), 2),
                              input_std=round(float(x.std()), 3))), flush=True)
        torch.npu.empty_cache()
    os._exit(0)


main()
