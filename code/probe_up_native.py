"""Does the upsampler work in ITS OWN latent convention?

Self-consistency test that needs no external target: take a real HR Wan latent,
downsample it 3x, and ask the upsampler to restore it. If the module is a valid
3x latent upsampler for this VAE, that round trip should be decent. If it fails
even here, its input/output convention differs from ours (different VAE,
normalization, or channel order) -- which is fixable, unlike a dead module.

Also tries the reverse normalization and a couple of plausible scalings.
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


def down3(z):
    b, c, t, h, w = z.shape
    y = F.interpolate(z.permute(0, 2, 1, 3, 4).flatten(0, 1), scale_factor=1/3,
                      mode="bilinear", align_corners=False, antialias=True)
    return y.unflatten(0, (b, t)).permute(0, 2, 1, 3, 4).contiguous()


def main():
    from vendor.wan_vae import WanVAE
    dev = "npu:0"
    spec = importlib.util.spec_from_file_location("uf",
        ROOT / "h2w/vendor/UltraFlash/inference/sr/latent_upsampler/ultra_latent_up_v1.py")
    um = importlib.util.module_from_spec(spec); spec.loader.exec_module(um)

    results = []
    for ckname in ("2K-ultraLatentUpsampler-step_28100.pth",
                   "1K-ultraLatentUpsampler-step_19900.pth",
                   "latent_upsampler.pth"):
        try:
            o = torch.load(f"{UF}/{ckname}", map_location="cpu", weights_only=False)
            sd = o["model"] if isinstance(o, dict) and "model" in o else o
            up = um.UltraLatentUpsampler(in_channels=16, out_channels=16, mid_channels=128,
                num_blocks=8, upsample_scale=3, activation="relu", residual=False,
                memory_init="replicate", zero_init_final=False, default_parallel=True)
            up.load_state_dict(sd, strict=True)
            up = up.to(dev).eval().requires_grad_(False)
        except Exception as e:
            print(json.dumps(dict(ckpt=ckname, load_error=str(e)[:120])), flush=True)
            continue

        ids = sorted(f[:-4] for f in os.listdir(f"{PAIRS}/wan")
                     if f.endswith(".npy") and not f.endswith(".tmp.npy"))
        random.Random(1234).shuffle(ids)
        row = dict(ckpt=ckname)
        accum = {}
        for vid in ids[:3]:
            w_hr = torch.from_numpy(np.load(f"{PAIRS}/wan/{vid}.npy")).float()[:, :5].unsqueeze(0).to(dev)
            lo = down3(w_hr)
            variants = {
                "as_is": lo,
                "times_std": lo * float(w_hr.std()) / max(float(lo.std()), 1e-6),
                "unit_norm": (lo - lo.mean()) / lo.std(),
            }
            for tag, z in variants.items():
                with torch.no_grad():
                    y = up(z, parallel=True)
                accum.setdefault(tag, []).append(psnr(y, w_hr))
            tri = F.interpolate(lo.permute(0, 2, 1, 3, 4).flatten(0, 1), scale_factor=3,
                                mode="bilinear", align_corners=False)
            tri = tri.unflatten(0, (lo.shape[0], lo.shape[2])).permute(0, 2, 1, 3, 4)
            accum.setdefault("trilinear_ref", []).append(psnr(tri, w_hr))
            torch.npu.empty_cache()
        for k, v in accum.items():
            row[k] = round(sum(v) / len(v), 2)
        results.append(row); print(json.dumps(row), flush=True)
        del up; torch.npu.empty_cache()
    os._exit(0)


main()
