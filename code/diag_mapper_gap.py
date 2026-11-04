"""Where exactly do the 20 dB between mapper output and the true Wan latent go?

Decompose the chain the mapper is supposed to reproduce:
  A  true Wan HR latent                      -- the supervision target
  B  mapper(H3 LR) -> Wan LR, then upsampler -- what we produce
  C  true Wan LR latent, then upsampler      -- perfect mapper, real upsampler
  D  mapper(H3 LR) output before upsampling  -- mapper alone vs true Wan LR

If C is already far from A, the frozen upsampler caps the chain and the mapper
cannot be the fix. If C is close to A but B is not, the mapper is the problem.
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
PAIRS = ROOT / "data/wanpairs"; PIX = ROOT / "data/pixels"
MAPPER_TRAINED = ROOT / "results/mapper_2359/mapper_up/best.pt"


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def main():
    from h3wan.model import Mapper
    from tiny_decoder import WanTinyDecoder
    from vendor.wan_vae import WanVAE
    dev = "npu:0"

    sd = torch.load(MAP, map_location="cpu", mmap=True, weights_only=False)["model"]
    spec = importlib.util.spec_from_file_location("uf",
        ROOT / "h2w/vendor/UltraFlash/inference/sr/latent_upsampler/ultra_latent_up_v1.py")
    um = importlib.util.module_from_spec(spec); spec.loader.exec_module(um)
    up = um.UltraLatentUpsampler(in_channels=16, out_channels=16, mid_channels=128,
        num_blocks=8, upsample_scale=3, activation="relu", residual=False,
        memory_init="replicate", zero_init_final=False, default_parallel=True)
    up.load_state_dict({k[16:]: v for k, v in sd.items()
        if k.startswith("joint.upsampler.")}, strict=True); up = up.to(dev).eval()

    m_orig = Mapper(); m_orig.load_state_dict({k[13:]: v for k, v in sd.items()
        if k.startswith("joint.mapper.")}, strict=True); m_orig = m_orig.to(dev).eval(); del sd
    m_tr = None
    if MAPPER_TRAINED.exists():
        b = torch.load(MAPPER_TRAINED, map_location="cpu", weights_only=False)
        m_tr = Mapper(); m_tr.load_state_dict(b["mapper"]); m_tr = m_tr.to(dev).eval()
        print(f"our trained mapper: step {b['step']}", flush=True)

    tiny = WanTinyDecoder(TINY, dtype=torch.float32).to(dev).eval()
    wan = WanVAE(vae_pth=WAN_PTH, dtype=torch.bfloat16, device=dev)
    wan.model.to(dtype=torch.bfloat16).eval().requires_grad_(False)

    ids = sorted(f[:-4] for f in os.listdir(f"{PAIRS}/wan")
                 if f.endswith(".npy") and not f.endswith(".tmp.npy"))
    random.Random(1234).shuffle(ids)
    rows = []
    for vid in ids[:3]:
        w_hr = torch.from_numpy(np.load(f"{PAIRS}/wan/{vid}.npy")).float()[:, :5].unsqueeze(0).to(dev)
        h3 = torch.from_numpy(np.load(f"{PAIRS}/h3/{vid}.npy")).float()[:, :5].unsqueeze(0).to(dev)
        lrpx = np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")
        hrpx = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")

        # true Wan LR latent: encode the LR pixels the mapper is meant to convert
        lr = torch.from_numpy(np.asarray(lrpx[:22])).permute(3, 0, 1, 2).unsqueeze(0).to(dev).float() / 127.5 - 1
        wan.model.clear_cache()
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            w_lr = wan.model.encode(lr[:, :, :17].to(torch.bfloat16), wan.scale).float()
        wan.model.clear_cache()

        with torch.no_grad():
            lo_orig = m_orig(h3)
            up_orig = up(lo_orig, parallel=True)
            up_true = up(w_lr, parallel=True)
            lo_tr = m_tr(h3) if m_tr else None
            up_tr = up(lo_tr, parallel=True) if m_tr else None

        row = dict(clip=vid,
                   lr_latent_shape=list(w_lr.shape),
                   mapper_vs_trueWanLR_orig=round(psnr(lo_orig, w_lr), 2),
                   upsampled_trueWanLR_vs_hr=round(psnr(up_true, w_hr), 2),
                   upsampled_mapper_vs_hr_orig=round(psnr(up_orig, w_hr), 2))
        if m_tr:
            row["mapper_vs_trueWanLR_trained"] = round(psnr(lo_tr, w_lr), 2)
            row["upsampled_mapper_vs_hr_trained"] = round(psnr(up_tr, w_hr), 2)

        n = 17
        gt = torch.from_numpy(np.asarray(hrpx[:n])).permute(3, 0, 1, 2).unsqueeze(0).to(dev).float() / 127.5 - 1
        for tag, z in (("A_true_hr", w_hr), ("C_upsampled_trueLR", up_true),
                       ("B_upsampled_mapper", up_orig)) + \
                      ((("B2_upsampled_mapper_trained", up_tr),) if m_tr else ()):
            px = tiny.decode_latents(z.float()).float().clamp(-1, 1)
            row[f"px_{tag}"] = round(psnr(px[:, :, :n], gt), 2)
        rows.append(row); print(json.dumps(row), flush=True)
        torch.npu.empty_cache()

    a = lambda k: round(sum(r[k] for r in rows if k in r) / len(rows), 2)
    keys = [k for k in rows[0] if k not in ("clip", "lr_latent_shape")]
    print("\nMEAN " + json.dumps({k: a(k) for k in keys}), flush=True)
    os._exit(0)


main()
