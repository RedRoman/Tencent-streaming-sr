"""Feasibility probe for the H3 -> Wan -> UltraFlash route.

Loads the already-trained mapper + UltraFlash latent upsampler from the
h3_to_wan reproduction, runs our own stored H3 LR latents through them, decodes
the result with the official Wan VAE, and scores it against the HR decode.

The point is to find out whether the existing weights transfer to our data at
all, before committing to any training.
"""
import importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]
ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))

CKPT = "/temp/liusonghua/h3_to_wan_repro_32193/joint_mapper_upsampler_wan_dit_step_002000.pt"
UF = "/temp/rongxiang_workspace_backup_20260921/rongxiang/UltraFlash"
LOW = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/low/video"
PIX = ROOT / "data/pixels"


def load_upsampler_class():
    src = ROOT / "h2w/vendor/UltraFlash/inference/sr/latent_upsampler/ultra_latent_up_v1.py"
    spec = importlib.util.spec_from_file_location("uf_up_v1", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.UltraLatentUpsampler


def main():
    dev = "npu:0"
    from h3wan.model import Mapper

    blob = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)
    sd = blob["model"]
    print("checkpoint step", blob["step"], flush=True)

    mapper = Mapper()
    mp = {k[len("joint.mapper."):]: v for k, v in sd.items() if k.startswith("joint.mapper.")}
    mapper.load_state_dict(mp, strict=True)
    mapper = mapper.to(dev).eval()
    print("mapper loaded:", sum(p.numel() for p in mapper.parameters()) / 1e6, "M", flush=True)

    Up = load_upsampler_class()
    up = Up(in_channels=16, out_channels=16, mid_channels=128, num_blocks=8,
            upsample_scale=3, activation="relu", residual=False,
            memory_init="replicate", zero_init_final=False, default_parallel=True)
    upd = {k[len("joint.upsampler."):]: v for k, v in sd.items() if k.startswith("joint.upsampler.")}
    up.load_state_dict(upd, strict=True)
    up = up.to(dev).eval()
    print("upsampler loaded:", sum(p.numel() for p in up.parameters()) / 1e6, "M", flush=True)

    # Our LR latent, normalized H3 space, exactly what the mapper was trained on.
    ids = sorted(f[:-4] for f in os.listdir(f"{PIX}/hr") if f.endswith(".npy"))
    import random; random.Random(1234).shuffle(ids)
    vid = ids[0]
    z = torch.load(f"{LOW}/{vid}.pt.part", map_location="cpu", weights_only=False)
    print(f"\nclip {vid}  H3 LR latent {tuple(z.shape)}", flush=True)

    # Mapper takes 5 or 7 latent frames at a time.
    z5 = z[:, :5].unsqueeze(0).to(dev, torch.float32)
    t0 = time.time()
    with torch.no_grad():
        wan_low = mapper(z5)
        torch.npu.synchronize(); t_map = time.time() - t0
        t0 = time.time()
        wan_high = up(wan_low, parallel=True)
        torch.npu.synchronize(); t_up = time.time() - t0
    print(f"  mapper   {tuple(z5.shape)} -> {tuple(wan_low.shape)}   {t_map*1000:.0f} ms", flush=True)
    print(f"  upsample {tuple(wan_low.shape)} -> {tuple(wan_high.shape)}   {t_up*1000:.0f} ms", flush=True)
    exp = (1, 16, 5, z.shape[2] * 6, z.shape[3] * 6)
    print(f"  expected {exp}  match={tuple(wan_high.shape)==exp}", flush=True)
    print(f"  stats: wan_low mean {wan_low.mean():+.3f} std {wan_low.std():.3f} | "
          f"wan_high mean {wan_high.mean():+.3f} std {wan_high.std():.3f}", flush=True)
    os._exit(0)


main()
