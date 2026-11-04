"""Before distilling anything, is the teacher worth distilling?

Runs the full latent route with the real SR-DiT on held-out clips and scores the
decoded pixels against the HR reference, alongside bicubic and the pixel-domain
model already delivered. If the teacher does not clearly beat those, there is no
point spending days compressing it.
"""
import importlib.util, json, math, os, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
sys.path.insert(0, str(ROOT / "h2w/vendor/UltraFlash/inference/sr"))
SRW = "/temp/rongxiang_workspace_backup_20260921/rongxiang/UltraFlash/inference/checkpoints/2K-Causal-Sparse-SR-Dit-fidelity-step_12800.pth"
TINY = "/temp/rongxiang_workspace_backup_20260921/rongxiang/UltraFlash/inference/checkpoints/tiny_decoder.pth"
MAP = "/temp/liusonghua/h3_to_wan_repro_32193/joint_mapper_upsampler_wan_dit_step_002000.pt"
PAIRS = ROOT / "data/wanpairs"; PIX = ROOT / "data/pixels"


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def main():
    from h3wan.npu_attention import install
    from h3wan.sr_distill import forward_sr
    from h3wan.model import Mapper
    from tiny_decoder import WanTinyDecoder
    mod = install(); dev = "npu:0"

    sd = torch.load(MAP, map_location="cpu", mmap=True, weights_only=False)["model"]
    mapper = Mapper(); mapper.load_state_dict({k[13:]: v for k, v in sd.items()
        if k.startswith("joint.mapper.")}, strict=True); mapper = mapper.to(dev).eval()
    spec = importlib.util.spec_from_file_location("uf",
        ROOT / "h2w/vendor/UltraFlash/inference/sr/latent_upsampler/ultra_latent_up_v1.py")
    um = importlib.util.module_from_spec(spec); spec.loader.exec_module(um)
    up = um.UltraLatentUpsampler(in_channels=16, out_channels=16, mid_channels=128,
        num_blocks=8, upsample_scale=3, activation="relu", residual=False,
        memory_init="replicate", zero_init_final=False, default_parallel=True)
    up.load_state_dict({k[16:]: v for k, v in sd.items()
        if k.startswith("joint.upsampler.")}, strict=True); up = up.to(dev).eval(); del sd

    with torch.device("meta"):
        dit = mod.Transformer3DModel(model_type="t2v", patch_size=(1, 2, 2), text_len=512,
            in_dim=32, dim=1536, ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16,
            num_heads=12, num_layers=30, window_size=(-1, -1), qk_norm=True,
            cross_attn_norm=True, eps=1e-6, use_sparse_attn=True, sparse_causal=True,
            sparse_block_size=(2, 8, 8), sparse_top_k_ratio=1., sparse_kv_ratio=3.,
            sparse_local_range=9, sparse_use_kernel=False, stream_chunk_size=2,
            sparse_ref_spatial_tokens=1560, rope_max_seq_len=1024, rope_theta=10000.,
            rope_cache_multiple=1024)
    st = torch.load(SRW, map_location="cpu", mmap=True, weights_only=False)["model"]
    dit.load_state_dict(st, strict=True, assign=True); del st
    dit = dit.to(dev).to(torch.bfloat16)
    dit.time_embedding.float(); dit.time_projection.float(); dit.head.float()
    dit.eval().requires_grad_(False)
    tiny = WanTinyDecoder(TINY, dtype=torch.float32).to(dev).eval()
    print("loaded", flush=True)

    ids = sorted(f[:-4] for f in os.listdir(f"{PAIRS}/wan")
                 if f.endswith(".npy") and not f.endswith(".tmp.npy"))
    random.Random(1234).shuffle(ids)
    ctx = torch.zeros(1, 512, 4096, device=dev, dtype=torch.bfloat16)
    rows = []
    for vid in ids[:4]:
        h3 = torch.from_numpy(np.load(f"{PAIRS}/h3/{vid}.npy")).float()[:, :5].unsqueeze(0).to(dev)
        wtrue = torch.from_numpy(np.load(f"{PAIRS}/wan/{vid}.npy")).float()[:, :5].unsqueeze(0).to(dev)
        hrpx = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")
        lrpx = np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")

        with torch.no_grad():
            cond = up(mapper(h3), parallel=True).to(torch.bfloat16)
            noise = torch.randn_like(cond)
            out = forward_sr(dit, noise, cond, ctx, stream=True, offset=0)
            if isinstance(out, tuple):
                out = out[0]
            px_dit = tiny.decode_latents(out.float()).float().clamp(-1, 1)
            px_cond = tiny.decode_latents(cond.float()).float().clamp(-1, 1)
            px_true = tiny.decode_latents(wtrue).float().clamp(-1, 1)

        n = px_dit.shape[2]
        gt = torch.from_numpy(np.asarray(hrpx[:n])).permute(3, 0, 1, 2).unsqueeze(0).to(dev).float() / 127.5 - 1
        bic = F.interpolate(
            torch.from_numpy(np.asarray(lrpx[:n])).permute(0, 3, 1, 2).float().to(dev) / 127.5 - 1,
            scale_factor=3, mode="bicubic", align_corners=False).clamp(-1, 1).permute(1, 0, 2, 3).unsqueeze(0)
        rows.append(dict(clip=vid,
            teacher_dit=round(psnr(px_dit, gt), 3),
            mapper_only_no_dit=round(psnr(px_cond, gt), 3),
            true_wan_latent=round(psnr(px_true, gt), 3),
            bicubic=round(psnr(bic, gt), 3)))
        print(json.dumps(rows[-1]), flush=True)
        torch.npu.empty_cache()

    a = lambda k: round(sum(r[k] for r in rows) / len(rows), 3)
    print("\nMEAN " + json.dumps(dict(teacher_dit=a("teacher_dit"),
        mapper_only_no_dit=a("mapper_only_no_dit"), true_wan_latent=a("true_wan_latent"),
        bicubic=a("bicubic"))), flush=True)
    os._exit(0)


main()
