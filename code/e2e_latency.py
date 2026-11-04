"""End-to-end chunk latency of the full latent route, both attention backends.

Stages timed separately so the remaining gap to the 833 ms budget is attributable:
  mapper + UltraFlash upsampler
  SR-DiT (the former 93% of the cost)
  tiny decoder (replacing the Wan VAE)
"""
import importlib.util, json, math, os, sys, time
from pathlib import Path
import numpy as np, torch
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
sys.path.insert(0, str(ROOT / "h2w/vendor/UltraFlash/inference/sr"))
SRW = "/temp/rongxiang_workspace_backup_20260921/rongxiang/UltraFlash/inference/checkpoints/2K-Causal-Sparse-SR-Dit-fidelity-step_12800.pth"
TINY = "/temp/rongxiang_workspace_backup_20260921/rongxiang/UltraFlash/inference/checkpoints/tiny_decoder.pth"
MAPPER_CK = "/temp/liusonghua/h3_to_wan_repro_32193/joint_mapper_upsampler_wan_dit_step_002000.pt"
BUDGET = 833.0


def tm(fn, warm=1, it=3):
    for _ in range(warm):
        fn()
    torch.npu.synchronize(); t0 = time.time()
    for _ in range(it):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / it * 1000


def main():
    backend = os.environ.get("SOLH3_ATTN", "gather")
    dev = "npu:0"
    from h3wan.npu_attention import install
    from h3wan.sr_distill import projected_context, forward_sr
    from h3wan.model import Mapper
    from tiny_decoder import WanTinyDecoder
    mod = install()

    sd = torch.load(MAPPER_CK, map_location="cpu", mmap=True, weights_only=False)["model"]
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

    h3 = torch.randn(1, 24, 5, 16, 28, device=dev)
    ctx = torch.randn(1, 512, 4096, device=dev, dtype=torch.bfloat16)

    with torch.no_grad():
        ms_map = tm(lambda: up(mapper(h3), parallel=True))
        wan_hi = up(mapper(h3), parallel=True).to(torch.bfloat16)
        noise = torch.randn_like(wan_hi)
        ms_dit = tm(lambda: forward_sr(dit, noise, wan_hi, ctx, stream=True, offset=0))
        out = forward_sr(dit, noise, wan_hi, ctx, stream=True, offset=0)
        # stream=True returns (sample, caches)
        if isinstance(out, tuple):
            out = out[0]
        ms_dec = tm(lambda: tiny.decode_latents(out.float()))

    from h3wan.npu_attention import GATHER_STATS
    print(json.dumps(dict(gather_stats=GATHER_STATS)), flush=True)
    total = ms_map + ms_dit + ms_dec
    print(json.dumps(dict(backend=backend,
        mapper_upsampler_ms=round(ms_map, 1), sr_dit_ms=round(ms_dit, 1),
        tiny_decoder_ms=round(ms_dec, 1), total_ms=round(total, 1),
        budget_ms=BUDGET, over_budget_x=round(total / BUDGET, 2),
        realtime_x=round(BUDGET / total, 2),
        peak_gib=round(torch.npu.max_memory_allocated() / 2**30, 2))), flush=True)
    os._exit(0)


main()
