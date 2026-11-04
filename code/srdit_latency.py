"""Is the UltraFlash SR-DiT fast enough for the streaming budget?

This decides whether the latent route can exist at all. The mapper and the
latent upsampler together cost 2.47 ms/latent; the question is what the DiT adds.

Budget: one latent frame carries 4 pixel frames at 24 fps = 166.7 ms. The chunk
the DiT consumes is 5 latents, so a chunk must finish inside 833 ms -- and that
has to cover the Wan decode too.

Both paths are measured:
  stream=True   per-block KV cache, 5 latents at a time (what streaming needs)
  stream=False  the whole 15-latent window at once (the training-time path)
"""
import json, math, os, sys, time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
WEIGHTS = "/temp/rongxiang_workspace_backup_20260921/rongxiang/UltraFlash/inference/checkpoints/2K-Causal-Sparse-SR-Dit-fidelity-step_12800.pth"
MS_PER_LATENT = 1000 * 4 / 24


def main():
    from h3wan.npu_attention import install
    from h3wan.sr_distill import projected_context, forward_sr
    dev = "npu:0"
    mod = install()
    with torch.device("meta"):
        model = mod.Transformer3DModel(
            model_type="t2v", patch_size=(1, 2, 2), text_len=512, in_dim=32, dim=1536,
            ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16, num_heads=12,
            num_layers=30, window_size=(-1, -1), qk_norm=True, cross_attn_norm=True,
            eps=1e-6, use_sparse_attn=True, sparse_causal=True, sparse_block_size=(2, 8, 8),
            sparse_top_k_ratio=1., sparse_kv_ratio=3., sparse_local_range=9,
            sparse_use_kernel=False, stream_chunk_size=2, sparse_ref_spatial_tokens=1560,
            rope_max_seq_len=1024, rope_theta=10000., rope_cache_multiple=1024)
    state = torch.load(WEIGHTS, map_location="cpu", mmap=True, weights_only=False)["model"]
    model.load_state_dict(state, strict=True, assign=True); del state
    model = model.to(dev).to(dtype=torch.bfloat16)
    model.time_embedding.float(); model.time_projection.float(); model.head.float()
    model.eval().requires_grad_(False)
    n = sum(p.numel() for p in model.parameters())
    print(f"SR-DiT loaded: {n/1e9:.3f}B params, 30 layers, dim 1536", flush=True)
    print(f"budget: {MS_PER_LATENT:.1f} ms per latent frame\n", flush=True)

    # Our HR Wan latent grid for 1344x768
    H, W = 96, 168
    ctx = torch.randn(1, 512, 4096, device=dev, dtype=torch.bfloat16)
    rows = []

    for nlat, stream in ((5, True), (5, False), (15, False)):
        noise = torch.randn(1, 16, nlat, H, W, device=dev, dtype=torch.bfloat16)
        cond = torch.randn(1, 16, nlat, H, W, device=dev, dtype=torch.bfloat16)

        def run():
            with torch.no_grad():
                return forward_sr(model, noise, cond, ctx, stream=stream, offset=0)

        try:
            for _ in range(2):
                run()
            torch.npu.synchronize(); t0 = time.time()
            for _ in range(3):
                run()
            torch.npu.synchronize()
            ms = (time.time() - t0) / 3 * 1000
            per = ms / nlat
            rows.append(dict(latents=nlat, stream=stream, chunk_ms=round(ms, 1),
                             ms_per_latent=round(per, 1),
                             realtime_x=round(MS_PER_LATENT / per, 2),
                             peak_gib=round(torch.npu.max_memory_allocated() / 2**30, 2)))
            print(json.dumps(rows[-1]), flush=True)
        except Exception as e:
            print(json.dumps(dict(latents=nlat, stream=stream, error=str(e)[:200])), flush=True)
        torch.npu.empty_cache()

    json.dump(rows, open("results/srdit_latency.json", "w"), indent=2)
    print("\nbudget check: a 5-latent chunk must finish in 833 ms INCLUDING the "
          "Wan decode (~590 ms), so the DiT has ~240 ms", flush=True)
    os._exit(0)


main()
