"""Split the 141 ms block: mask GENERATION vs attention EXECUTION.

The fallback was assumed to be the cost, but swapping it changed nothing. The
other candidate is generate_draft_block_mask, which scores block similarity to
pick top-k -- work that happens before any attention runs.
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))

T = {}


def wrap(mod, name):
    orig = getattr(mod, name)
    def f(*a, **k):
        torch.npu.synchronize(); t0 = time.time()
        r = orig(*a, **k)
        torch.npu.synchronize()
        T[name] = T.get(name, 0.0) + (time.time() - t0) * 1000
        T[name + "_n"] = T.get(name + "_n", 0) + 1
        return r
    setattr(mod, name, f)


def main():
    from h3wan.npu_attention import install
    import math
    mod = install()
    from vendor.ultraflash_npu import attention as attn
    print(json.dumps(dict(bound=attn._block_sparse_attn_fallback.__name__)), flush=True)
    for nm in ("generate_draft_block_mask", "_block_sparse_attn_fallback",
               "_window_partition_3d", "_window_reverse_3d"):
        if hasattr(attn, nm):
            wrap(attn, nm)
    print(json.dumps(dict(after_wrap=attn._block_sparse_attn_fallback.__name__)), flush=True)

    from h3wan.sr_distill import projected_context, forward_sr
    W = "/temp/rongxiang_workspace_backup_20260921/rongxiang/UltraFlash/inference/checkpoints/2K-Causal-Sparse-SR-Dit-fidelity-step_12800.pth"
    dev = "npu:0"
    with torch.device("meta"):
        m = mod.Transformer3DModel(model_type="t2v", patch_size=(1, 2, 2), text_len=512,
            in_dim=32, dim=1536, ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16,
            num_heads=12, num_layers=30, window_size=(-1, -1), qk_norm=True,
            cross_attn_norm=True, eps=1e-6, use_sparse_attn=True, sparse_causal=True,
            sparse_block_size=(2, 8, 8), sparse_top_k_ratio=1., sparse_kv_ratio=3.,
            sparse_local_range=9, sparse_use_kernel=False, stream_chunk_size=2,
            sparse_ref_spatial_tokens=1560, rope_max_seq_len=1024, rope_theta=10000.,
            rope_cache_multiple=1024)
    st = torch.load(W, map_location="cpu", mmap=True, weights_only=False)["model"]
    m.load_state_dict(st, strict=True, assign=True); del st
    m = m.to(dev).to(torch.bfloat16)
    m.time_embedding.float(); m.time_projection.float(); m.head.float(); m.eval()

    noise = torch.randn(1, 16, 5, 96, 168, device=dev, dtype=torch.bfloat16)
    cond = torch.randn(1, 16, 5, 96, 168, device=dev, dtype=torch.bfloat16)
    ctx = torch.randn(1, 512, 4096, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        forward_sr(m, noise, cond, ctx, stream=True, offset=0)   # warm
        T.clear()
        torch.npu.synchronize(); t0 = time.time()
        forward_sr(m, noise, cond, ctx, stream=True, offset=0)
        torch.npu.synchronize(); total = (time.time() - t0) * 1000

    rows = {k: round(v, 1) for k, v in T.items() if not k.endswith("_n")}
    counts = {k[:-2]: v for k, v in T.items() if k.endswith("_n")}
    print(json.dumps(dict(total_forward_ms=round(total, 1), per_fn_ms=rows,
                          calls=counts,
                          accounted=round(sum(rows.values()), 1))), flush=True)
    os._exit(0)


main()
