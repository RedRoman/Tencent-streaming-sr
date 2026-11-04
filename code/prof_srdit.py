"""Where do the 4227 ms go? Time one block and its parts at the real shapes."""
import json, math, os, sys, time
from pathlib import Path
import torch
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
WEIGHTS = "/temp/rongxiang_workspace_backup_20260921/rongxiang/UltraFlash/inference/checkpoints/2K-Causal-Sparse-SR-Dit-fidelity-step_12800.pth"


def tm(fn, warm=2, it=5):
    for _ in range(warm):
        fn()
    torch.npu.synchronize(); t0 = time.time()
    for _ in range(it):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / it * 1000


def main():
    from h3wan.npu_attention import install
    from h3wan.sr_distill import projected_context
    dev = "npu:0"
    mod = install()
    with torch.device("meta"):
        m = mod.Transformer3DModel(
            model_type="t2v", patch_size=(1, 2, 2), text_len=512, in_dim=32, dim=1536,
            ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16, num_heads=12,
            num_layers=30, window_size=(-1, -1), qk_norm=True, cross_attn_norm=True,
            eps=1e-6, use_sparse_attn=True, sparse_causal=True, sparse_block_size=(2, 8, 8),
            sparse_top_k_ratio=1., sparse_kv_ratio=3., sparse_local_range=9,
            sparse_use_kernel=False, stream_chunk_size=2, sparse_ref_spatial_tokens=1560,
            rope_max_seq_len=1024, rope_theta=10000., rope_cache_multiple=1024)
    st = torch.load(WEIGHTS, map_location="cpu", mmap=True, weights_only=False)["model"]
    m.load_state_dict(st, strict=True, assign=True); del st
    m = m.to(dev).to(dtype=torch.bfloat16)
    m.time_embedding.float(); m.time_projection.float(); m.head.float()
    m.eval().requires_grad_(False)

    T, H, W = 5, 96, 168
    noise = torch.randn(1, 16, T, H, W, device=dev, dtype=torch.bfloat16)
    cond = torch.randn(1, 16, T, H, W, device=dev, dtype=torch.bfloat16)
    ctx = torch.randn(1, 512, 4096, device=dev, dtype=torch.bfloat16)

    with torch.no_grad():
        x = m.patch_embedding(torch.cat((noise, cond), 1))
        grid = torch.tensor([x.shape[2:]], dtype=torch.long)
        m._ensure_rope_freqs(grid, temporal_offset=0, device=dev)
        x = x.flatten(2).transpose(1, 2)
        seq = torch.tensor([x.shape[1]])
        from h3wan.npu_attention import sinusoidal_embedding_1d
        with torch.autocast("npu", enabled=False):
            e = m.time_embedding(sinusoidal_embedding_1d(m.freq_dim, torch.full((1,), 1000., device=dev)))
            e0 = m.time_projection(e).unflatten(1, (6, m.dim))
        cx = projected_context(m, ctx)
        m.clear_cross_kv()
        _, gh, gw = grid[0].tolist()
        nb = math.ceil(gh / 8) * math.ceil(gw / 8)
        topk = min(max(int(nb * nb * 1560 / (gh * gw)), 1), max(nb * nb * math.ceil(T / 7) - 1, 1))
        kw = dict(e=e0, seq_lens=seq, grid_sizes=grid, freqs=m.freqs, context=cx,
                  context_lens=None, top_k=topk, local_num=None, local_range=9,
                  temporal_offset=0, train_img=False)
        print(json.dumps(dict(tokens=x.shape[1], blocks=nb, top_k=topk)), flush=True)

        blk = m.blocks[0]
        ms_block = tm(lambda: blk(x, **kw))
        print(json.dumps(dict(stage="one_block_total", ms=round(ms_block, 2),
                              x30=round(ms_block * 30, 1))), flush=True)

        # self-attention alone
        xn = blk.norm1(x).to(torch.bfloat16)
        for _tk in (topk, 198, 99, 40, 16):
            _ms = tm(lambda tk=_tk: blk.self_attn(xn, seq, grid, m.freqs, top_k=tk,
                                           local_num=None, local_range=9,
                                           temporal_offset=0, train_img=False))
            print(json.dumps(dict(stage="self_attn_topk", top_k=_tk, ms=round(_ms,2), x30=round(_ms*30,1))), flush=True)
        ms_attn = tm(lambda: blk.self_attn(xn, seq, grid, m.freqs, top_k=topk,
                                           local_num=None, local_range=9,
                                           temporal_offset=0, train_img=False))
        print(json.dumps(dict(stage="self_attn", ms=round(ms_attn, 2),
                              pct=round(ms_attn / ms_block * 100, 1))), flush=True)
        # ffn alone
        ms_ffn = tm(lambda: blk.ffn(xn))
        print(json.dumps(dict(stage="ffn", ms=round(ms_ffn, 2),
                              pct=round(ms_ffn / ms_block * 100, 1))), flush=True)
        # cross-attention alone
        ms_cross = tm(lambda: blk.cross_attn(blk.norm3(x).to(torch.bfloat16), cx, None))
        print(json.dumps(dict(stage="cross_attn", ms=round(ms_cross, 2),
                              pct=round(ms_cross / ms_block * 100, 1))), flush=True)
    os._exit(0)


main()
