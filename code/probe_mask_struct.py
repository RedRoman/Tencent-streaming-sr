"""What does the block mask actually encode, and is it separable?

exact_block_attention receives a [b, heads, n_qblocks, n_kblocks] bool mask and
expands it to token level. If that mask factorizes into (temporal causal) x
(spatial local), each part can be expressed compactly instead of materialized.
"""
import json, math, os, sys
from pathlib import Path
import torch
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))

captured = {}


def main():
    from h3wan.npu_attention import install
    mod = install()
    from vendor.ultraflash_npu import attention as attn

    orig = attn._block_sparse_attn_fallback

    def spy(q_w, k_w, v_w, attention_mask, num_heads):
        if "mask" not in captured:
            captured["mask"] = attention_mask.detach().clone()
            captured["shapes"] = dict(q=list(q_w.shape), k=list(k_w.shape),
                                      mask=list(attention_mask.shape), heads=num_heads)
        return orig(q_w, k_w, v_w, attention_mask, num_heads)

    attn._block_sparse_attn_fallback = spy

    # minimal forward to trigger one attention call
    import math as _m
    from h3wan.sr_distill import projected_context
    from h3wan.npu_attention import sinusoidal_embedding_1d
    W = "/temp/rongxiang_workspace_backup_20260921/rongxiang/UltraFlash/inference/checkpoints/2K-Causal-Sparse-SR-Dit-fidelity-step_12800.pth"
    dev = "npu:0"
    with torch.device("meta"):
        m = mod.Transformer3DModel(
            model_type="t2v", patch_size=(1, 2, 2), text_len=512, in_dim=32, dim=1536,
            ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16, num_heads=12,
            num_layers=30, window_size=(-1, -1), qk_norm=True, cross_attn_norm=True,
            eps=1e-6, use_sparse_attn=True, sparse_causal=True, sparse_block_size=(2, 8, 8),
            sparse_top_k_ratio=1., sparse_kv_ratio=3., sparse_local_range=9,
            sparse_use_kernel=False, stream_chunk_size=2, sparse_ref_spatial_tokens=1560,
            rope_max_seq_len=1024, rope_theta=10000., rope_cache_multiple=1024)
    st = torch.load(W, map_location="cpu", mmap=True, weights_only=False)["model"]
    m.load_state_dict(st, strict=True, assign=True); del st
    m = m.to(dev).to(torch.bfloat16)
    m.time_embedding.float(); m.time_projection.float(); m.head.float(); m.eval()

    T, H, Wd = 5, 96, 168
    noise = torch.randn(1, 16, T, H, Wd, device=dev, dtype=torch.bfloat16)
    cond = torch.randn(1, 16, T, H, Wd, device=dev, dtype=torch.bfloat16)
    ctx = torch.randn(1, 512, 4096, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        x = m.patch_embedding(torch.cat((noise, cond), 1))
        grid = torch.tensor([x.shape[2:]], dtype=torch.long)
        m._ensure_rope_freqs(grid, temporal_offset=0, device=dev)
        x = x.flatten(2).transpose(1, 2)
        seq = torch.tensor([x.shape[1]])
        with torch.autocast("npu", enabled=False):
            e = m.time_embedding(sinusoidal_embedding_1d(m.freq_dim, torch.full((1,), 1000., device=dev)))
            e0 = m.time_projection(e).unflatten(1, (6, m.dim))
        cx = projected_context(m, ctx); m.clear_cross_kv()
        _, gh, gw = grid[0].tolist()
        nb = _m.ceil(gh / 8) * _m.ceil(gw / 8)
        topk = min(max(int(nb * nb * 1560 / (gh * gw)), 1), max(nb * nb * _m.ceil(T / 7) - 1, 1))
        m.blocks[0](x, e=e0, seq_lens=seq, grid_sizes=grid, freqs=m.freqs, context=cx,
                    context_lens=None, top_k=topk, local_num=None, local_range=9,
                    temporal_offset=0, train_img=False)

    msk = captured["mask"]
    print(json.dumps(captured["shapes"]), flush=True)
    b, nh, nq, nk = msk.shape
    print(json.dumps(dict(mask_shape=[b, nh, nq, nk],
                          density=round(float(msk.float().mean()), 4),
                          heads_identical=bool((msk[0] == msk[0, :1]).all()))), flush=True)

    # Does it factorize? blocks are laid out (temporal, spatial) row-major.
    nbs = nb                      # spatial blocks
    nt_q, nt_k = nq // nbs, nk // nbs
    print(json.dumps(dict(spatial_blocks=nbs, temporal_q=nt_q, temporal_k=nt_k)), flush=True)
    m0 = msk[0, 0].reshape(nt_q, nbs, nt_k, nbs)
    temporal = m0.any(dim=3).any(dim=1)           # [nt_q, nt_k]
    spatial = m0.any(dim=2).any(dim=0)            # [nbs, nbs]  (approx)
    recon = temporal[:, None, :, None] & spatial[None, :, None, :]
    print("temporal block pattern (rows=query chunk, cols=key chunk):", flush=True)
    for r in temporal.int().cpu().tolist():
        print("   ", r, flush=True)
    print(json.dumps(dict(separable=bool((recon == m0).all()),
                          spatial_density=round(float(spatial.float().mean()), 4))), flush=True)
    torch.save({"mask": msk[0, 0].cpu()}, "results/block_mask.pt")
    os._exit(0)


main()
