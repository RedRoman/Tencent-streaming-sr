"""Which mask forms does npu_fusion_attention accept cheaply?

The block mask is [12,198,198] at 13% density, per-head and not separable, so it
cannot be replaced by a plain causal flag. The question is whether attention can
be driven per query-block over only its allowed keys (a gather), which avoids
materializing any token-level mask at all.
"""
import json, os, time
import torch, torch_npu  # noqa: F401

dev = "npu:0"; HEADS, HD = 12, 128
NQB = NKB = 198; BS = 101; N = NQB * BS


def tm(fn, warm=2, it=5):
    for _ in range(warm):
        fn()
    torch.npu.synchronize(); t0 = time.time()
    for _ in range(it):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / it * 1000


def main():
    torch.npu.init()
    q = torch.randn(1, N, HEADS, HD, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, N, HEADS, HD, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, N, HEADS, HD, device=dev, dtype=torch.bfloat16)
    blk = torch.load("results/block_mask.pt")["mask"].to(dev)   # [198,198] head 0
    print(json.dumps(dict(block_mask=list(blk.shape),
                          density=round(float(blk.float().mean()), 3))), flush=True)

    # (a) baseline: no mask at all
    ms_a = tm(lambda: torch_npu.npu_fusion_attention(q, k, v, HEADS, "BSND",
                                                     scale=HD ** -.5, keep_prob=1.)[0])
    print(json.dumps(dict(mode="no_mask", ms=round(ms_a, 2), x30=round(ms_a * 30, 1))), flush=True)

    # (b) built-in causal (sparse_mode=2 is the standard causal preset)
    try:
        ms_b = tm(lambda: torch_npu.npu_fusion_attention(
            q, k, v, HEADS, "BSND", scale=HD ** -.5, keep_prob=1., sparse_mode=2,
            atten_mask=torch.triu(torch.ones(2048, 2048, device=dev, dtype=torch.bool), 1))[0])
        print(json.dumps(dict(mode="builtin_causal_sparse_mode2",
                              ms=round(ms_b, 2), x30=round(ms_b * 30, 1))), flush=True)
    except Exception as e:
        print(json.dumps(dict(mode="builtin_causal_sparse_mode2", error=str(e)[:160])), flush=True)

    # (c) per-query-block gather: run each query block against only its allowed keys
    rows = [torch.nonzero(blk[i], as_tuple=False).flatten() for i in range(NQB)]
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(tuple(r.tolist()), []).append(i)
    print(json.dumps(dict(distinct_key_sets=len(groups),
                          mean_keys=round(sum(len(r) for r in rows) / NQB, 1))), flush=True)

    idx = {}
    for keyset, qs in groups.items():
        kidx = torch.tensor(keyset, device=dev)
        kt = (kidx[:, None] * BS + torch.arange(BS, device=dev)[None]).flatten()
        qt = torch.tensor(qs, device=dev)
        qt = (qt[:, None] * BS + torch.arange(BS, device=dev)[None]).flatten()
        idx[keyset] = (qt, kt)

    def gathered():
        out = torch.empty_like(q)
        for keyset, (qt, kt) in idx.items():
            qq = q.index_select(1, qt); kk = k.index_select(1, kt); vv = v.index_select(1, kt)
            o = torch_npu.npu_fusion_attention(qq, kk, vv, HEADS, "BSND",
                                               scale=HD ** -.5, keep_prob=1.)[0]
            out.index_copy_(1, qt, o)
        return out
    try:
        ms_c = tm(gathered, warm=1, it=3)
        print(json.dumps(dict(mode="per_block_gather", ms=round(ms_c, 2),
                              x30=round(ms_c * 30, 1),
                              speedup_vs_130=round(130.6 / ms_c, 1))), flush=True)
    except Exception as e:
        print(json.dumps(dict(mode="per_block_gather", error=str(e)[:200])), flush=True)
    os._exit(0)


main()
