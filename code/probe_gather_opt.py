"""Make the exact-mask path fast by batching query blocks instead of looping.

The faithful option is to run each query block against only its allowed keys.
Naively that is 195 separate kernel launches. Since every query block allows
about the same NUMBER of key blocks (26 on average), blocks can be padded to a
common key count and batched into one call, turning 195 launches into a handful.
"""
import json, os, time
import torch, torch_npu  # noqa: F401

dev = "npu:0"; HEADS, HD = 12, 128
NQB = NKB = 198; BS = 101; N = NQB * BS


def tm(fn, warm=1, it=3):
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
    blk = torch.load("results/block_mask.pt")["mask"].to(dev)

    counts = blk.sum(1)
    print(json.dumps(dict(key_blocks_per_query=dict(
        min=int(counts.min()), max=int(counts.max()),
        mean=round(float(counts.float().mean()), 1)))), flush=True)

    # Bucket query blocks by how many key blocks they need, pad to the bucket max,
    # and run one batched attention per bucket.
    order = torch.argsort(counts)
    buckets, cur, cur_max = [], [], 0
    for qi in order.tolist():
        c = int(counts[qi])
        if cur and c > cur_max * 1.5:
            buckets.append((cur, cur_max)); cur, cur_max = [], 0
        cur.append(qi); cur_max = max(cur_max, c)
    if cur:
        buckets.append((cur, cur_max))
    print(json.dumps(dict(buckets=len(buckets),
                          sizes=[len(b) for b, _ in buckets][:10])), flush=True)

    ar = torch.arange(BS, device=dev)
    prepared = []
    for qs, kmax in buckets:
        qidx = torch.tensor(qs, device=dev)
        qtok = (qidx[:, None] * BS + ar[None]).reshape(-1)
        ks = torch.zeros(len(qs), kmax, dtype=torch.long, device=dev)
        valid = torch.zeros(len(qs), kmax, dtype=torch.bool, device=dev)
        for j, qi in enumerate(qs):
            idx = torch.nonzero(blk[qi], as_tuple=False).flatten()
            ks[j, :len(idx)] = idx
            valid[j, :len(idx)] = True
        prepared.append((qtok, ks, valid, len(qs), kmax))

    def batched():
        out = torch.empty_like(q)
        for qtok, ks, valid, nq, kmax in prepared:
            ktok = (ks[:, :, None] * BS + ar[None, None]).reshape(nq, kmax * BS)
            qq = q.index_select(1, qtok).reshape(nq, BS, HEADS, HD)
            kk = k[0].index_select(0, ktok.reshape(-1)).reshape(nq, kmax * BS, HEADS, HD)
            vv = v[0].index_select(0, ktok.reshape(-1)).reshape(nq, kmax * BS, HEADS, HD)
            mask = (~valid[:, :, None].expand(nq, kmax, BS).reshape(nq, 1, 1, kmax * BS)
                    ).expand(nq, HEADS, BS, kmax * BS).contiguous()
            o = torch_npu.npu_fusion_attention(qq, kk, vv, HEADS, "BSND",
                                               atten_mask=mask, scale=HD ** -.5, keep_prob=1.)[0]
            out.index_copy_(1, qtok, o.reshape(1, nq * BS, HEADS, HD))
        return out

    try:
        ms = tm(batched)
        print(json.dumps(dict(mode="bucketed_gather", ms=round(ms, 2),
                              x30=round(ms * 30, 1),
                              speedup_vs_130=round(130.6 / ms, 1))), flush=True)
    except Exception as e:
        print(json.dumps(dict(mode="bucketed_gather", error=str(e)[:220])), flush=True)
    os._exit(0)


main()
