"""Confirm the mask materialization is the 130 ms, not the attention math."""
import json, os, time
import torch, torch_npu  # noqa: F401

dev = "npu:0"; N, HEADS, HD = 20160, 12, 128
NQB = NKB = 198          # block counts actually used
BS = N // NQB            # tokens per block


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
    small = torch.rand(1, HEADS, NQB, NKB, device=dev) > 0.5

    print(json.dumps(dict(tokens=N, blocks=NQB, tokens_per_block=BS,
          dense_mask_elems=f"{1*HEADS*N*N/1e9:.2f}G",
          dense_mask_gib=round(HEADS * N * N / 2**30, 2))), flush=True)

    ms_nomask = tm(lambda: torch_npu.npu_fusion_attention(
        q, k, v, HEADS, "BSND", scale=HD ** -0.5, keep_prob=1.)[0])
    print(json.dumps(dict(stage="fused_attention_no_mask", ms=round(ms_nomask, 2),
                          x30=round(ms_nomask * 30, 1))), flush=True)

    def expand_only():
        return small.reshape(1, HEADS, NQB, 1, NKB, 1).expand(
            1, HEADS, NQB, BS, NKB, BS).reshape(1, HEADS, NQB * BS, NKB * BS).contiguous()
    try:
        ms_expand = tm(expand_only, warm=1, it=3)
        print(json.dumps(dict(stage="mask_materialization_only",
                              ms=round(ms_expand, 2), x30=round(ms_expand * 30, 1))), flush=True)
    except Exception as e:
        print(json.dumps(dict(stage="mask_materialization_only", error=str(e)[:150])), flush=True)
    os._exit(0)


main()
