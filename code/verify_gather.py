"""Is the gather backend numerically identical to the materialized-mask path?

Same inputs, same mask, both backends. The gather is only a win if it changes
nothing: any real difference means the model is no longer the one that was
trained. bf16 reassociation gives a small nonzero floor, so compare against
the spread between two runs of the SAME backend.
"""
import json, os, sys, time
from pathlib import Path
import torch, torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
dev = "npu:0"


def main():
    from h3wan.npu_attention import exact_block_attention, gather_block_attention
    torch.manual_seed(0)
    HEADS, HD = 12, 128
    NQB = NKB = 198; BS = 101
    C = HEADS * HD
    qw = torch.randn(NQB, BS, C, device=dev, dtype=torch.bfloat16)
    kw = torch.randn(NKB, BS, C, device=dev, dtype=torch.bfloat16)
    vw = torch.randn(NKB, BS, C, device=dev, dtype=torch.bfloat16)
    blk = torch.load("results/block_mask.pt")["mask"].to(dev)
    mask = blk[None, None].expand(1, HEADS, NQB, NKB).contiguous()

    with torch.no_grad():
        a1 = exact_block_attention(qw, kw, vw, mask, HEADS).float()
        a2 = exact_block_attention(qw, kw, vw, mask, HEADS).float()
        g1 = gather_block_attention(qw, kw, vw, mask, HEADS).float()

    def rel(x, y):
        return float((x - y).norm() / y.norm())
    def cos(x, y):
        return float(F.cosine_similarity(x.flatten(), y.flatten(), dim=0))

    print(json.dumps(dict(
        same_backend_rel=round(rel(a2, a1), 8),
        gather_vs_exact_rel=round(rel(g1, a1), 8),
        gather_vs_exact_cos=round(cos(g1, a1), 8),
        max_abs_diff=round(float((g1 - a1).abs().max()), 6),
        ref_scale=round(float(a1.abs().mean()), 6))), flush=True)

    def tm(fn, warm=1, it=3):
        for _ in range(warm):
            fn()
        torch.npu.synchronize(); t0 = time.time()
        for _ in range(it):
            fn()
        torch.npu.synchronize()
        return (time.time() - t0) / it * 1000

    with torch.no_grad():
        ms_e = tm(lambda: exact_block_attention(qw, kw, vw, mask, HEADS))
        ms_g = tm(lambda: gather_block_attention(qw, kw, vw, mask, HEADS))
    print(json.dumps(dict(exact_ms=round(ms_e, 2), gather_ms=round(ms_g, 2),
                          speedup=round(ms_e / ms_g, 2),
                          exact_x30=round(ms_e * 30, 1), gather_x30=round(ms_g * 30, 1))),
          flush=True)
    os._exit(0)


main()
