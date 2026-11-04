"""Does dropping the spatial-local part of the mask change the output?

Three attention backends on identical inputs, compared against the exact
materialized-mask result the model was trained with:
  exact      current path (correct, 130 ms)
  temporal   keep only temporal causality, drop spatial locality
  none       no mask at all
If temporal-only stays close to exact, the 9.5x speedup is free; if not, the
spatial constraint is load-bearing and gather is the only safe option.
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
dev = "npu:0"; HEADS, HD = 12, 128
NQB = NKB = 198; BS = 101; N = NQB * BS
NBS, NT = 66, 3     # spatial blocks, temporal chunks


def main():
    from h3wan.npu_attention import exact_block_attention
    torch.manual_seed(0)
    q = torch.randn(N, BS and HEADS * HD, device=dev, dtype=torch.bfloat16).reshape(N, HEADS * HD)
    # exact_block_attention signature: (q_w,k_w,v_w,mask,heads) with q_w [nq*bs? ...]
    qw = torch.randn(N, 1, HEADS * HD, device=dev, dtype=torch.bfloat16)
    kw = torch.randn(N, 1, HEADS * HD, device=dev, dtype=torch.bfloat16)
    vw = torch.randn(N, 1, HEADS * HD, device=dev, dtype=torch.bfloat16)
    blk = torch.load("results/block_mask.pt")["mask"].to(dev)          # [198,198] head0
    full = blk[None, None].expand(1, HEADS, NQB, NKB).contiguous()

    with torch.no_grad():
        ref = exact_block_attention(qw, kw, vw, full, HEADS).float()

    # temporal-only mask: allow every spatial block within allowed time chunks
    t_allow = torch.zeros(NQB, NKB, dtype=torch.bool, device=dev)
    for qi in range(NQB):
        for ki in range(NKB):
            if ki // NBS <= qi // NBS:
                t_allow[qi, ki] = True
    tmask = t_allow[None, None].expand(1, HEADS, NQB, NKB).contiguous()
    with torch.no_grad():
        tem = exact_block_attention(qw, kw, vw, tmask, HEADS).float()

    nmask = torch.ones(1, HEADS, NQB, NKB, dtype=torch.bool, device=dev)
    with torch.no_grad():
        non = exact_block_attention(qw, kw, vw, nmask, HEADS).float()

    def rel(a):
        return round(float((a - ref).norm() / ref.norm()), 5)
    def cos(a):
        return round(float(torch.nn.functional.cosine_similarity(
            a.flatten(), ref.flatten(), dim=0)), 5)

    print(json.dumps(dict(
        exact_density=round(float(blk.float().mean()), 3),
        temporal_density=round(float(t_allow.float().mean()), 3),
        temporal_rel_err=rel(tem), temporal_cos=cos(tem),
        nomask_rel_err=rel(non), nomask_cos=cos(non))), flush=True)
    os._exit(0)


main()
