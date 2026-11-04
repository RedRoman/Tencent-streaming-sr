"""Same question at a tractable size: does dropping spatial locality matter?

Uses the real block-mask pattern but a smaller token count so every variant
fits the fused-attention kernel. The mask structure (temporal causal x spatial
local, 13% density) is preserved, only the tokens-per-block shrink.
"""
import json, os, sys
from pathlib import Path
import torch, torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
dev = "npu:0"; HEADS, HD = 12, 128
NBS, NT = 66, 3
NQB = NKB = NBS * NT
BS = 8                      # tokens per block, shrunk from 101
N = NQB * BS


def attend(q, k, v, allow_blocks):
    """allow_blocks: [heads, nqb, nkb] bool -> token-level attention."""
    tok = allow_blocks[None].reshape(1, HEADS, NQB, 1, NKB, 1).expand(
        1, HEADS, NQB, BS, NKB, BS).reshape(1, HEADS, NQB * BS, NKB * BS)
    return torch_npu.npu_fusion_attention(
        q, k, v, HEADS, "BSND", atten_mask=(~tok).contiguous(),
        scale=HD ** -0.5, keep_prob=1.)[0].float()


def main():
    torch.npu.init(); torch.manual_seed(0)
    q = torch.randn(1, N, HEADS, HD, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, N, HEADS, HD, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, N, HEADS, HD, device=dev, dtype=torch.bfloat16)
    blk = torch.load("results/block_mask.pt")["mask"].to(dev)
    exact = blk[None].expand(HEADS, NQB, NKB).contiguous()

    t_allow = torch.zeros(NQB, NKB, dtype=torch.bool, device=dev)
    qi = torch.arange(NQB, device=dev)[:, None] // NBS
    ki = torch.arange(NKB, device=dev)[None] // NBS
    t_allow = ki <= qi
    temporal = t_allow[None].expand(HEADS, NQB, NKB).contiguous()
    nomask = torch.ones(HEADS, NQB, NKB, dtype=torch.bool, device=dev)

    with torch.no_grad():
        ref = attend(q, k, v, exact)
        tem = attend(q, k, v, temporal)
        non = attend(q, k, v, nomask)

    def rel(a):
        return round(float((a - ref).norm() / ref.norm()), 4)
    def cos(a):
        return round(float(F.cosine_similarity(a.flatten(), ref.flatten(), dim=0)), 5)

    print(json.dumps(dict(
        tokens=N, tokens_per_block=BS,
        exact_density=round(float(blk.float().mean()), 3),
        temporal_density=round(float(t_allow.float().mean()), 3),
        temporal_rel_err=rel(tem), temporal_cos=cos(tem),
        nomask_rel_err=rel(non), nomask_cos=cos(non))), flush=True)
    os._exit(0)


main()
