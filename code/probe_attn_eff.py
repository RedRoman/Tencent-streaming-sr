"""Is the 4227 ms the model's cost, or the sparse-attention fallback's cost?

`sparse_use_kernel=False` in the loaded config, so block-sparse attention runs
through a Python per-head mask path instead of a fused kernel. Time the pieces
separately: the fused NPU attention and the FFN at the real shapes, and see what
30 layers would cost if attention used the fused kernel.
"""
import json, os, time
import torch, torch.nn as nn
import torch_npu  # noqa: F401

D, FFN, HEADS, L = 1536, 8960, 12, 30
N, HD = 20160, 128
dev = "npu:0"


def timeit(fn, warm=3, it=10):
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

    def fused_dense():
        return torch_npu.npu_fusion_attention(q, k, v, HEADS, "BSND",
                                              scale=HD ** -0.5, keep_prob=1.)[0]
    ms_attn = timeit(fused_dense)
    tf_attn = 2 * 2 * N * N * D / (ms_attn / 1000) / 1e12
    print(json.dumps(dict(stage="fused_dense_attention_1layer",
                          ms=round(ms_attn, 2), tflops=round(tf_attn, 1))), flush=True)

    x = torch.randn(1, N, D, device=dev, dtype=torch.bfloat16)
    ff = nn.Sequential(nn.Linear(D, FFN), nn.GELU(), nn.Linear(FFN, D)).to(dev, torch.bfloat16)
    with torch.no_grad():
        ms_ffn = timeit(lambda: ff(x))
    tf_ffn = 2 * 2 * D * FFN * N / (ms_ffn / 1000) / 1e12
    print(json.dumps(dict(stage="ffn_1layer", ms=round(ms_ffn, 2),
                          tflops=round(tf_ffn, 1))), flush=True)

    proj = nn.Linear(D, D * 4).to(dev, torch.bfloat16)
    with torch.no_grad():
        ms_proj = timeit(lambda: proj(x))
    print(json.dumps(dict(stage="qkv_out_proj_1layer", ms=round(ms_proj, 2))), flush=True)

    per_layer = ms_attn + ms_ffn + ms_proj
    print(json.dumps(dict(stage="projected_30_layers_fused_dense",
                          ms_per_layer=round(per_layer, 2),
                          ms_total=round(per_layer * L, 1),
                          vs_measured_4227=round(4227 / (per_layer * L), 1),
                          budget_ms=231)), flush=True)
    os._exit(0)


main()
