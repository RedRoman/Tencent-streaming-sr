"""Marginal cost of one HR keyframe on the ACTUAL deployed attention.

My earlier 226 ms came from the base H3 dense-attention block. Deployment runs
the VDN hybrid (window softmax with chunk=5/radius=1, plus a linear long-range
branch), whose arithmetic is different in both directions: windowing makes the
softmax part cheaper than dense, while the linear branch adds per-token work
dense never had. So that number cannot be used and this measures the real one.

Method: build ONE VDN hybrid attention over the real config, feed it the token
counts a streaming step carries with and without an HR keyframe appended, and
scale by num_layers. Same caveat as before -- a lower bound on the increment,
since it omits per-step overheads that do not scale with tokens.
"""
import importlib.util, json, os, sys, time
import torch
import torch.nn as nn
import torch_npu  # noqa: F401

CK = "ckpts/h3-base/transformer"
VDN = "ckpts/stage-dmd-step-250/diffusers/modeling_vdn_h3.py"
dev = "npu:0"


def load_vdn():
    spec = importlib.util.spec_from_file_location("vdn_h3", VDN)
    m = importlib.util.module_from_spec(spec)
    sys.modules["vdn_h3"] = m
    spec.loader.exec_module(m)
    return m


def tm(fn, w=2, it=6):
    for _ in range(w):
        fn()
    torch.npu.synchronize(); t = time.time()
    for _ in range(it):
        fn()
    torch.npu.synchronize()
    return (time.time() - t) / it * 1000


def main():
    cfg = json.load(open(f"{CK}/config.json"))
    tcfg = json.load(open("ckpts/stage-dmd-step-250/diffusers/config.json"))["vdn"]["transform"]
    L, H, AD, D = (cfg["num_layers"], cfg["num_attention_heads"],
                   cfg["attention_head_dim"], cfg["hidden_size"])
    v = load_vdn()
    from diffusers.models.transformers import transformer_minimax_h3 as base
    orig = base.MiniMaxH3Attention(D, H, AD, cfg["qk_norm_eps"]).to(dev, torch.bfloat16).eval()
    la = tcfg["linear_attention"]
    hyb = v.HybridAttention(
        orig, D, delta_rule=la["delta_rule"],
        radius=tcfg["softmax_attention"]["radius"],
        chunk=tcfg["softmax_attention"]["chunk"],
        enable_softmax_gate=tcfg["enable_softmax_gate"],
        linear_head_dim=la["linear_head_dim"], softmax_impl="decomposed",
        anchor_frames=tcfg["anchor_frames"], enable_text_state=False,
        bridge=la["bridge"], a_fp32=la["a_fp32"],
        short_conv=tuple(la["short_conv"]["targets"]),
    ).to(dev, torch.bfloat16).eval()
    print(f"hybrid built; layers {L}", flush=True)

    rd = cfg["rope_freq_dim"] * 2
    rows = []
    # A streaming step carries a chunk of LR latent frames (S=112 tokens each).
    # An HR keyframe appends one frame of 1008 tokens.
    for tag, F, S in (("LR 5 frames", 5, 112), ("LR 5 + HR kf", 6, 112)):
        n = F * S if tag.startswith("LR 5 f") else 5 * 112 + 1008
        try:
            lay = v.SequenceLayout(seq_len=n, video_start=0, num_frames=F,
                                   tokens_per_frame=n // F)
            hyb.layout = lay
            x = torch.randn(1, n, D, device=dev, dtype=torch.bfloat16)
            rot = (torch.randn(n, rd, device=dev, dtype=torch.float32),
                   torch.randn(n, rd, device=dev, dtype=torch.float32))
            with torch.no_grad():
                ms = tm(lambda: hyb(x, rot, None))
            rows.append(dict(tag=tag, tokens=n, attn_ms=round(ms, 3),
                             all_layers_ms=round(ms * L, 1)))
            print(f"{tag:14s} {n:5d} tok  attn {ms:7.3f} ms  x{L} = {ms*L:8.1f} ms",
                  flush=True)
        except Exception as e:
            print(f"{tag:14s} FAILED {str(e)[:150]}", flush=True)
    if len(rows) == 2:
        inc = rows[1]["all_layers_ms"] - rows[0]["all_layers_ms"]
        print(f"\nkeyframe increment (attention only): {inc:.0f} ms", flush=True)
        for iv in (8, 15, 30):
            print(f"  over {iv} frames: {inc/iv:.2f} ms/frame", flush=True)
        json.dump(dict(rows=rows, increment_ms=round(inc, 1)),
                  open("results/vdn_kf_cost.json", "w"), indent=2)
    os._exit(0)


main()
