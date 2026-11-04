"""Marginal cost of HR keyframe tokens, measured on one transformer block.

The full 33B model is 61.7 GiB in bf16 and does not fit one 61 GiB card, so it
cannot be timed standalone here. In deployment that is moot -- the keyframe
would come from the H3 instance already running across several cards -- and the
number that matters is marginal: how much longer a step takes when 1008 HR
tokens are appended to the 112 LR tokens it already carries.

One block timed at each token count, multiplied by num_layers. An honest lower
bound: it omits per-step overheads that do not scale with tokens.
"""
import inspect, json, os, time
import torch
import torch_npu  # noqa: F401
from diffusers.models.transformers import transformer_minimax_h3 as mod

cfg = json.load(open("ckpts/h3-base/transformer/config.json"))
L, H, AD, D = (cfg["num_layers"], cfg["num_attention_heads"],
               cfg["attention_head_dim"], cfg["hidden_size"])
dev = "npu:0"
sig = inspect.signature(mod.MiniMaxH3TransformerBlock.__init__)
print("ctor:", list(sig.parameters)[1:], flush=True)
kw = dict(hidden_size=D, num_attention_heads=H, attention_head_dim=AD,
          ffn_dim=cfg["ffn_dim"], norm_eps=cfg["norm_eps"],
          qk_norm_eps=cfg["qk_norm_eps"], time_embed_dim=cfg["time_embed_dim"])
kw = {k: v for k, v in kw.items() if k in sig.parameters}
blk = mod.MiniMaxH3TransformerBlock(**kw).to(dev, torch.bfloat16).eval()
pn = sum(p.numel() for p in blk.parameters())
print(f"block {pn/1e9:.3f}B  x{L} = {pn*L/1e9:.2f}B", flush=True)
print("forward:", list(inspect.signature(blk.forward).parameters), flush=True)


def tm(fn, w=3, it=8):
    for _ in range(w):
        fn()
    torch.npu.synchronize(); t = time.time()
    for _ in range(it):
        fn()
    torch.npu.synchronize()
    return (time.time() - t) / it * 1000


# Token counts: LR is 8x14 = 112 per latent frame, HR is 24x42 = 1008.
# A streaming step carries a window of latent frames, so sweep totals that
# bracket both regimes.
rows = []
temb = torch.randn(1, cfg["time_embed_dim"], device=dev, dtype=torch.bfloat16)
for tag, n in (("LR 1 frame", 112), ("LR 5 frames", 560), ("HR 1 frame", 1008),
               ("LR5+HR1", 1568), ("HR 5 frames", 5040)):
    try:
        # hidden_states is (batch, seq, hidden); adaln_indices is 1-D over the
        # sequence (it selects one AdaLN row per token); cos/sin are
        # (seq, rotary_dim) with rotary_dim = 2 * rope_freq_dim.
        x = torch.randn(1, n, D, device=dev, dtype=torch.bfloat16)
        ai = torch.zeros(n, dtype=torch.long, device=dev)
        rd = cfg["rope_freq_dim"] * 2
        rot = (torch.randn(n, rd, device=dev, dtype=torch.float32),
               torch.randn(n, rd, device=dev, dtype=torch.float32))
        with torch.no_grad():
            ms = tm(lambda: blk(x, temb, ai, rot, None))
        tot = ms * L
        rows.append(dict(tag=tag, tokens=n, block_ms=round(ms, 3),
                         all_layers_ms=round(tot, 1)))
        print(f"{tag:14s} {n:5d} tok  block {ms:7.3f} ms  x{L} = {tot:8.1f} ms", flush=True)
    except Exception as e:
        print(f"{tag:14s} {n:5d} tok  FAILED {str(e)[:90]}", flush=True)
        torch.npu.empty_cache()
json.dump(rows, open("results/h3_block_cost.json", "w"), indent=2)
os._exit(0)
