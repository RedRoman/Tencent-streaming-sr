"""Marginal cost of HR keyframe tokens, measured on one transformer block.

The full 33B model needs two cards for weights alone, so it cannot be timed
standalone here -- and in deployment the keyframe would come from the H3
instance that is ALREADY running, making the standalone number the wrong one
anyway. What matters is the marginal cost: how much longer a step takes when
1008 HR tokens are appended to the 112 LR tokens it already carries.

One block, timed at several token counts, multiplied by num_layers. That is an
honest lower bound on the increment (it omits per-step overheads that do not
scale with tokens, and any KV-cache growth).
"""
import json, os, time, inspect
import torch, torch.nn as nn
import torch_npu  # noqa: F401
from diffusers import MiniMaxH3Transformer3DModel as M
from diffusers.models.transformers import transformer_minimax_h3 as mod

cfg = json.load(open("ckpts/h3-base/transformer/config.json"))
L, H, D = cfg["num_layers"], cfg["num_attention_heads"], cfg["hidden_size"]
print(f"layers {L}  heads {H}  hidden {D}  ffn {cfg['ffn_dim']}", flush=True)
blocks = [n for n in dir(mod) if "Block" in n]
print("block classes:", blocks, flush=True)
dev = "npu:0"


def tm(fn, w=3, it=10):
    for _ in range(w): fn()
    torch.npu.synchronize(); t = time.time()
    for _ in range(it): fn()
    torch.npu.synchronize(); return (time.time() - t) / it * 1000


# A single block is ~0.6B params; build one and reuse it for every token count.
cls = getattr(mod, blocks[0])
print(inspect.signature(cls.__init__), flush=True)
os._exit(0)
