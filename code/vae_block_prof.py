"""Attention or FFN -- which half of the VAE decoder owns the 5847 ms?

The decoder turns out to be a 36-layer 2.4B ViT, not a CNN, so both levers on
the table apply to it: sparsity attacks attention, pruning attacks FFN. Which
one is worth doing depends entirely on where the time actually is, and that
split changes with resolution because attention is quadratic in tokens while
the FFN is linear.

Per-block attn/ff timing at both grids, plus the token counts that explain it.
"""
import json, os, time
from collections import defaultdict
import torch
import torch_npu  # noqa: F401
from diffusers import AutoencoderKLMiniMaxH3

dev = "npu:0"
HIGH = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/high/video"
LOW = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/low/video"


def main():
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        "ckpts/h3-base/vae", torch_dtype=torch.float32).to(dev).eval()
    mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
    zh = torch.load(f"{HIGH}/H3O20261001_0001.pt.part", map_location="cpu", weights_only=False)
    zl = torch.load(f"{LOW}/H3O20261001_0001.pt.part", map_location="cpu", weights_only=False)

    acc = defaultdict(float)
    seen = {}
    stack = []

    def pre(m, i):
        torch.npu.synchronize(); stack.append(time.time())

    def post(m, i, o):
        torch.npu.synchronize(); acc[m._pn] += time.time() - stack.pop()
        if m._pn == "attn" and "tok" not in seen:
            x = i[0]
            seen["tok"] = tuple(x.shape)

    hooks = []
    for blk in vae.decoder.transformer_blocks:
        for nm in ("attn", "ff"):
            mod = getattr(blk, nm)
            mod._pn = nm
            hooks.append(mod.register_forward_pre_hook(pre))
            hooks.append(mod.register_forward_hook(post))

    rows = []
    for tag, z in (("LR 16x28", zl), ("HR 48x84", zh)):
        w = z[:, :5].unsqueeze(0).to(dev, torch.float32) * std + mean
        with torch.no_grad():
            vae.decode(w)
        acc.clear(); seen.clear()
        torch.npu.synchronize(); t0 = time.time()
        with torch.no_grad():
            vae.decode(w)
        torch.npu.synchronize()
        tot = (time.time() - t0) * 1000
        a, f = acc["attn"] * 1000, acc["ff"] * 1000
        print(f"\n{tag}  total {tot:8.1f} ms   attn input {seen.get('tok')}", flush=True)
        print(f"  attn  {a:8.1f} ms  {a/tot*100:5.1f}%   ({a/36:6.2f} ms/layer)", flush=True)
        print(f"  ff    {f:8.1f} ms  {f/tot*100:5.1f}%   ({f/36:6.2f} ms/layer)", flush=True)
        print(f"  other {tot-a-f:8.1f} ms  {(tot-a-f)/tot*100:5.1f}%", flush=True)
        rows.append(dict(grid=tag, total_ms=round(tot, 1), attn_ms=round(a, 1),
                         ff_ms=round(f, 1), tokens=list(seen.get("tok", ()))))
        torch.npu.empty_cache()
    for h in hooks:
        h.remove()
    json.dump(rows, open("results/vae_block_prof.json", "w"), indent=2)
    os._exit(0)


main()
