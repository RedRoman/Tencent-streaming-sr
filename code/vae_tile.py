"""Is the decoder's attention already windowed? The cost ratio says it might be.

attn and ff split 42.8/47.2 at BOTH grids, and both reported the same attention
input shape. If the decoder tiles the spatial grid into fixed windows then HR
is not a bigger attention, it is 9x more of the same one -- which would mean
attention sparsity has almost no headroom here, since the attention is already
local.
"""
import os, inspect, torch, torch_npu
from collections import Counter
from diffusers import AutoencoderKLMiniMaxH3
dev = "npu:0"
vae = AutoencoderKLMiniMaxH3.from_pretrained("ckpts/h3-base/vae",
                                             torch_dtype=torch.float32).to(dev).eval()
mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
D = "/temp/yunfeng/h3_hrx0kd_transfer_20261008"
cnt = Counter()
def hook(m, i, o): cnt[tuple(i[0].shape)] += 1
hs = [b.attn.register_forward_hook(hook) for b in vae.decoder.transformer_blocks]
for tag, p in (("LR", f"{D}/low/video"), ("HR", f"{D}/high/video")):
    z = torch.load(f"{p}/H3O20261001_0001.pt.part", map_location="cpu", weights_only=False)
    cnt.clear()
    with torch.no_grad():
        vae.decode(z[:, :5].unsqueeze(0).to(dev, torch.float32) * std + mean)
    print(f"{tag}: attention calls per decode =", dict(cnt), flush=True)
    torch.npu.empty_cache()
for h in hs: h.remove()
print("\n_split_tiles source:")
print(inspect.getsource(type(vae)._split_tiles)[:900])
os._exit(0)
