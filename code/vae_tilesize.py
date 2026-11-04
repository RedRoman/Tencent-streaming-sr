"""HR pays 14x the tile count for 9x the pixels -- is that overlap overhead
recoverable by using bigger tiles?

LR 448x256 needs 2 tiles, HR 1344x768 needs 28. Area ratio is 9x, so 1.56x of
the HR cost is redundant work on tile overlaps. Bigger tiles would cut it, if
memory allows and the knob exists.
"""
import inspect, os, time, torch, torch_npu
from collections import Counter
from diffusers import AutoencoderKLMiniMaxH3
dev = "npu:0"
vae = AutoencoderKLMiniMaxH3.from_pretrained("ckpts/h3-base/vae",
                                             torch_dtype=torch.float32).to(dev).eval()
print(inspect.signature(vae.enable_tiling), flush=True)
defaults = {k: getattr(vae, k) for k in dir(vae)
            if "tile" in k.lower() and isinstance(getattr(vae, k), (int, float, bool))}
print("tile attrs:", defaults, flush=True)
mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
z = torch.load("/temp/yunfeng/h3_hrx0kd_transfer_20261008/high/video/H3O20261001_0001.pt.part",
               map_location="cpu", weights_only=False)
w = z[:, :5].unsqueeze(0).to(dev, torch.float32) * std + mean
cnt = Counter()
hs = [b.attn.register_forward_hook(lambda m, i, o: cnt.update([1]))
      for b in vae.decoder.transformer_blocks]
for ts in (None, 768, 1024, 1536):
    try:
        if ts is not None:
            vae.enable_tiling(tile_sample_min_height=ts, tile_sample_min_width=ts)
        cnt.clear()
        with torch.no_grad():
            vae.decode(w)
        torch.npu.synchronize(); t = time.time()
        with torch.no_grad():
            vae.decode(w)
        torch.npu.synchronize()
        print(f"tile {str(ts):>6s}  {(time.time()-t)*1000:8.1f} ms  attn calls {cnt[1]//2}",
              flush=True)
        torch.npu.empty_cache()
    except Exception as e:
        print(f"tile {str(ts):>6s}  FAILED {str(e)[:90]}", flush=True)
        torch.npu.empty_cache()
os._exit(0)
