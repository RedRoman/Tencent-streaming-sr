import torch, torch_npu, json
from diffusers import AutoencoderKLMiniMaxH3
vae = AutoencoderKLMiniMaxH3.from_pretrained("ckpts/h3-base/vae", torch_dtype=torch.float32).eval()
d = vae.decoder
print("decoder class:", type(d).__name__)
for n, m in d.named_children():
    p = sum(q.numel() for q in m.parameters())
    extra = f"  len={len(m)}" if hasattr(m, "__len__") else ""
    print(f"  {n:20s} {type(m).__name__:30s} {p/1e6:8.1f}M{extra}")
# dive into the big one
big = max(d.named_children(), key=lambda kv: sum(q.numel() for q in kv[1].parameters()))
print(f"\n== inside {big[0]}")
for n, m in list(big[1].named_children())[:3]:
    p = sum(q.numel() for q in m.parameters())
    print(f"  {n:20s} {type(m).__name__:34s} {p/1e6:8.1f}M")
    for n2, m2 in m.named_children():
        print(f"      {n2:18s} {type(m2).__name__:30s} "
              f"{sum(q.numel() for q in m2.parameters())/1e6:8.2f}M")
print("\nconfig:", {k: v for k, v in vae.config.items()
                    if isinstance(v, (int, float, str, bool)) and not k.startswith("_")})
import os; os._exit(0)
