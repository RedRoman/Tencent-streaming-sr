"""Pin down the Wan temporal protocol before encoding 800 clips.

H3 gives 37 latent frames for 124 pixel frames. Wan compresses time 4x but its
first latent frame covers 1 pixel frame (causal), so chunk boundaries have to be
checked rather than assumed.
"""
import os, sys, numpy as np, torch
from pathlib import Path
import torch_npu  # noqa: F401
HOME=os.environ["HOME"]; ROOT=Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0,str(ROOT/"h2w"))
from vendor.wan_vae import WanVAE
dev="npu:0"
wan=WanVAE(vae_pth="/temp/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",dtype=torch.bfloat16,device=dev)
wan.model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
hr=np.load(f"{ROOT}/data/pixels/hr/"+sorted(os.listdir(f"{ROOT}/data/pixels/hr"))[0],mmap_mode="r")
print("hr pixels", hr.shape)
for nf in (17, 21, 33, 61, 124):
    x=torch.from_numpy(np.asarray(hr[:nf])).permute(3,0,1,2).unsqueeze(0).float().to(dev)/255
    wan.model.clear_cache()
    with torch.no_grad(), torch.autocast("npu",dtype=torch.bfloat16):
        z=wan.model.encode((x*2-1).to(torch.bfloat16),wan.scale).float()
    wan.model.clear_cache()
    print(f"  {nf:3d} pixel frames -> wan latent {tuple(z.shape)}  std {z.std():.3f}")
os._exit(0)
