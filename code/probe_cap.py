"""Find a batch/crop/width setting that actually uses the card."""
import time, torch, torch.nn.functional as F
import torch_npu  # noqa: F401
from pixel_model import PixelSR
dev = "npu:0"; torch.npu.init()
print("%-34s%10s%12s%12s" % ("config", "peak GiB", "ms/step", "crops/s"), flush=True)
for width, blocks, crop, bs in ((96,12,64,8),(96,12,96,16),(128,16,96,16),(128,16,128,16),
                                (160,20,128,16),(128,16,128,32),(160,20,128,32)):
    try:
        torch.npu.reset_peak_memory_stats()
        m = PixelSR(width, blocks, 3, 1).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        lr = torch.randn(bs,3,crop,crop,device=dev)
        hr = torch.randn(bs,3,crop*3,crop*3,device=dev)
        def step():
            with torch.autocast("npu", dtype=torch.bfloat16):
                loss = F.l1_loss(m(lr,frames=1).float(), hr)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        for _ in range(3): step()
        torch.npu.synchronize(); t0=time.time()
        for _ in range(10): step()
        torch.npu.synchronize(); ms=(time.time()-t0)/10*1000
        n=sum(p.numel() for p in m.parameters())
        print("%-34s%10.2f%12.1f%12.0f" % (
            f"w{width} x{blocks} crop{crop} bs{bs} ({n/1e6:.1f}M)",
            torch.npu.max_memory_allocated()/2**30, ms, bs/ms*1000), flush=True)
        del m, opt, lr, hr; torch.npu.empty_cache()
    except Exception as e:
        print("%-34s  OOM/err %s" % (f"w{width} x{blocks} crop{crop} bs{bs}", str(e)[:60]), flush=True)
        torch.npu.empty_cache()
import os; os._exit(0)
