"""How large can the refiner get and still fit the streaming budget?

Per 5-latent chunk the budget is 833 ms; H3 decoding already spends ~424 ms, so
the refiner has ~409 ms for 17 frames = 24 ms/frame before the stream slips.
Measure candidates at the real 256x448 -> 768x1344 size.
"""
import json, os, time
import torch
import torch_npu  # noqa: F401
from pixel_model import PixelSR

BUDGET_MS_PER_FRAME = (833 - 424) / 17


def main():
    dev = "npu:0"; torch.npu.init()
    print(f"refiner budget: {BUDGET_MS_PER_FRAME:.1f} ms/frame "
          f"(833 ms chunk - 424 ms H3 decode, 17 frames)\n", flush=True)
    print("%-26s%9s%11s%11s%9s" % ("config", "params", "ms/frame", "chunk_ms", "fits"), flush=True)
    rows = []
    for w, b in ((128, 16), (160, 20), (192, 24), (256, 24), (256, 32)):
        m = PixelSR(w, b, 3, 1).to(dev).eval()
        x = torch.randn(1, 3, 256, 448, device=dev)
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            for _ in range(3):
                m(x, frames=1)
            torch.npu.synchronize(); t0 = time.time()
            for _ in range(15):
                m(x, frames=1)
            torch.npu.synchronize()
        ms = (time.time() - t0) / 15 * 1000
        n = sum(p.numel() for p in m.parameters())
        fits = ms <= BUDGET_MS_PER_FRAME
        rows.append(dict(width=w, blocks=b, params_m=round(n / 1e6, 2),
                         ms_per_frame=round(ms, 2), chunk_ms=round(424 + ms * 17, 1),
                         fits=fits))
        print("%-26s%8.2fM%11.2f%11.1f%9s"
              % (f"w{w} x{b}", n / 1e6, ms, 424 + ms * 17, "yes" if fits else "NO"), flush=True)
        del m; torch.npu.empty_cache()
    json.dump(rows, open("results/bigger_refiner_latency.json", "w"), indent=2)
    os._exit(0)


main()
