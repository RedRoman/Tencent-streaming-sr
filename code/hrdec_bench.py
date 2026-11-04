"""Which student sizes fit the budget? Measure before training any of them."""
import json, os, time
import torch
import torch_npu  # noqa: F401
from hrdec_model import HRDecoder

TEACHER_MS = 5846.8      # measured, 5-latent HR chunk
BUDGET_MS = 708.0        # 17 frames at 24 fps
dev = "npu:0"


def tm(fn, w=2, it=5):
    for _ in range(w):
        fn()
    torch.npu.synchronize(); t = time.time()
    for _ in range(it):
        fn()
    torch.npu.synchronize()
    return (time.time() - t) / it * 1000


def main():
    print("%-30s%9s%10s%9s%8s" % ("config", "params", "ms/chunk", "vs 708", "speedup"))
    rows = []
    # The first sweep topped out at 150 ms against a 708 ms budget, so speed is
    # not the binding constraint -- quality is. Spend the headroom.
    # Four stages = 16x, which is the VAE's spatial compression. The first two
    # sweeps used five and were therefore timing a 32x model.
    cfgs = [("w384 b2221", 384, (2, 2, 2, 1)),
            ("w512 b4422", 512, (4, 4, 2, 2)),
            ("w512 b6632", 512, (6, 6, 3, 2)),
            ("w768 b4422", 768, (4, 4, 2, 2)),
            ("w768 b6642", 768, (6, 6, 4, 2)),
            ("w1024 b4422", 1024, (4, 4, 2, 2))]
    for name, w, sb in cfgs:
        try:
            m = HRDecoder(w, sb).to(dev).eval()
            z = torch.randn(1, 24, 5, 48, 84, device=dev)
            with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                ms = tm(lambda: m(z))
            n = sum(p.numel() for p in m.parameters())
            rows.append(dict(name=name, width=w, blocks=list(sb),
                             params_m=round(n / 1e6, 2), ms=round(ms, 1),
                             fits=ms <= BUDGET_MS, speedup=round(TEACHER_MS / ms, 1)))
            print("%-30s%8.2fM%10.1f%9s%7.1fx" % (
                name, n / 1e6, ms, "yes" if ms <= BUDGET_MS else "NO", TEACHER_MS / ms),
                flush=True)
            del m, z; torch.npu.empty_cache()
        except Exception as e:
            print("%-30s  error %s" % (name, str(e)[:60]), flush=True)
            torch.npu.empty_cache()
    json.dump(rows, open("results/hrdec_bench.json", "w"), indent=2)
    os._exit(0)


main()
