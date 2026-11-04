"""How large can the refiner be, under a correct latency model?

My earlier budget assumed decode and refine run back to back: 833 - 424 = 409 ms
for 17 frames = 24 ms/frame. But the producer, the decoder and the refiner can be
pipelined -- in steady state the chunk period is max(stage times), not the sum.
Under that model the refiner budget is the full 833 ms per 17 frames = 49 ms/frame,
and the decode overlaps with it.

Measure real per-frame cost across a wide capacity range, and report both models.
"""
import json, os, time
import torch
import torch_npu  # noqa: F401
from pixel_model import PixelSR

SERIAL_MS = (833 - 424) / 17     # old assumption
PIPE_MS = 833 / 17               # decode overlapped with refine


def tm(fn, warm=3, it=12):
    for _ in range(warm):
        fn()
    torch.npu.synchronize(); t0 = time.time()
    for _ in range(it):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / it * 1000


def main():
    dev = "npu:0"; torch.npu.init()
    print(f"budget serial {SERIAL_MS:.1f} ms/frame | pipelined {PIPE_MS:.1f} ms/frame\n", flush=True)
    print("%-20s%9s%11s%9s%9s" % ("config", "params", "ms/frame", "serial", "pipelined"), flush=True)
    rows = []
    for w, b in ((160, 20), (192, 24), (224, 24), (256, 24), (256, 32), (320, 32)):
        try:
            m = PixelSR(w, b, 3, 1).to(dev).eval()
            x = torch.randn(1, 3, 256, 448, device=dev)
            with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
                ms = tm(lambda: m(x, frames=1))
            n = sum(p.numel() for p in m.parameters())
            rows.append(dict(width=w, blocks=b, params_m=round(n / 1e6, 2),
                             ms_per_frame=round(ms, 2),
                             fits_serial=ms <= SERIAL_MS, fits_pipelined=ms <= PIPE_MS))
            print("%-20s%8.2fM%11.2f%9s%9s" % (
                f"w{w} x{b}", n / 1e6, ms,
                "yes" if ms <= SERIAL_MS else "NO",
                "yes" if ms <= PIPE_MS else "NO"), flush=True)
            del m; torch.npu.empty_cache()
        except Exception as e:
            print("%-20s  error %s" % (f"w{w} x{b}", str(e)[:70]), flush=True)
            torch.npu.empty_cache()
    json.dump(rows, open("results/capacity.json", "w"), indent=2)
    os._exit(0)


main()
