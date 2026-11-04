"""Latency of a pixel-domain 3x SR net at the real output size.

Answers whether moving the refiner out of latent space keeps the streaming
budget. Budget: 24 fps => 41.7 ms per output frame; one latent frame covers 4
output frames, so the per-latent budget is 166.7 ms minus the 78 ms the LR
decode already costs.
"""
import json, os, time
import torch, torch.nn as nn, torch.nn.functional as F
import torch_npu  # noqa: F401

LR_H, LR_W, SCALE = 256, 448, 3
BUDGET_FRAME_MS = 1000 / 24
LR_DECODE_MS_PER_LATENT = 78.0


class PixelSR(nn.Module):
    """Residual conv body at LR resolution, one PixelShuffle to HR.

    Keeping the body at LR resolution is what makes this cheap: all heavy
    convolutions see 448x256, and only the final shuffle touches 1344x768.
    """

    def __init__(self, width=64, blocks=8, scale=SCALE):
        super().__init__()
        self.head = nn.Conv2d(3, width, 3, padding=1)
        self.body = nn.ModuleList(
            nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True),
                          nn.Conv2d(width, width, 3, padding=1))
            for _ in range(blocks))
        self.tail = nn.Conv2d(width, 3 * scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)
        self.scale = scale

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        h = self.head(x)
        for b in self.body:
            h = h + b(h)
        return base + self.shuffle(self.tail(h))


def timeit(fn, warmup=3, iters=15):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / iters * 1000


def main():
    dev = "npu:0"
    torch.npu.init()
    rows = []
    print("budget: %.1f ms per output frame (24 fps); LR decode already costs "
          "%.0f ms per latent frame\n" % (BUDGET_FRAME_MS, LR_DECODE_MS_PER_LATENT), flush=True)
    print("%-22s%8s%11s%11s%12s%14s" % ("config", "params", "ms/frame", "ms/latent",
                                        "total/latent", "realtime"), flush=True)
    for width, blocks in ((32, 4), (64, 8), (96, 12), (128, 16)):
        m = PixelSR(width, blocks).to(dev, torch.bfloat16).eval()
        n = sum(p.numel() for p in m.parameters())
        # One latent frame decodes to 4 pixel frames; batch them together.
        x = torch.randn(4, 3, LR_H, LR_W, device=dev, dtype=torch.bfloat16)
        with torch.no_grad():
            ms_batch = timeit(lambda: m(x))
        ms_frame = ms_batch / 4
        total = LR_DECODE_MS_PER_LATENT + ms_batch
        rows.append(dict(width=width, blocks=blocks, params_m=round(n / 1e6, 2),
                         ms_per_frame=round(ms_frame, 2), ms_per_latent=round(ms_batch, 1),
                         total_ms_per_latent=round(total, 1),
                         realtime_x=round(166.7 / total, 2)))
        print("%-22s%8.2fM%11.2f%11.1f%12.1f%13.2fx"
              % (f"w{width} x{blocks} blocks", n / 1e6, ms_frame, ms_batch, total, 166.7 / total),
              flush=True)
        del m
        torch.npu.empty_cache()
    json.dump(rows, open("results/pixel_sr_latency.json", "w"), indent=2)
    print("\nwrote results/pixel_sr_latency.json", flush=True)
    os._exit(0)


main()
