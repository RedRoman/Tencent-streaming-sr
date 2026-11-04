"""Was my decoding wrong all along?

The repro decodes a 7-latent window as TWO independent calls -- main z[:,:,:5]
-> 17 frames and tail z[:,:,5:] -> 5 frames -- each with fresh decoder memory.
I had been handing the decoder 5 or 7 latents in a single call. The tiny decoder
is causal with MemBlock state, so the call pattern changes the output.

Compare both patterns on the TRUE Wan latent, where the answer must come out
near-perfect if the decode is right.
"""
import json, os, random, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
sys.path.insert(0, str(ROOT / "h2w/vendor/UltraFlash/inference/sr"))
TINY = "/temp/rongxiang_workspace_backup_20260921/rongxiang/UltraFlash/inference/checkpoints/tiny_decoder.pth"
WAN_PTH = "/temp/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
PIX = ROOT / "data/pixels"


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def main():
    from tiny_decoder import WanTinyDecoder
    from vendor.wan_vae import WanVAE
    dev = "npu:0"
    tiny = WanTinyDecoder(TINY, dtype=torch.float32).to(dev).eval()
    wan = WanVAE(vae_pth=WAN_PTH, dtype=torch.bfloat16, device=dev)
    wan.model.to(dtype=torch.bfloat16).eval().requires_grad_(False)

    def wan_enc(x):
        wan.model.clear_cache()
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            z = wan.model.encode(x.to(torch.bfloat16), wan.scale).float()
        wan.model.clear_cache()
        return z

    ids = sorted(f[:-4] for f in os.listdir(f"{PIX}/hr") if f.endswith(".npy"))
    random.Random(1234).shuffle(ids)
    for vid in ids[:3]:
        hrpx = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")
        gt22 = torch.from_numpy(np.asarray(hrpx[:22])).permute(3, 0, 1, 2).unsqueeze(0).to(dev).float() / 127.5 - 1
        # encode exactly as the repro does: main 17 -> 5, tail 5 -> 2
        z = torch.cat([wan_enc(gt22[:, :, :17]), wan_enc(gt22[:, :, 17:22])], dim=2)

        with torch.no_grad():
            # (a) my old way: one call over all 7
            a = tiny.decode_latents(z).float().clamp(-1, 1)
            # (b) repro way: two calls, fresh memory each
            parts = []
            for part, expect in ((z[:, :, :5], 17), (z[:, :, 5:], 5)):
                r = tiny.decode_latents(part).float().clamp(-1, 1)
                parts.append((r, expect))
            b = torch.cat([p for p, _ in parts], dim=2)
            # (c) only the main 5 latents -> 17 frames, the config I used most
            c = tiny.decode_latents(z[:, :, :5]).float().clamp(-1, 1)

        row = dict(clip=vid, latent=list(z.shape[1:]),
                   single_call_frames=a.shape[2], split_call_frames=b.shape[2],
                   main_only_frames=c.shape[2])
        n = min(a.shape[2], 22)
        row["psnr_single_call"] = round(psnr(a[:, :, :n], gt22[:, :, :n]), 2)
        n2 = min(b.shape[2], 22)
        row["psnr_split_call"] = round(psnr(b[:, :, :n2], gt22[:, :, :n2]), 2)
        n3 = min(c.shape[2], 17)
        row["psnr_main_only"] = round(psnr(c[:, :, :n3], gt22[:, :, :n3]), 2)
        print(json.dumps(row), flush=True)
        torch.npu.empty_cache()
    os._exit(0)


main()
