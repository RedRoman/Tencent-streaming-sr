"""Build (H3 LR latent, Wan HR latent) supervision for the mapper.

Temporal contract, taken from the h3_to_wan reproduction because it is the one
setting where H3 and Wan latent counts line up exactly:

  22 consecutive pixel frames
    H3  : clip_length=17 (-> 5) + tail 5 (-> 2)  = 7 latents
    Wan : encode 17 (-> 5) and 5 (-> 2) separately = 7 latents

Both VAEs are run on OUR pixels (no reuse of the stored H3 latents) so the two
sides are guaranteed to describe the same frames. 124 frames give 5 segments.

Stored per clip:
  h3  [24, 35, 16, 28]  normalized H3 latent  (LR 256x448)
  wan [16, 35, 96, 168] Wan latent            (HR 768x1344)
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, torch
import torch_npu  # noqa: F401
from diffusers import AutoencoderKLMiniMaxH3

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
PIX = ROOT / "data/pixels"
OUT = ROOT / "data/wanpairs"
H3_VAE = str(ROOT / "ckpts/h3-base/vae")
WAN_PTH = "/temp/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
SEG, NSEG = 22, 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()
    from vendor.wan_vae import WanVAE

    (OUT / "h3").mkdir(parents=True, exist_ok=True)
    (OUT / "wan").mkdir(parents=True, exist_ok=True)
    dev = "npu:0"
    h3 = AutoencoderKLMiniMaxH3.from_pretrained(H3_VAE, torch_dtype=torch.float32).to(dev).eval()
    hm = torch.tensor(h3.config.latents_mean, device=dev).view(1, 24, 1, 1, 1)
    hs = torch.tensor(h3.config.latents_std, device=dev).view(1, 24, 1, 1, 1)
    wan = WanVAE(vae_pth=WAN_PTH, dtype=torch.bfloat16, device=dev)
    wan.model.to(dtype=torch.bfloat16).eval().requires_grad_(False)

    ids = sorted(f[:-4] for f in os.listdir(f"{PIX}/hr") if f.endswith(".npy"))
    ids = ids[args.shard :: args.shards]
    print(f"shard {args.shard}/{args.shards}: {len(ids)} clips", flush=True)

    def wan_encode(x):
        wan.model.clear_cache()
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            z = wan.model.encode((x * 2 - 1).to(torch.bfloat16), wan.scale).float()
        wan.model.clear_cache()
        return z

    t0, done = time.time(), 0
    for vid in ids:
        dh, dw = OUT / "h3" / f"{vid}.npy", OUT / "wan" / f"{vid}.npy"
        if dh.exists() and dw.exists():
            done += 1; continue
        try:
            lrpx = np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")
            hrpx = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")
            hs_all, ws_all = [], []
            for s in range(NSEG):
                a, b = s * SEG, (s + 1) * SEG
                lr = torch.from_numpy(np.asarray(lrpx[a:b])).permute(3, 0, 1, 2) \
                        .unsqueeze(0).float().to(dev) / 255
                hr = torch.from_numpy(np.asarray(hrpx[a:b])).permute(3, 0, 1, 2) \
                        .unsqueeze(0).float().to(dev) / 255
                with torch.no_grad():
                    z = h3.encode(lr * 2 - 1).latent_dist.mode()
                hs_all.append(((z - hm) / hs).squeeze(0).cpu())
                ws_all.append(torch.cat([wan_encode(hr[:, :, :17]),
                                         wan_encode(hr[:, :, 17:22])], dim=2).squeeze(0).cpu())
            H = torch.cat(hs_all, dim=1)
            W = torch.cat(ws_all, dim=1)
            assert H.shape[1] == W.shape[1], (H.shape, W.shape)
            assert W.shape[-2:] == (H.shape[-2] * 6, H.shape[-1] * 6), (H.shape, W.shape)
            np.save(str(dh) + ".tmp.npy", H.to(torch.float16).numpy()); os.replace(str(dh) + ".tmp.npy", dh)
            np.save(str(dw) + ".tmp.npy", W.to(torch.float16).numpy()); os.replace(str(dw) + ".tmp.npy", dw)
        except Exception as e:
            print(f"SKIP {vid}: {str(e)[:140]}", file=sys.stderr, flush=True); continue
        done += 1
        if done % 5 == 0 or done == 1:
            el = time.time() - t0
            print(f"[{args.shard}] {done}/{len(ids)} {vid} h3{tuple(H.shape)} wan{tuple(W.shape)} "
                  f"{el/done:.1f}s/clip eta {(len(ids)-done)*el/done/60:.0f}min", flush=True)
    print(f"shard {args.shard} done {done} in {(time.time()-t0)/60:.1f} min", flush=True)
    os._exit(0)


main()
