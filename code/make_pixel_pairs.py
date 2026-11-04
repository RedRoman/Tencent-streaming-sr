"""Decode the stored LR/HR latent pairs into the pixel pairs the SR net trains on.

Both sides go through the same frozen H3 VAE with the same normalization, so a
later PSNR against the HR decode is measured in exactly the domain the model is
asked to produce. Saved as uint8 to keep 800 clips on disk at a sane size;
that costs <=0.5/255 of quantization, far below the differences being measured.

LR  : [24,37,16,28] -> [124,256,448,3]
HR  : [24,37,48,84] -> [124,768,1344,3]
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch_npu  # noqa: F401
from diffusers import AutoencoderKLMiniMaxH3

HOME = os.environ["HOME"]
VAE_DIR = f"{HOME}/modelarts/user-job-dir/sol_h3/ckpts/h3-base/vae"
LOW = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/low/video"
HIGH = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/high/video"
OUT = f"{HOME}/modelarts/user-job-dir/sol_h3/data/pixels"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="0 = all clips")
    args = ap.parse_args()

    os.makedirs(f"{OUT}/lr", exist_ok=True)
    os.makedirs(f"{OUT}/hr", exist_ok=True)
    dev = "npu:0"
    vae = AutoencoderKLMiniMaxH3.from_pretrained(VAE_DIR, torch_dtype=torch.float32).to(dev).eval()
    mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)

    ids = sorted(f[:-8] for f in os.listdir(HIGH) if f.endswith(".pt.part"))
    if args.limit:
        ids = ids[: args.limit]
    ids = ids[args.shard :: args.shards]
    print(f"shard {args.shard}/{args.shards}: {len(ids)} clips", flush=True)

    def decode(path, suffix):
        z = torch.load(path, map_location="cpu", weights_only=False)
        z = z.unsqueeze(0).to(dev, torch.float32) * std + mean
        with torch.no_grad():
            px = vae.decode(z).sample.float().clamp(-1, 1)
        arr = ((px[0].permute(1, 2, 3, 0) + 1) * 127.5).round().clamp(0, 255).byte()
        return arr.cpu().numpy()

    t0, done = time.time(), 0
    for vid in ids:
        dst_lr, dst_hr = f"{OUT}/lr/{vid}.npy", f"{OUT}/hr/{vid}.npy"
        if os.path.exists(dst_lr) and os.path.exists(dst_hr):
            done += 1
            continue
        try:
            if not os.path.exists(dst_lr):
                a = decode(f"{LOW}/{vid}.pt.part", "lr")
                np.save(dst_lr + ".tmp.npy", a); os.replace(dst_lr + ".tmp.npy", dst_lr)
            if not os.path.exists(dst_hr):
                a = decode(f"{HIGH}/{vid}.pt.part", "hr")
                np.save(dst_hr + ".tmp.npy", a); os.replace(dst_hr + ".tmp.npy", dst_hr)
        except Exception as e:
            print(f"SKIP {vid}: {str(e)[:120]}", file=sys.stderr, flush=True)
            continue
        done += 1
        if done % 5 == 0 or done == 1:
            el = time.time() - t0
            print(f"[{args.shard}] {done}/{len(ids)} {vid} {el/done:.1f}s/clip "
                  f"eta {(len(ids)-done)*el/done/60:.0f}min", flush=True)
    print(f"shard {args.shard} done: {done} clips in {(time.time()-t0)/60:.1f} min", flush=True)
    os._exit(0)


main()
