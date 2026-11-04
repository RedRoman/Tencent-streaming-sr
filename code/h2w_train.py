"""Train H3 latent -> Wan HR latent directly, bypassing the broken upsampler.

Supervision is the Wan VAE's own encode of the HR frames, which is a real point
in Wan latent space rather than a proxy. Every eval decodes with the tiny
decoder and reports pixel PSNR against the HR reference and against bicubic --
the latent MSE alone has already proven unreliable on this project.
"""
from __future__ import annotations

import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
sys.path.insert(0, str(ROOT / "h2w/vendor/UltraFlash/inference/sr"))
from h2w_model import H3ToWanHR

PAIRS = ROOT / "data/wanpairs"; PIX = ROOT / "data/pixels"
TINY = "/temp/rongxiang_workspace_backup_20260921/rongxiang/UltraFlash/inference/checkpoints/tiny_decoder.pth"
VAL_CLIPS = 24
SEG = 7            # Wan main(5) + tail(2), the trained temporal contract
CROP = 8           # latent-space crop in H3 units -> 48 in Wan units


class Pairs:
    def __init__(self, ids):
        self.ids = list(ids); self._mm = {}

    def _get(self, v):
        if v not in self._mm:
            self._mm[v] = (np.load(f"{PAIRS}/h3/{v}.npy", mmap_mode="r"),
                           np.load(f"{PAIRS}/wan/{v}.npy", mmap_mode="r"))
        return self._mm[v]

    def sample(self, rng, crop=CROP):
        h, w = self._get(self.ids[rng.randrange(len(self.ids))])
        ch = min(crop, h.shape[2]) if crop else h.shape[2]
        cw = min(crop, h.shape[3]) if crop else h.shape[3]
        s = rng.randrange(0, h.shape[1] // SEG) * SEG
        y = rng.randrange(0, h.shape[2] - ch + 1)
        x = rng.randrange(0, h.shape[3] - cw + 1)
        crop = None
        a = np.asarray(h[:, s:s + SEG, y:y + ch, x:x + cw])
        b = np.asarray(w[:, s:s + SEG, y * 6:(y + ch) * 6, x * 6:(x + cw) * 6])
        return a, b


def batch(ds, rng, bs, device, crop=CROP):
    hs, ws = zip(*(ds.sample(rng, crop) for _ in range(bs)))
    return (torch.from_numpy(np.stack(hs)).to(device).float(),
            torch.from_numpy(np.stack(ws)).to(device).float())


@torch.no_grad()
def evaluate(model, ds, device, tiny, n=12):
    """Latent MSE on a fixed probe plus pixel PSNR on a few whole windows."""
    model.eval()
    if not hasattr(ds, "_probe"):
        pr = random.Random(20261010)
        ds._probe = [batch(ds, pr, 2, device) for _ in range(n)]
        ds._full = []
        for v in ds.ids[:3]:
            h, w = ds._get(v)
            ds._full.append((v,
                torch.from_numpy(np.asarray(h[:, :SEG])).unsqueeze(0).to(device).float(),
                torch.from_numpy(np.asarray(w[:, :SEG])).unsqueeze(0).to(device).float()))
    mses, base = [], []
    for a, b in ds._probe:
        with torch.autocast("npu", dtype=torch.bfloat16):
            p = model(a).float()
        mses.append(F.mse_loss(p, b).item())
        base.append(F.mse_loss(b.mean().expand_as(b), b).item())

    def dec(z):
        out = []
        for part in (z[:, :, :5], z[:, :, 5:]):
            out.append(tiny.decode_latents(part.float()).float().clamp(-1, 1))
        return torch.cat(out, dim=2)

    px, pxb, pxt = [], [], []
    for vid, a, b in ds._full:
        with torch.autocast("npu", dtype=torch.bfloat16):
            p = model(a).float()
        hr = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")
        lr = np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")
        g = torch.from_numpy(np.asarray(hr[:22])).permute(3, 0, 1, 2).unsqueeze(0).to(a.device).float() / 127.5 - 1
        bic = F.interpolate(
            torch.from_numpy(np.asarray(lr[:22])).permute(0, 3, 1, 2).to(a.device).float() / 127.5 - 1,
            scale_factor=3, mode="bicubic", align_corners=False).clamp(-1, 1).permute(1, 0, 2, 3).unsqueeze(0)
        for lst, z in ((px, p), (pxt, b)):
            d = dec(z); k = min(d.shape[2], 22)
            lst.append(float(-10 * torch.log10(F.mse_loss(d[:, :, :k], g[:, :, :k]) + 1e-12)))
        k = min(bic.shape[2], 22)
        pxb.append(float(-10 * torch.log10(F.mse_loss(bic[:, :, :k], g[:, :, :k]) + 1e-12)))
        torch.npu.empty_cache()
    model.train()
    return (sum(mses) / len(mses), sum(base) / len(base),
            sum(px) / len(px), sum(pxb) / len(pxb), sum(pxt) / len(pxt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=CROP)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pixel", type=float, default=0.0,
                    help="weight on pixel L1 through the frozen tiny decoder; latent "
                         "MSE alone has repeatedly failed to track pixel quality here")
    ap.add_argument("--lpips", type=float, default=0.0,
                    help="weight on LPIPS in the decoded pixel domain")
    ap.add_argument("--pixel-frames", type=int, default=5,
                    help="latent frames decoded for the pixel terms (memory control)")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()

    from tiny_decoder import WanTinyDecoder
    torch.manual_seed(args.seed)
    dev = "npu:0"; args.out.mkdir(parents=True, exist_ok=True)

    ids = sorted(f[:-4] for f in os.listdir(f"{PAIRS}/wan")
                 if f.endswith(".npy") and not f.endswith(".tmp.npy"))
    ids = [i for i in ids if (PAIRS / "h3" / f"{i}.npy").exists()]
    random.Random(1234).shuffle(ids)
    val, train = Pairs(ids[:VAL_CLIPS]), Pairs(ids[VAL_CLIPS:])
    print(f"train {len(train.ids)} val {len(val.ids)} clips", flush=True)

    model = H3ToWanHR(width=args.width).to(dev)
    model.grad_checkpointing = True
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    tiny = WanTinyDecoder(TINY, dtype=torch.float32).to(dev).eval()

    perceptual = None
    if args.lpips > 0:
        import lpips as lpips_pkg
        perceptual = lpips_pkg.LPIPS(net="vgg").to(dev).eval()
        for q in perceptual.parameters():
            q.requires_grad_(False)
        print(f"LPIPS on, weight {args.lpips}", flush=True)
    if args.pixel > 0:
        print(f"pixel L1 on, weight {args.pixel}, {args.pixel_frames} latent frames", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.03, final_div_factor=50)
    start = 1
    if args.resume and args.resume.exists():
        rb = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(rb["model"]); start = rb["step"] + 1
        for _ in range(rb["step"]):
            sched.step()
        print(f"resumed at {rb['step']}", flush=True)

    rng = random.Random(args.seed)
    (args.out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    log = (args.out / "metrics.jsonl").open("a")
    t0, best = time.time(), -1e9

    for step in range(start, args.steps + 1):
        a, b = batch(train, rng, args.batch, dev, args.crop)
        with torch.autocast("npu", dtype=torch.bfloat16):
            p = model(a)
            pf = p.float()
            loss = F.mse_loss(pf, b)
            if args.pixel > 0 or args.lpips > 0:
                # Decode a prefix through the frozen tiny decoder; gradients flow
                # to the latent, the decoder itself stays fixed.
                k = min(args.pixel_frames, pf.shape[2])
                dp = tiny.decode_latents(pf[:, :, :k])
                with torch.no_grad():
                    dt = tiny.decode_latents(b[:, :, :k])
                if args.pixel > 0:
                    loss = loss + args.pixel * F.l1_loss(dp.float(), dt.float())
                if args.lpips > 0:
                    fp = dp.float().clamp(-1, 1)[0].permute(1, 0, 2, 3)
                    ft = dt.float().clamp(-1, 1)[0].permute(1, 0, 2, 3)
                    loss = loss + args.lpips * perceptual(fp, ft).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 100 == 0 or step == 1:
            rec = dict(step=step, loss=round(loss.item(), 5), grad_norm=round(float(gn), 3),
                       peak_gib=round(torch.npu.max_memory_allocated() / 2**30, 2),
                       elapsed=round(time.time() - t0, 1))
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()

        if step % args.eval_every == 0 or step == args.steps:
            m, tr, pp, pb, pt = evaluate(model, val, dev, tiny)
            rec = dict(step=step, val_latent_mse=round(m, 5), trivial_mse=round(tr, 5),
                       px_ours=round(pp, 2), px_bicubic=round(pb, 2), px_ceiling=round(pt, 2),
                       gain_vs_bicubic=round(pp - pb, 2), elapsed=round(time.time() - t0, 1))
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            blob = {"model": model.state_dict(), "step": step, "args": vars(args)}
            torch.save(blob, args.out / "latest.pt")
            if pp > best:
                best = pp
                torch.save({**blob, "px_ours": pp}, args.out / "best.pt")

    (args.out / "COMPLETE.json").write_text(json.dumps(
        dict(complete=True, steps=args.steps, best_px=best,
             wall_seconds=time.time() - t0), indent=2))
    print(f"done {(time.time()-t0)/60:.1f} min, best pixel PSNR {best:.2f}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
