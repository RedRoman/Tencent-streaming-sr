"""Train a TAEHV decoder for OUR H3 VAE latents.

madebyollin's taeh3 has the right architecture (24 latent channels, patch_size 2,
16x spatial, t4 temporal) and runs 68x faster than the real VAE (6.1 ms vs 418),
but its weights only reach 8.4 dB against our decodes -- it was trained for a
different H3 VAE variant. The architecture is reusable; the weights are not.

Target: reproduce the real H3 VAE decode. 5 latent frames -> 17 pixel frames,
matching clip_length=17. Supervision is the VAE's own output, so the student can
at best match it -- which is exactly what we want for a drop-in replacement.

Pretrained weights are used as initialization where shapes allow; a sane starting
point beats random init even when the source convention differs.
"""
from __future__ import annotations

import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "assets/taehv"))
LOW = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/low/video"
PIX = ROOT / "data/pixels"
PRETRAINED = ROOT / "assets/taehv/taeh3.pth"
VAL_CLIPS = 24
LAT = 5          # latent frames per window
FRAMES = 17      # pixel frames the VAE emits for those


class Clips:
    """H3 latents paired with the LR pixels the real VAE decodes them into."""

    def __init__(self, ids):
        self.ids = list(ids); self._z = {}; self._p = {}

    def get(self, vid):
        if vid not in self._z:
            z = torch.load(f"{LOW}/{vid}.pt.part", map_location="cpu", weights_only=False)
            self._z[vid] = z.float()
            self._p[vid] = np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")
        return self._z[vid], self._p[vid]

    def sample(self, rng):
        z, px = self.get(self.ids[rng.randrange(len(self.ids))])
        nblk = min(z.shape[1] // LAT, px.shape[0] // FRAMES)
        b = rng.randrange(0, nblk)
        a = z[:, b * LAT:(b + 1) * LAT]
        t = np.asarray(px[b * FRAMES:(b + 1) * FRAMES])
        return a, t


def to_batch(ds, rng, bs, device):
    zs, ps = zip(*(ds.sample(rng) for _ in range(bs)))
    z = torch.stack(zs).to(device)                       # [B,24,5,16,28]
    p = torch.from_numpy(np.stack(ps)).to(device)        # [B,17,H,W,3] uint8
    p = p.permute(0, 1, 4, 2, 3).float() / 255.0         # NTCHW in [0,1]
    return z, p


def psnr01(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


@torch.no_grad()
def evaluate(model, ds, device, n=16):
    model.eval()
    if not hasattr(ds, "_probe"):
        pr = random.Random(20261010)
        ds._probe = [to_batch(ds, pr, 1, device) for _ in range(n)]
    ours, nearest = [], []
    for z, p in ds._probe:
        zt = z.permute(0, 2, 1, 3, 4).contiguous()
        out = model.decode_video(zt, parallel=True, show_progress_bar=False).float().clamp(0, 1)
        k = min(out.shape[1], p.shape[1])
        ours.append(psnr01(out[:, :k], p[:, :k]))
        # reference point: upsampling the latent's own mean color per frame
        flat = z.mean(1, keepdim=True).permute(0, 2, 1, 3, 4)
        up = F.interpolate(flat.flatten(0, 1), size=p.shape[-2:], mode="nearest")
        up = up.unflatten(0, (z.shape[0], z.shape[2])).expand(-1, -1, 3, -1, -1)
        kk = min(up.shape[1], p.shape[1])
        nearest.append(psnr01(up[:, :kk].clamp(0, 1), p[:, :kk]))
    model.train()
    return sum(ours) / len(ours), sum(nearest) / len(nearest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lpips", type=float, default=0.0)
    ap.add_argument("--init", choices=["pretrained", "scratch"], default="pretrained")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()

    from taehv import TAEHV
    torch.manual_seed(args.seed)
    dev = "npu:0"; args.out.mkdir(parents=True, exist_ok=True)

    model = TAEHV(checkpoint_path=None, arch_name="taeh3")
    if args.init == "pretrained":
        sd = torch.load(str(PRETRAINED), map_location="cpu", weights_only=False)
        own = model.state_dict()
        ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(ok, strict=False)
        print(f"init from pretrained: {len(ok)}/{len(own)} tensors matched", flush=True)
    # The encoder is unused for decoding; train only the decoder.
    model = model.to(dev)
    for q in model.encoder.parameters():
        q.requires_grad_(False)
    params = [q for q in model.decoder.parameters() if q.requires_grad]
    print(f"decoder params {sum(q.numel() for q in params)/1e6:.2f}M", flush=True)

    perceptual = None
    if args.lpips > 0:
        import lpips as lp
        perceptual = lp.LPIPS(net="vgg").to(dev).eval()
        for q in perceptual.parameters():
            q.requires_grad_(False)
        print(f"LPIPS on, weight {args.lpips}", flush=True)

    ids = sorted(f[:-8] for f in os.listdir(LOW) if f.endswith(".pt.part"))
    ids = [i for i in ids if (PIX / "lr" / f"{i}.npy").exists()]
    random.Random(1234).shuffle(ids)
    val, train = Clips(ids[:VAL_CLIPS]), Clips(ids[VAL_CLIPS:])
    print(f"train {len(train.ids)} val {len(val.ids)} clips", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
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
        z, p = to_batch(train, rng, args.batch, dev)
        zt = z.permute(0, 2, 1, 3, 4).contiguous()
        with torch.autocast("npu", dtype=torch.bfloat16):
            out = model.decode_video(zt, parallel=True, show_progress_bar=False)
            of = out.float()
            k = min(of.shape[1], p.shape[1])
            loss = F.l1_loss(of[:, :k], p[:, :k])
            if perceptual is not None:
                a = (of[:, :k].clamp(0, 1) * 2 - 1).flatten(0, 1)
                b = (p[:, :k] * 2 - 1).flatten(0, 1)
                loss = loss + args.lpips * perceptual(a, b).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()

        if step % 100 == 0 or step == 1:
            rec = dict(step=step, loss=round(loss.item(), 5), grad_norm=round(float(gn), 3),
                       peak_gib=round(torch.npu.max_memory_allocated() / 2**30, 2),
                       elapsed=round(time.time() - t0, 1))
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()

        if step % args.eval_every == 0 or step == args.steps:
            pp, nn_ = evaluate(model, val, dev)
            rec = dict(step=step, val_psnr=round(pp, 3), naive_psnr=round(nn_, 3),
                       elapsed=round(time.time() - t0, 1))
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            blob = {"model": model.state_dict(), "step": step, "args": vars(args)}
            torch.save(blob, args.out / "latest.pt")
            if pp > best:
                best = pp
                torch.save({**blob, "val_psnr": pp}, args.out / "best.pt")

    (args.out / "COMPLETE.json").write_text(json.dumps(
        dict(complete=True, steps=args.steps, best_val_psnr=best,
             wall_seconds=time.time() - t0), indent=2))
    print(f"done {(time.time()-t0)/60:.1f} min, best {best:.3f} dB", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
