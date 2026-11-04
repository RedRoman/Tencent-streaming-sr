"""Distil the 2.42B H3 HR VAE decoder into a small convolutional student.

Supervision is the teacher's own output: data/pixels/hr/*.npy was produced by
running this exact decoder, so this is plain distillation with a perfect,
already-cached target rather than a reconstruction task.

Shape contract is measured, not assumed (hrdec_align.py):
    latents[5k:5k+5] -> frames[17k:17k+17], at 39-42 dB against the cached
full-clip decode. That 40 dB is the CEILING for any chunk-based student scored
this way -- the teacher itself does not reproduce its own whole-clip decode
exactly when run in chunks, and the student is being asked to match the latter.

Crops are latent-aligned: one HR latent cell is exactly 16 HR pixels, so a
`--crop` cell window maps to crop*16 pixels with no resampling.
"""
from __future__ import annotations

import argparse, json, os, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from hrdec_model import HRDecoder

HOME = os.environ["HOME"]
ROOT = f"{HOME}/modelarts/user-job-dir/sol_h3"
LAT = f"{ROOT}/data/hrlat"
PIX = f"{ROOT}/data/pixels/hr"
VAL_CLIPS = 24
N_LAT, N_FRAMES = 5, 17


class Chunks:
    def __init__(self, ids):
        self.ids = list(ids)
        self._mm = {}

    def _get(self, vid):
        if vid not in self._mm:
            if len(self._mm) > 64:
                self._mm.clear()
            self._mm[vid] = (np.load(f"{LAT}/{vid}.npy", mmap_mode="r"),
                             np.load(f"{PIX}/{vid}.npy", mmap_mode="r"))
        return self._mm[vid]

    def sample(self, rng, crop):
        lat, pix = self._get(self.ids[rng.randrange(len(self.ids))])
        n_chunk = min(lat.shape[0] // N_LAT, pix.shape[0] // N_FRAMES)
        k = rng.randrange(n_chunk)
        cy = rng.randrange(0, lat.shape[2] - crop + 1)
        cx = rng.randrange(0, lat.shape[3] - crop + 1)
        z = np.asarray(lat[N_LAT * k:N_LAT * k + N_LAT, :, cy:cy + crop, cx:cx + crop],
                       dtype=np.float32)                       # [5,24,h,w]
        y = np.asarray(pix[N_FRAMES * k:N_FRAMES * k + N_FRAMES,
                           cy * 16:(cy + crop) * 16, cx * 16:(cx + crop) * 16])
        return z, y


def to_batch(items, device):
    z = torch.from_numpy(np.stack([a for a, _ in items])).to(device)
    z = z.permute(0, 2, 1, 3, 4).contiguous()                  # [B,24,5,h,w]
    y = torch.from_numpy(np.stack([b for _, b in items])).to(device)
    y = y.permute(0, 1, 4, 2, 3).flatten(0, 1).float().div_(127.5).sub_(1.0)
    return z, y


def psnr_from_mse(mse):
    return float(-10 * torch.log10(mse / 4 + 1e-12))


_LP = None


@torch.no_grad()
def evaluate(model, ds, device, crop, n=12, want_lpips=True):
    """Fixed probe so repeated evaluations agree exactly."""
    model.eval()
    if not hasattr(ds, "_probe"):
        pr = random.Random(20261012)
        ds._probe = [to_batch([ds.sample(pr, crop)], device) for _ in range(n)]
    global _LP
    if want_lpips and _LP is None:
        import lpips as _l
        _LP = _l.LPIPS(net="vgg").to(device).eval()
        for q in _LP.parameters():
            q.requires_grad_(False)
    ps, ls = [], []
    for z, y in ds._probe:
        with torch.autocast("npu", dtype=torch.bfloat16):
            o = model(z).float()
        ps.append(psnr_from_mse(F.mse_loss(o, y)))
        if want_lpips:
            ls.append(float(_LP(o.clamp(-1, 1), y).mean()))
    model.train()
    return sum(ps) / len(ps), (sum(ls) / len(ls) if ls else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--blocks", default="4,4,2,2,2")
    ap.add_argument("--crop", type=int, default=12, help="latent cells; x16 = pixels")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lpips", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sb = tuple(int(x) for x in args.blocks.split(","))

    device = "npu:0"
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    have = {f[:-4] for f in os.listdir(LAT) if f.endswith(".npy")} & \
           {f[:-4] for f in os.listdir(PIX) if f.endswith(".npy")
            and not f.endswith(".tmp.npy")}
    ids = sorted(have)
    random.Random(1234).shuffle(ids)
    val, train = Chunks(ids[:VAL_CLIPS]), Chunks(ids[VAL_CLIPS:])
    print(f"train {len(train.ids)} val {len(val.ids)} clips  crop {args.crop} cells "
          f"= {args.crop*16}px", flush=True)

    model = HRDecoder(args.width, sb).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"params {n/1e6:.2f}M  width {args.width} blocks {sb}", flush=True)

    import lpips as _l
    perc = _l.LPIPS(net="vgg").to(device).eval()
    for q in perc.parameters():
        q.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0,
                            betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.steps,
                                                pct_start=0.03, final_div_factor=50)
    start = 1
    if args.resume and args.resume.exists():
        rb = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(rb["model"]); start = rb["step"] + 1
        for _ in range(rb["step"]):
            sched.step()
        print(f"resumed at step {rb['step']}", flush=True)

    (args.out / "config.json").write_text(json.dumps(
        {**vars(args), "blocks": list(sb), "params": n}, indent=2, default=str))
    log = (args.out / "metrics.jsonl").open("a")
    rng = random.Random(args.seed)
    t0, best = time.time(), -1e9

    for step in range(start, args.steps + 1):
        z, y = to_batch([train.sample(rng, args.crop) for _ in range(args.batch)], device)
        with torch.autocast("npu", dtype=torch.bfloat16):
            o = model(z)
            of = o.float()
            loss = F.l1_loss(of, y)
            if args.lpips > 0:
                loss = loss + args.lpips * perc(of, y).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 100 == 0 or step == 1:
            rec = dict(step=step, loss=round(loss.item(), 5), grad_norm=round(float(gn), 3),
                       peak_gib=round(torch.npu.max_memory_allocated() / 2**30, 2),
                       elapsed=round(time.time() - t0, 1))
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()

        if step % args.eval_every == 0 or step == args.steps:
            p, lp = evaluate(model, val, device, args.crop)
            rec = dict(step=step, val_psnr=round(p, 3), elapsed=round(time.time() - t0, 1))
            if lp is not None:
                rec["val_lpips"] = round(lp, 4)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            blob = dict(model=model.state_dict(), step=step,
                        args=vars(args), blocks=list(sb))
            torch.save(blob, args.out / "latest.pt")
            # Rank by LPIPS when it is in the objective: PSNR and perceptual
            # quality ordered the arms differently in the refiner sweep.
            score = -lp if lp is not None else p
            if score > best:
                best = score
                torch.save(blob, args.out / "best.pt")

    (args.out / "COMPLETE.json").write_text(json.dumps(
        dict(complete=True, steps=args.steps, best=best,
             wall_seconds=time.time() - t0), indent=2))
    os._exit(0)


if __name__ == "__main__":
    # Guarded: hrdec_eval.py imports the data constants from here, and an
    # unguarded call made that import run argparse and exit.
    main()
