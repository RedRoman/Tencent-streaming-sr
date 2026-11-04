"""Train a Sol-H3 LR-conditioned HR latent refiner (flow matching).

One arm = one temporal window. Architecture, weights, data, schedule and seed
are identical across arms; only `--chunk` (how many consecutive latent frames
the refiner sees at once) changes, so a quality difference is attributable to
temporal context alone. `--cond none` is the control that removes the LR
channels entirely.

Objective is rectified flow on the HR latent, matching H3's own training signal:
    x_t = (1-t)*x_HR + t*noise,  target velocity = noise - x_HR.
The LR condition is clean at every t.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import torch_npu  # noqa: F401

from solh3_model import SolH3Refiner

LOW = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/low/video"
HIGH = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/high/video"
UPSCALER = "/temp/h3_training_assets_20260921/H3-LBH-Latent-Upscaler/minimax_h3_latent_upscaler_3d_bf16.safetensors"
DEGRADED = "/home/ma-user/modelarts/user-job-dir/sol_h3/data/degraded_lr"
VAL_IDS = 64  # held out from the 800, never trained on


class Pairs:
    """Lazy LR/HR latent pairs with an in-RAM cache (~7GB for all 800)."""

    def __init__(self, ids, lr_source="h3"):
        self.ids = ids
        self.lr_source = lr_source
        self._cache = {}

    def __len__(self):
        return len(self.ids)

    def get(self, i):
        vid = self.ids[i]
        if vid not in self._cache:
            if self.lr_source == "degraded":
                lo = torch.load(f"{DEGRADED}/{vid}.pt", map_location="cpu", weights_only=False)
            else:
                lo = torch.load(f"{LOW}/{vid}.pt.part", map_location="cpu", weights_only=False)
            hi = torch.load(f"{HIGH}/{vid}.pt.part", map_location="cpu", weights_only=False)
            self._cache[vid] = (lo.float(), hi.float())
        return self._cache[vid]

    def sample(self, rng, chunk):
        """One [24,chunk,48,84] HR target plus its upsampled LR condition."""
        lo, hi = self.get(rng.randrange(len(self.ids)))
        t0 = rng.randrange(0, hi.shape[1] - chunk + 1)
        hr = hi[:, t0 : t0 + chunk]
        lr = lo[:, t0 : t0 + chunk]
        return lr, hr


def upsample_lr(lr):
    """LR [B,24,T,16,28] -> HR grid [B,24,T,48,84] by trilinear 3x spatial."""
    b, c, t, h, w = lr.shape
    out = F.interpolate(lr.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w),
                        scale_factor=3, mode="bilinear", align_corners=False)
    return out.reshape(b, t, c, h * 3, w * 3).permute(0, 2, 1, 3, 4).contiguous()


def batch(ds, rng, bs, chunk, device, raw_lr=False):
    """Returns (lr_cond, hr). With `raw_lr` the LR keeps its own 16x28 grid and a
    learned head inside the model does the re-gridding."""
    lrs, hrs = zip(*(ds.sample(rng, chunk) for _ in range(bs)))
    lr = torch.stack(lrs).to(device, non_blocking=True)
    hr = torch.stack(hrs).to(device, non_blocking=True)
    return (lr if raw_lr else upsample_lr(lr)), hr


@torch.no_grad()
def evaluate(model, ds, chunk, device, rng, n=32, objective="noise", raw_lr=False):
    """Mean velocity MSE and one-step-to-clean latent PSNR on held-out clips.

    The probe set is built once from a fixed seed and reused by every call, so
    repeated evaluations of one checkpoint agree and arms differ only by what
    they learned. Drawing fresh clips/timesteps/noise each time put ~0.44 dB of
    sampling noise on this number -- wider than the gaps between arms.
    """
    model.eval()
    if not hasattr(ds, "_probe") or ds._probe_chunk != (chunk, objective, raw_lr):
        probe_rng = random.Random(20261009)
        probe = []
        for _ in range(n):
            lr, hr = batch(ds, probe_rng, 2, chunk, device, raw_lr)
            g = torch.Generator(device="cpu").manual_seed(len(probe))
            t = torch.rand(hr.shape[0], generator=g).to(device)
            # The bridge endpoint lives on the HR grid, so a raw LR must be
            # interpolated for it even when the model re-grids internally.
            if objective == "bridge":
                src = upsample_lr(lr) if raw_lr else lr
            else:
                src = torch.randn(hr.shape, generator=g).to(device)
            probe.append((lr, hr, t, src))
        ds._probe = probe
        ds._probe_chunk = (chunk, objective, raw_lr)
    vel, psnr = [], []
    for lr, hr, t, src in ds._probe:
        xt = (1 - t).view(-1, 1, 1, 1, 1) * hr + t.view(-1, 1, 1, 1, 1) * src
        with torch.autocast("npu", dtype=torch.bfloat16):
            pred = model(xt, lr, t).float()
        vel.append(F.mse_loss(pred, src - hr).item())
        x0 = xt - t.view(-1, 1, 1, 1, 1) * pred
        psnr.append(-10 * torch.log10(F.mse_loss(x0, hr) + 1e-12).item())
    model.train()
    return sum(vel) / len(vel), sum(psnr) / len(psnr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, required=True, help="latent frames the refiner sees")
    ap.add_argument("--cond", choices=["concat", "none"], default="concat")
    ap.add_argument("--learned-upsample", action="store_true",
                    help="re-grid the LR condition with a trained head instead of "
                         "interpolation (LR and HR latents are different coordinate "
                         "systems, so interpolation is a bad prior)")
    ap.add_argument("--objective", choices=["noise", "bridge"], default="noise",
                    help="noise: flow from pure noise (H3's own training signal). "
                         "bridge: flow from the upsampled LR, so t=1 already carries "
                         "the condition and few-step sampling starts near the answer.")
    ap.add_argument("--lr-source", choices=["h3", "degraded"], default="h3",
                    help="h3: H3's own LR generation; degraded: bicubic-downsampled HR")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--frames-per-step", type=int, default=40,
                    help="batch*chunk held constant across arms so every arm sees "
                         "the same number of HR latent frames per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--time-taps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "npu:0"
    args.out.mkdir(parents=True, exist_ok=True)
    args.batch = max(1, args.frames_per_step // args.chunk)

    ids = sorted(f[:-8] for f in os.listdir(HIGH) if f.endswith(".pt.part"))
    rng_split = random.Random(1234)
    rng_split.shuffle(ids)
    val_ds = Pairs(ids[:VAL_IDS], args.lr_source)
    train_ds = Pairs(ids[VAL_IDS:], args.lr_source)
    print(f"train {len(train_ds)} val {len(val_ds)} clips lr_source={args.lr_source}", flush=True)

    model = SolH3Refiner(width=args.width, time_taps=args.time_taps, cond_mode=args.cond,
                         learned_upsample=args.learned_upsample).to(device)
    params = sum(p.numel() for p in model.parameters())
    if not args.no_pretrained:
        loaded, skipped = model.load_pretrained_upscaler(UPSCALER)
        print(f"seeded from upscaler: {loaded} keys loaded, {skipped} skipped", flush=True)
    model = model.to(device)
    model.grad_checkpointing = True
    print(f"params {params/1e6:.1f}M  chunk={args.chunk} batch={args.batch} "
          f"({args.batch*args.chunk} frames/step) cond={args.cond} taps={args.time_taps}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.05, final_div_factor=20)
    rng = random.Random(args.seed)
    rng_val = random.Random(999)

    (args.out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    log = (args.out / "metrics.jsonl").open("a")
    t_start = time.time()
    best = -1e9

    for step in range(1, args.steps + 1):
        lr_cond, hr = batch(train_ds, rng, args.batch, args.chunk, device,
                            raw_lr=args.learned_upsample)
        t = torch.rand(hr.shape[0], device=device)
        if args.objective == "bridge":
            src = upsample_lr(lr_cond) if args.learned_upsample else lr_cond
        else:
            src = torch.randn_like(hr)
        xt = (1 - t).view(-1, 1, 1, 1, 1) * hr + t.view(-1, 1, 1, 1, 1) * src

        with torch.autocast("npu", dtype=torch.bfloat16):
            pred = model(xt, lr_cond, t)
            loss = F.mse_loss(pred.float(), src - hr)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 50 == 0 or step == 1:
            rec = dict(step=step, loss=round(loss.item(), 5), grad_norm=round(float(gn), 3),
                       lr=round(sched.get_last_lr()[0], 7),
                       peak_gib=round(torch.npu.max_memory_allocated() / 2**30, 2),
                       elapsed=round(time.time() - t_start, 1))
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()

        if step % args.eval_every == 0 or step == args.steps:
            v, p = evaluate(model, val_ds, args.chunk, device, None, objective=args.objective,
                            raw_lr=args.learned_upsample)
            rec = dict(step=step, val_velocity_mse=round(v, 5), val_latent_psnr=round(p, 3),
                       elapsed=round(time.time() - t_start, 1))
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            torch.save({"model": model.state_dict(), "step": step, "args": vars(args)},
                       args.out / "latest.pt")
            if p > best:
                best = p
                torch.save({"model": model.state_dict(), "step": step, "args": vars(args),
                            "val_latent_psnr": p}, args.out / "best.pt")

    (args.out / "COMPLETE.json").write_text(json.dumps(
        dict(complete=True, steps=args.steps, best_val_latent_psnr=best,
             wall_seconds=time.time() - t_start), indent=2))
    print(f"done in {(time.time()-t_start)/60:.1f} min, best val latent PSNR {best:.3f}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
