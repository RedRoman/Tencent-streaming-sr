"""Train the flow-matching manifold-alignment stage.

Convention: t=1 is the stage-1 estimate (the source), t=0 is the true HR latent
(the target), z_t = (1-t)*z_hr + t*z_src, and the velocity target is
(z_src - z_hr) -- integrating from t=1 down to 0 moves the estimate onto the
manifold. The head is zero-initialised, so an untrained model integrates to
exactly the stage-1 output and every gain is attributable.

Loss has two terms and they are there for different reasons:
  velocity MSE   flow matching needs a latent-space objective; there is no way
                 around that.
  pixel loss     on the one-step x0 estimate, decoded through the frozen
                 distilled decoder. This is the guard rail: latent MSE and pixel
                 quality are NOT monotonically related in this space (measured:
                 a model 4.7 dB better in latent decoded 6.1 dB worse), so a
                 purely latent objective is exactly how this route failed last
                 time. Checkpoint selection is by pixel LPIPS only.

The LR condition is noise-augmented with a random sigma, which SimpleGVR
identifies as one of the two choices that decide whether cascaded SR works.
"""
from __future__ import annotations

import argparse, json, os, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from latflow_model import LatentFlow
from latsr_model import LatentSR
from hrdec_model import HRDecoder
from latsr_train import (HLAT, PIX, LLAT, VAL_CLIPS, N_LAT, N_FRAMES,
                         FRAMES_PER_LAT, Clips, to_batch, crop_latent, psnr_from_mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", type=Path, default=Path("results/latsr_2200/w256/best.pt"))
    ap.add_argument("--decoder", type=Path, default=Path("results/hrdec_1200/s768/best.pt"))
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--crop", type=int, default=16)
    ap.add_argument("--kf-chunks", type=int, default=2)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--pixel", type=float, default=1.0, help="weight on the decoded x0 loss")
    ap.add_argument("--lpips", type=float, default=1.0)
    ap.add_argument("--cond-noise", type=float, default=0.3,
                    help="max sigma of the noise added to the LR condition")
    ap.add_argument("--src-sigma", type=float, default=0.5,
                    help="max sigma of the noise perturbing the flow SOURCE. This "
                         "is what makes the stage generative rather than another "
                         "deterministic map (LSRNA's noise addition).")
    ap.add_argument("--nfe", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dev = "npu:0"
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    s1b = torch.load(args.stage1, map_location="cpu", weights_only=False)
    sa = s1b["args"]
    stage1 = LatentSR(sa["width"], sa["lr_blocks"], sa["hr_blocks"]).to(dev).eval()
    stage1.load_state_dict(s1b["model"])
    for p in stage1.parameters():
        p.requires_grad_(False)
    db = torch.load(args.decoder, map_location="cpu", weights_only=False)
    dec = HRDecoder(db["args"]["width"], tuple(db["blocks"])).to(dev).eval()
    dec.load_state_dict(db["model"])
    for p in dec.parameters():
        p.requires_grad_(False)
    print(f"stage1 {args.stage1} (step {s1b['step']}) + decoder frozen", flush=True)

    ids = sorted({f[:-4] for f in os.listdir(HLAT) if f.endswith(".npy")} &
                 {f[:-4] for f in os.listdir(PIX) if f.endswith(".npy")} &
                 {f[:-8] for f in os.listdir(LLAT) if f.endswith(".pt.part")})
    random.Random(1234).shuffle(ids)
    val, train = Clips(ids[:VAL_CLIPS]), Clips(ids[VAL_CLIPS:])

    model = LatentFlow(args.width, args.blocks).to(dev)
    n = sum(p.numel() for p in model.parameters())
    print(f"flow params {n/1e6:.2f}M  crop {args.crop} cells  cond-noise "
          f"<= {args.cond_noise}", flush=True)

    import lpips as _l
    perc = _l.LPIPS(net="vgg").to(dev).eval()
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
        {**vars(args), "params": n}, indent=2, default=str))
    log = (args.out / "metrics.jsonl").open("a")
    rng = random.Random(args.seed)
    t0, best = time.time(), -1e9

    def prep(items, B, train_mode=True):
        lo, kf, tgt, crops, dist = to_batch(items, dev, args.crop)
        with torch.no_grad():
            kf_rep = kf.repeat_interleave(N_LAT, 0)
            z_src = stage1(lo, N_LAT, stage1.encode_key(kf), kf_rep, dist)
        lr_up = F.interpolate(lo, size=z_src.shape[-2:], mode="bilinear",
                              align_corners=False)
        if train_mode and args.cond_noise > 0:
            # Random sigma per sample, so the model cannot assume a clean LR.
            s = torch.rand(lr_up.shape[0], 1, 1, 1, device=dev) * args.cond_noise
            lr_up = lr_up + torch.randn_like(lr_up) * s
        return lo, kf, tgt, crops, dist, z_src, lr_up

    @torch.no_grad()
    def evaluate(n_probe=12):
        model.eval()
        if not hasattr(val, "_fprobe"):
            pr = random.Random(20261013)
            val._fprobe = [[val.sample(pr, args.kf_chunks, args.crop)]
                           for _ in range(n_probe)]
        # Two paths, because adding noise at inference broke the old clean
        # attribution (zero-init no longer means "step 0 == stage 1"):
        #   det   sigma=0, the deterministic ODE -- comparable to the last run
        #   gen   sigma>0, the generative path this change exists to enable
        # Reporting only one of them would hide whether the noise is helping or
        # simply being removed again.
        ps, ls, bs, gs = [], [], [], []
        for items in val._fprobe:
            lo, kf, tgt, crops, dist, z_src, lr_up = prep(items, 1, train_mode=False)
            g = torch.Generator(device=dev).manual_seed(1234)   # same draw every eval
            with torch.autocast("npu", dtype=torch.bfloat16):
                zd = model.sample(z_src, lr_up, kf, dist, nfe=args.nfe, z1=z_src)
                zg = model.sample(z_src, lr_up, kf, dist, nfe=args.nfe, z1=z_src,
                                  sigma=args.src_sigma * 0.5, generator=g)
                od = dec(crop_latent(zd, crops, args.crop, 1, N_LAT)).float()
                og = dec(crop_latent(zg, crops, args.crop, 1, N_LAT)).float()
                b = dec(crop_latent(z_src, crops, args.crop, 1, N_LAT)).float()
            ps.append(psnr_from_mse(F.mse_loss(od, tgt)))
            ls.append(float(perc(od.clamp(-1, 1), tgt).mean()))
            gs.append(float(perc(og.clamp(-1, 1), tgt).mean()))
            bs.append(float(perc(b.clamp(-1, 1), tgt).mean()))
        model.train()
        return (sum(ps) / len(ps), sum(ls) / len(ls), sum(bs) / len(bs),
                sum(gs) / len(gs))

    for step in range(start, args.steps + 1):
        items = [train.sample(rng, args.kf_chunks, args.crop) for _ in range(args.batch)]
        lo, kf, tgt, crops, dist, z_src, lr_up = prep(items, args.batch)
        # The flow's target: the true HR latent for this chunk, [B*5,24,48,84].
        z_hr = torch.from_numpy(np.stack([it[5] for it in items])).to(dev).flatten(0, 1)
        # Stochastic source: a different perturbation every step, so the model
        # must learn to transport a WHOLE NEIGHBOURHOOD of starts onto the
        # manifold rather than memorise one deterministic correction.
        sig = torch.rand(z_src.shape[0], 1, 1, 1, device=dev) * args.src_sigma
        z_start = z_src + torch.randn_like(z_src) * sig
        t = torch.rand(z_src.shape[0], device=dev)
        z_t = (1 - t)[:, None, None, None] * z_hr + t[:, None, None, None] * z_start
        with torch.autocast("npu", dtype=torch.bfloat16):
            v = model(z_t, lr_up, kf, t, dist, z_src)
            vf = v.float()
            vel = F.mse_loss(vf, (z_start - z_hr))
            # one-step x0 from the current point, decoded and scored in pixels
            x0 = z_t.float() - t[:, None, None, None] * vf
            o = dec(crop_latent(x0, crops, args.crop, args.batch, N_LAT)).float()
            pix = F.l1_loss(o, tgt) + args.lpips * perc(o, tgt).mean()
            loss = vel + args.pixel * pix
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 100 == 0 or step == 1:
            rec = dict(step=step, loss=round(loss.item(), 5), vel=round(float(vel), 5),
                       pix=round(float(pix), 5), grad_norm=round(float(gn), 3),
                       peak_gib=round(torch.npu.max_memory_allocated() / 2**30, 2),
                       elapsed=round(time.time() - t0, 1))
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()

        if step % args.eval_every == 0 or step == args.steps:
            p, lp, base, lg = evaluate()
            rec = dict(step=step, val_psnr=round(p, 3), val_lpips=round(lp, 4),
                       gen_lpips=round(lg, 4), stage1_lpips=round(base, 4),
                       gain_det=round(base - lp, 4), gain_gen=round(base - lg, 4),
                       elapsed=round(time.time() - t0, 1))
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            blob = dict(model=model.state_dict(), step=step, args=vars(args))
            torch.save(blob, args.out / "latest.pt")
            if -min(lp, lg) > best:
                best = -min(lp, lg)
                torch.save(blob, args.out / "best.pt")

    (args.out / "COMPLETE.json").write_text(json.dumps(
        dict(complete=True, steps=args.steps, best_lpips=-best,
             wall_seconds=time.time() - t0), indent=2))
    os._exit(0)


if __name__ == "__main__":
    main()
