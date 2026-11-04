"""Train latent SR with the distilled decoder in the loop, loss measured in pixels.

The previous latent attempt optimised latent MSE and failed by 6.9 dB: a model
4.7 dB BETTER in latent MSE decoded 6.1 dB WORSE than plain bicubic, because the
VAE latent space is not isometric to pixels. So nothing here is scored in latent
space. The frozen student decoder is attached to the output and every loss and
metric is computed on the pixels it produces, which is the domain the result is
delivered in.

Baselines this has to clear, all measured through the same decoder on the same
clips (latent_probe.py):
    copy the keyframe   0.2550   <- where step 0 starts, by construction
    pixel pipeline      0.1910   <- what we already ship; below this is worthless
    true HR latent      0.1449   <- the ceiling, unreachable
"""
from __future__ import annotations

import argparse, json, os, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from latsr_model import LatentSR
from hrdec_model import HRDecoder

HOME = os.environ["HOME"]
ROOT = f"{HOME}/modelarts/user-job-dir/sol_h3"
HLAT = f"{ROOT}/data/hrlat"
PIX = f"{ROOT}/data/pixels/hr"
LLAT = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/low/video"
VAL_CLIPS = 24
N_LAT, N_FRAMES = 5, 17
FRAMES_PER_LAT = 124 / 37


class Clips:
    def __init__(self, ids):
        self.ids = list(ids)
        self._mm = {}

    def _get(self, vid):
        if vid not in self._mm:
            if len(self._mm) > 48:
                self._mm.clear()
            lo = torch.load(f"{LLAT}/{vid}.pt.part", map_location="cpu",
                            weights_only=False).permute(1, 0, 2, 3).float().numpy()
            self._mm[vid] = (np.load(f"{HLAT}/{vid}.npy", mmap_mode="r"),
                             np.load(f"{PIX}/{vid}.npy", mmap_mode="r"), lo)
        return self._mm[vid]

    def sample(self, rng, kf_chunks, crop, fixed=None):
        vid = self.ids[rng.randrange(len(self.ids))]
        hl, px, ll = self._get(vid)
        n = min(hl.shape[0] // N_LAT, px.shape[0] // N_FRAMES, ll.shape[0] // N_LAT)
        k = rng.randrange(n) if fixed is None else min(fixed[0], n - 1)
        j = rng.randint(0, kf_chunks) if fixed is None else fixed[1]
        kk = max(k - j, 0)
        lo = np.asarray(ll[N_LAT * k:N_LAT * k + N_LAT], dtype=np.float32)
        kf = np.asarray(hl[N_LAT * kk], dtype=np.float32)        # one HR latent frame
        # Pixel supervision on a latent-aligned crop: 1 HR latent cell = 16 px,
        # so a `crop` cell window is crop*16 pixels with no resampling.
        cy = rng.randrange(0, hl.shape[2] - crop + 1)
        cx = rng.randrange(0, hl.shape[3] - crop + 1)
        tgt = np.asarray(px[N_FRAMES * k:N_FRAMES * k + N_FRAMES,
                            cy * 16:(cy + crop) * 16, cx * 16:(cx + crop) * 16])
        d = (k - kk) * N_LAT * FRAMES_PER_LAT
        hr = np.asarray(hl[N_LAT * k:N_LAT * k + N_LAT], dtype=np.float32)
        return lo, kf, tgt, (cy, cx), np.float32(d), hr


def to_batch(items, device, crop):
    lo = torch.from_numpy(np.stack([it[0] for it in items])).to(device)
    lo = lo.flatten(0, 1)                                        # [B*5,24,16,28]
    kf = torch.from_numpy(np.stack([it[1] for it in items])).to(device)
    tgt = torch.from_numpy(np.stack([it[2] for it in items])).to(device)
    tgt = tgt.permute(0, 1, 4, 2, 3).flatten(0, 1).float().div_(127.5).sub_(1.0)
    crops = [it[3] for it in items]
    dist = torch.tensor([[it[4]] for it in items], device=device)
    return lo, kf, tgt, crops, dist


def crop_latent(z, crops, crop, B, T):
    """z: [B*T,24,48,84] -> [B,24,T,crop,crop] per-sample crop, decoder-ready."""
    z = z.unflatten(0, (B, T))
    out = [z[i, :, :, cy:cy + crop, cx:cx + crop] for i, (cy, cx) in enumerate(crops)]
    return torch.stack(out).permute(0, 2, 1, 3, 4).contiguous()


def psnr_from_mse(mse):
    return float(-10 * torch.log10(mse / 4 + 1e-12))


_LP = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", type=Path, default=Path("results/hrdec_1200/s768/best.pt"))
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--lr-blocks", type=int, default=8)
    ap.add_argument("--hr-blocks", type=int, default=6)
    ap.add_argument("--crop", type=int, default=12, help="HR latent cells; x16 = px")
    ap.add_argument("--kf-chunks", type=int, default=2,
                    help="keyframe sampled 0..N chunks back (1 chunk = 17 frames)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lpips", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dev = "npu:0"
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    db = torch.load(args.decoder, map_location="cpu", weights_only=False)
    dec = HRDecoder(db["args"]["width"], tuple(db["blocks"])).to(dev).eval()
    dec.load_state_dict(db["model"])
    for p in dec.parameters():
        p.requires_grad_(False)
    print(f"decoder {args.decoder} frozen, "
          f"{sum(p.numel() for p in dec.parameters())/1e6:.1f}M", flush=True)

    ids = sorted({f[:-4] for f in os.listdir(HLAT) if f.endswith(".npy")} &
                 {f[:-4] for f in os.listdir(PIX) if f.endswith(".npy")} &
                 {f[:-8] for f in os.listdir(LLAT) if f.endswith(".pt.part")})
    random.Random(1234).shuffle(ids)
    val, train = Clips(ids[:VAL_CLIPS]), Clips(ids[VAL_CLIPS:])
    print(f"train {len(train.ids)} val {len(val.ids)}  crop {args.crop} cells "
          f"= {args.crop*16}px  keyframe 0..{args.kf_chunks} chunks back", flush=True)

    model = LatentSR(args.width, args.lr_blocks, args.hr_blocks).to(dev)
    n = sum(p.numel() for p in model.parameters())
    print(f"params {n/1e6:.2f}M", flush=True)

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

    def run(items, B):
        lo, kf, tgt, crops, dist = to_batch(items, dev, args.crop)
        kfeat = model.encode_key(kf)
        kf_rep = kf.repeat_interleave(N_LAT, 0)
        z = model(lo, N_LAT, kfeat, kf_rep, dist)              # [B*5,24,48,84]
        zc = crop_latent(z, crops, args.crop, B, N_LAT)
        out = dec(zc)                                          # [B*17,3,px,px]
        return out, tgt

    @torch.no_grad()
    def evaluate(n_probe=12):
        model.eval()
        if not hasattr(val, "_probe"):
            pr = random.Random(20261012)
            val._probe = [[val.sample(pr, args.kf_chunks, args.crop)] for _ in range(n_probe)]
        global _LP
        if _LP is None:
            _LP = perc
        ps, ls = [], []
        for items in val._probe:
            with torch.autocast("npu", dtype=torch.bfloat16):
                o, y = run(items, 1)
            o = o.float()
            ps.append(psnr_from_mse(F.mse_loss(o, y)))
            ls.append(float(_LP(o.clamp(-1, 1), y).mean()))
        model.train()
        return sum(ps) / len(ps), sum(ls) / len(ls)

    for step in range(start, args.steps + 1):
        items = [train.sample(rng, args.kf_chunks, args.crop) for _ in range(args.batch)]
        with torch.autocast("npu", dtype=torch.bfloat16):
            out, tgt = run(items, args.batch)
            of = out.float()
            loss = F.l1_loss(of, tgt) + args.lpips * perc(of, tgt).mean()
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
            p, lp = evaluate()
            rec = dict(step=step, val_psnr=round(p, 3), val_lpips=round(lp, 4),
                       elapsed=round(time.time() - t0, 1))
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            blob = dict(model=model.state_dict(), step=step, args=vars(args))
            torch.save(blob, args.out / "latest.pt")
            if -lp > best:
                best = -lp
                torch.save(blob, args.out / "best.pt")

    (args.out / "COMPLETE.json").write_text(json.dumps(
        dict(complete=True, steps=args.steps, best_lpips=-best,
             wall_seconds=time.time() - t0), indent=2))
    os._exit(0)


if __name__ == "__main__":
    main()
