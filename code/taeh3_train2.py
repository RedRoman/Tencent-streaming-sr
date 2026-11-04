"""Train the TAEHV decoder for our H3 VAE latents, with the correct frame contract.

The first attempt trained 5 latents -> 5 frames, which is not what the model
emits: `_decode_h3_video` trims every 5*t_upscale chunk by frames_to_trim and
drops a 3*t_upscale tail, so T latents yield 4T-18 frames for T >= 10 and the
output of a 5-latent window is a degenerate stub. Training against the VAE's
first 5 frames taught the model a meaningless mapping -- it reached 26 dB on
that target yet lost 3 dB end-to-end.

Here the window is a multiple of 5 latents, the target is the VAE's decode of
the SAME latents sliced to the TAE's own output length, and the alignment is
asserted rather than assumed.
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


def tae_out_frames(T, t_upscale=4, trim=3):
    """Frames `_decode_h3_video` returns for T latents (whole chunks only)."""
    chunk = 5 * t_upscale
    n = (T * t_upscale // chunk) * chunk          # drop the partial chunk
    kept = n // chunk * (chunk - trim)            # per-chunk prefix trim
    return max(kept - 3 * t_upscale, 0)           # tail drop


class Clips:
    """H3 latents with the matching LR pixels (the VAE's own decode target)."""

    def __init__(self, ids, lat):
        self.ids = list(ids); self.lat = lat; self._z = {}; self._p = {}

    def get(self, vid):
        if vid not in self._z:
            self._z[vid] = torch.load(f"{LOW}/{vid}.pt.part", map_location="cpu",
                                      weights_only=False).float()
            self._p[vid] = np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")
        return self._z[vid], self._p[vid]

    def sample(self, rng):
        z, px = self.get(self.ids[rng.randrange(len(self.ids))])
        T = self.lat
        nf = tae_out_frames(T)
        # H3 latent i covers pixels [4i-3, 4i]; the TAE drops the first 3 frames
        # of each 20-frame chunk, so its first output frame is pixel index 3.
        # Offset measured against the real VAE decode: the TAE's first output
        # frame aligns with the window's first pixel frame (offset 0), not 4s+3.
        maxstart = min(z.shape[1] - T, (px.shape[0] - nf) // 4)
        if maxstart < 0:
            raise ValueError("clip too short for this window")
        s = rng.randrange(0, maxstart + 1)
        a = z[:, s:s + T]
        p0 = s * 4
        t = np.asarray(px[p0:p0 + nf])
        return a, t


def to_batch(ds, rng, bs, device):
    zs, ps = zip(*(ds.sample(rng) for _ in range(bs)))
    z = torch.stack(zs).to(device)
    p = torch.from_numpy(np.stack(ps)).to(device)
    return z, p.permute(0, 1, 4, 2, 3).float() / 255.0


def psnr01(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


@torch.no_grad()
def evaluate(model, ds, device, n=12):
    model.eval()
    if not hasattr(ds, "_probe"):
        pr = random.Random(20261010)
        ds._probe = [to_batch(ds, pr, 1, device) for _ in range(n)]
    out = []
    for z, p in ds._probe:
        o = model.decode_video(z.permute(0, 2, 1, 3, 4).contiguous(),
                               parallel=True, show_progress_bar=False).float().clamp(0, 1)
        k = min(o.shape[1], p.shape[1])
        out.append(psnr01(o[:, :k], p[:, :k]))
    model.train()
    return sum(out) / len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", type=int, default=10, help="latent window, multiple of 5")
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lpips", type=float, default=0.0)
    ap.add_argument("--init", choices=["pretrained", "scratch"], default="pretrained")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert args.latents % 5 == 0, "window must be a multiple of 5 latents"

    from taehv import TAEHV
    torch.manual_seed(args.seed)
    dev = "npu:0"; args.out.mkdir(parents=True, exist_ok=True)

    model = TAEHV(checkpoint_path=None, arch_name="taeh3")
    if args.init == "pretrained":
        sd = torch.load(str(PRETRAINED), map_location="cpu", weights_only=False)
        own = model.state_dict()
        ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(ok, strict=False)
        print(f"init pretrained: {len(ok)}/{len(own)} matched", flush=True)
    model = model.to(dev)
    for q in model.encoder.parameters():
        q.requires_grad_(False)
    params = [q for q in model.decoder.parameters() if q.requires_grad]
    nf = tae_out_frames(args.latents)
    print(f"decoder {sum(q.numel() for q in params)/1e6:.2f}M | "
          f"{args.latents} latents -> {nf} frames", flush=True)

    perceptual = None
    if args.lpips > 0:
        import lpips as lp
        perceptual = lp.LPIPS(net="vgg").to(dev).eval()
        for q in perceptual.parameters():
            q.requires_grad_(False)

    ids = sorted(f[:-8] for f in os.listdir(LOW) if f.endswith(".pt.part"))
    ids = [i for i in ids if (PIX / "lr" / f"{i}.npy").exists()]
    random.Random(1234).shuffle(ids)
    val, train = Clips(ids[:VAL_CLIPS], args.latents), Clips(ids[VAL_CLIPS:], args.latents)
    print(f"train {len(train.ids)} val {len(val.ids)}", flush=True)

    # sanity: the model's real output length must match what we sliced
    zc, pc = to_batch(val, random.Random(0), 1, dev)
    with torch.no_grad():
        probe = model.decode_video(zc.permute(0, 2, 1, 3, 4).contiguous(),
                                   parallel=True, show_progress_bar=False)
    print(json.dumps(dict(target_frames=int(pc.shape[1]),
                          model_frames=int(probe.shape[1]),
                          aligned=bool(probe.shape[1] == pc.shape[1]))), flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.03, final_div_factor=50)
    rng = random.Random(args.seed)
    (args.out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    log = (args.out / "metrics.jsonl").open("a")
    t0, best = time.time(), -1e9

    for step in range(1, args.steps + 1):
        z, p = to_batch(train, rng, args.batch, dev)
        with torch.autocast("npu", dtype=torch.bfloat16):
            o = model.decode_video(z.permute(0, 2, 1, 3, 4).contiguous(),
                                   parallel=True, show_progress_bar=False)
            of = o.float()
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
            pp = evaluate(model, val, dev)
            rec = dict(step=step, val_psnr=round(pp, 3), elapsed=round(time.time() - t0, 1))
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
