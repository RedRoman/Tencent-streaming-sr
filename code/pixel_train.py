"""Train the pixel-domain Sol-H3 refiner.

Loss and evaluation are both pixel PSNR against the HR decode, the same number
the demo is judged by, so the proxy/deliverable mismatch that sank the latent
route cannot recur. bicubic is logged every time as the bar to beat.

Arms differ in exactly one setting at a time:
  --window N        frames processed jointly (1 = strictly streaming)
  --lr-source       h3 (H3's own LR generation) | degraded (bicubic-downsampled HR)
"""
from __future__ import annotations

import argparse, json, os, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from pixel_model import PixelSR, KeyframeRefLatent

HOME = os.environ["HOME"]
PIX = f"{HOME}/modelarts/user-job-dir/sol_h3/data/pixels"
TEXT = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/low/text"
HIGH_LAT = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/high/video"
HR_LAT_NPY = f"{HOME}/modelarts/user-job-dir/sol_h3/data/hrlat"
# One HR latent cell covers 16 HR pixels = 16/3 LR pixels, and the clip has
# 3.351 pixel frames per latent (124/37, the token_drop accounting).
LAT_PER_LRPX = 3 / 16
FRAMES_PER_LAT = 124 / 37
VAL_CLIPS = 24
CROP_LR = 64  # 64x64 LR crop -> 192x192 HR
TEXT_DIM = 5120


_TEXT_CACHE = {}
_LAT_CACHE = {}


def hr_latent(vid):
    """H3's HR latent for the clip, [T,24,48,84] -- no VAE decode involved.

    Memory-mapped from the local fp16 mirror (make_hrlat_cache.py). Reading the
    .pt.part files off the shared mount instead cost 0.61 s/step against 0.22
    without a reference: 48 clips per step is 48 NFS round trips.
    """
    if vid not in _LAT_CACHE:
        _LAT_CACHE[vid] = np.load(f"{HR_LAT_NPY}/{vid}.npy", mmap_mode="r")
    return _LAT_CACHE[vid]


def pooled_text(vid):
    """H3's prompt embedding for this clip, mean-pooled over tokens.

    [117,5120] per clip and constant for the whole clip, so pooling once and
    caching costs nothing. This is the H3-internal signal; a conventional SR
    model has no equivalent.
    """
    if vid not in _TEXT_CACHE:
        import torch as _t
        d = _t.load(f"{TEXT}/{vid}.pt.part", map_location="cpu", weights_only=False)
        _TEXT_CACHE[vid] = d["prompt_embeds"].float().mean(0)
    return _TEXT_CACHE[vid]


class Clips:
    """Memory-mapped uint8 pixel pairs; crops are drawn per sample."""

    def __init__(self, ids, lr_source="h3"):
        self.ids = list(ids)
        self._seen = set(self.ids)
        self.lr_source = lr_source
        self._mm = {}

    def _get(self, vid):
        if vid not in self._mm:
            self._mm[vid] = (np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r"),
                             np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r"))
        return self._mm[vid]

    def refresh(self, all_ids):
        """Adopt newly decoded clips (decode and training overlap)."""
        new = [i for i in all_ids if i not in self._seen]
        if new:
            self.ids.extend(new)
            self._seen.update(new)
        return len(new)

    def sample(self, rng, window, ref_max=0, ref_d=None, ref_latent=False):
        vid = self.ids[rng.randrange(len(self.ids))]
        lr, hr = self._get(vid)
        T = lr.shape[0]
        # With a keyframe reference the window must sit far enough into the clip
        # that the keyframe actually exists `ref_max` frames back.
        lo = ref_max if ref_max else 0
        t0 = rng.randrange(lo, T - window + 1)
        if ref_latent:
            # 3*y must be a multiple of 16 so the HR-latent crop is exact rather
            # than resampled; y therefore steps in 16s.
            y = rng.randrange(0, (lr.shape[1] - CROP_LR) // 16 + 1) * 16
            x = rng.randrange(0, (lr.shape[2] - CROP_LR) // 16 + 1) * 16
        else:
            y = rng.randrange(0, lr.shape[1] - CROP_LR + 1)
            x = rng.randrange(0, lr.shape[2] - CROP_LR + 1)
        b = np.asarray(hr[t0:t0 + window, y * 3:(y + CROP_LR) * 3, x * 3:(x + CROP_LR) * 3])
        if self.lr_source == "degraded":
            # The conventional-SR condition: LR obtained by downsampling HR,
            # rather than H3 generating natively at low resolution. Deriving it
            # from the HR crop needs no extra decode pass.
            t = torch.from_numpy(b).permute(0, 3, 1, 2).float()
            t = F.interpolate(t, scale_factor=1 / 3, mode="bicubic",
                              align_corners=False).clamp(0, 255)
            a = t.permute(0, 2, 3, 1).round().to(torch.uint8).numpy()
        else:
            a = np.asarray(lr[t0:t0 + window, y:y + CROP_LR, x:x + CROP_LR])
        if not ref_max:
            return a, b, vid, None, 0
        # The keyframe is a PAST frame -- streaming, no lookahead -- cropped at
        # the same spatial position, so any content shift is real camera/object
        # motion rather than a framing artefact.
        d = rng.randint(0, ref_max) if ref_d is None else ref_d
        if ref_latent:
            z = hr_latent(vid)
            tl = min(int(round((t0 - d) / FRAMES_PER_LAT)), z.shape[0] - 1)
            cy, cx = int(y * LAT_PER_LRPX), int(x * LAT_PER_LRPX)
            cn = int(CROP_LR * LAT_PER_LRPX)
            r = np.asarray(z[tl, :, cy:cy + cn, cx:cx + cn], dtype=np.float32)
        else:
            r = np.asarray(hr[t0 - d, y * 3:(y + CROP_LR) * 3, x * 3:(x + CROP_LR) * 3])
        # The LR frame at the keyframe's time, same crop: differencing it with
        # `a` gives the local change for free, in the one domain where both are
        # available and aligned.
        lr_kf = np.asarray(lr[t0 - d, y:y + CROP_LR, x:x + CROP_LR])
        return a, b, vid, r, d, lr_kf


def to_t(arr, device):
    """uint8 [T,H,W,3] -> float [-1,1] [T,3,H,W]."""
    t = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
    return t.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)


def batch(ds, rng, bs, window, device, want_text=False, ref_max=0, ref_d=None,
          ref_latent=False):
    items = [ds.sample(rng, window, ref_max, ref_d, ref_latent) for _ in range(bs)]
    lrs = [it[0] for it in items]; hrs = [it[1] for it in items]
    vids = [it[2] for it in items]; refs = [it[3] for it in items]
    ds_ = [it[4] for it in items]
    lr = torch.cat([to_t(a, device) for a in lrs])
    hr = torch.cat([to_t(b, device) for b in hrs])
    txt = torch.stack([pooled_text(v) for v in vids]).to(device) if want_text else None
    if not ref_max:
        return lr, hr, txt, None, None, None
    if ref_latent:
        # already float latents, not uint8 pixels
        ref = torch.stack([torch.from_numpy(np.ascontiguousarray(r)) for r in refs]).to(device)
    else:
        ref = torch.cat([to_t(r[None], device) for r in refs])
    kfpx = torch.cat([to_t(it[5][None], device) for it in items])   # [B,3,h,w]
    rep_t = lr.shape[0] // kfpx.shape[0]
    if rep_t > 1:
        kfpx = kfpx.repeat_interleave(rep_t, 0)
    change = (lr - kfpx).abs().mean(1, keepdim=True)
    dist = torch.tensor(ds_, device=device, dtype=torch.float32)[:, None]
    return lr, hr, txt, ref, dist, change


def psnr_from_mse(mse):
    # Inputs are in [-1,1]; PSNR is defined on a peak-to-peak range of 2.
    return float(-10 * torch.log10(mse / 4 + 1e-12))


_EVAL_LPIPS = None


@torch.no_grad()
def evaluate(model, ds, window, device, n=24, want_lpips=False, shuffle_text=False):
    """Fixed probe set so repeated evaluations agree exactly."""
    model.eval()
    if not hasattr(ds, "_probe") or ds._probe_w != window:
        pr = random.Random(20261009)
        ds._probe = [batch(ds, pr, 2, window, device, want_text=True) for _ in range(n)]
        ds._probe_w = window
    global _EVAL_LPIPS
    if want_lpips and _EVAL_LPIPS is None:
        import lpips as _lp
        _EVAL_LPIPS = _lp.LPIPS(net="vgg").to(device).eval()
        for q in _EVAL_LPIPS.parameters():
            q.requires_grad_(False)
    ours, base, lo, lb = [], [], [], []
    for lr, hr, txt, _r, _d, _c in ds._probe:
        t_in = txt if model.film is not None else None
        if t_in is not None and shuffle_text:
            # Decisive control: hand the model the WRONG caption. If the score
            # does not move, the text branch is decoration, not information.
            t_in = t_in.roll(1, dims=0)
        with torch.autocast("npu", dtype=torch.bfloat16):
            out = model(lr, frames=window, text=t_in).float()
        bic = F.interpolate(lr, scale_factor=3, mode="bicubic", align_corners=False).clamp(-1, 1)
        ours.append(psnr_from_mse(F.mse_loss(out, hr)))
        base.append(psnr_from_mse(F.mse_loss(bic, hr)))
        if want_lpips:
            lo.append(float(_EVAL_LPIPS(out.clamp(-1, 1), hr).mean()))
            lb.append(float(_EVAL_LPIPS(bic, hr).mean()))
    model.train()
    if want_lpips:
        return (sum(ours) / len(ours), sum(base) / len(base),
                sum(lo) / len(lo), sum(lb) / len(lb))
    return sum(ours) / len(ours), sum(base) / len(base)


EVAL_DS = (0, 1, 2, 4, 8, 15, 30)


def _enc(model, ref, hw):
    """The latent reader needs the target grid; the pixel one derives it."""
    if isinstance(model.kref, KeyframeRefLatent):
        return model.kref.encode(ref, hw)
    return model.kref.encode(ref)


@torch.no_grad()
def evaluate_ref(model, ds, device, ref_max, n=10):
    """LPIPS as a function of keyframe distance, plus the wrong-clip control.

    A single aggregate number would hide the only thing worth knowing here: how
    fast the keyframe's value decays with distance. That curve is what sets the
    keyframe interval, and the no-motion-compensation lower bound it has to beat
    is in results/relay_probe.json.

    `shuffled` feeds a reference from a DIFFERENT clip. If the score does not
    move, the model is ignoring the reference and any apparent gain came from
    the extra parameters.
    """
    model.eval()
    key = ("_refprobe", ref_max)
    if getattr(model, "_refprobe_key", None) != key:
        # A FRESH rng per bucket, so every distance sees the identical clips,
        # crops and target frames and the only thing that varies is d. Sharing
        # one rng across buckets let content differences swamp the effect --
        # the smoke test came back non-monotonic (d=4 "better" than d=0).
        rl = isinstance(model.kref, KeyframeRefLatent)
        model._refprobe = {d: [batch(ds, random.Random(20261011 + i), 2, 1, device,
                                     want_text=False,
                                     ref_max=max(ref_max, max(EVAL_DS)), ref_d=d,
                                     ref_latent=rl)
                               for i in range(n)] for d in EVAL_DS}
        model._refprobe_key = key
    global _EVAL_LPIPS
    if _EVAL_LPIPS is None:
        import lpips as _lp
        _EVAL_LPIPS = _lp.LPIPS(net="vgg").to(device).eval()
        for q in _EVAL_LPIPS.parameters():
            q.requires_grad_(False)
    out = {}
    for d, probe in model._refprobe.items():
        vals, shuf = [], []
        for lr, hr, _t, ref, dist, chg in probe:
            with torch.autocast("npu", dtype=torch.bfloat16):
                rf = _enc(model, ref, lr.shape[-2:])
                o = model(lr, frames=1, ref_feat=rf, dist=dist, change=chg).float()
                os_ = model(lr, frames=1, ref_feat=rf.roll(1, dims=0), dist=dist,
                            change=chg).float()
            vals.append(float(_EVAL_LPIPS(o.clamp(-1, 1), hr).mean()))
            shuf.append(float(_EVAL_LPIPS(os_.clamp(-1, 1), hr).mean()))
        out[f"d{d}"] = round(sum(vals) / len(vals), 4)
        out[f"d{d}_shuf"] = round(sum(shuf) / len(shuf), 4)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--local-gate", type=int, default=0,
                    help="weigh the keyframe per position using the local "
                         "|LR_t - LR_keyframe|, not just the global distance")
    ap.add_argument("--ref-latent", type=int, default=0,
                    help="consume the keyframe as H3's HR LATENT instead of HR "
                         "pixels. The HR VAE needs >=5 latents and costs 5847 ms "
                         "for that minimum chunk, so pixels are not deployable.")
    ap.add_argument("--ref-max", type=int, default=0,
                    help="train with an HR keyframe reference sampled 0..N frames "
                         "back (0 = off). This is the RelayVSR mechanism: pay for "
                         "high resolution only on sparse keyframes.")
    ap.add_argument("--text", type=int, default=0,
                    help="number of FiLM injection sites fed by H3's own prompt "
                         "embedding (0 = off). This is the Sol-H3 signal a plain "
                         "SR model cannot see.")
    ap.add_argument("--lr-source", choices=["h3", "degraded"], default="h3")
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--temporal-taps", type=int, default=0,
                    help="0 = match window (1 when window==1)")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--clips-per-step", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lpips", type=float, default=0.0,
                    help="weight on LPIPS; L1 alone optimizes PSNR and leaves the "
                         "result soft, which is what the demo looks like today")
    ap.add_argument("--gan", type=float, default=0.0,
                    help="adversarial weight; the only term that can synthesize the "
                         "86%% of high-frequency energy 3x downsampling destroys")
    ap.add_argument("--gan-temporal", type=int, default=0,
                    help="if >0, the discriminator judges this many consecutive "
                         "frames jointly (must equal --window); the generator "
                         "stays per-frame")
    ap.add_argument("--gan-start", type=int, default=2000,
                    help="warm up on reconstruction first so the generator is not "
                         "chasing a discriminator from random init")
    ap.add_argument("--d-lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()

    taps = args.temporal_taps or (1 if args.window == 1 else 5)
    torch.manual_seed(args.seed)
    device = "npu:0"
    args.out.mkdir(parents=True, exist_ok=True)

    _hr = {f[:-4] for f in os.listdir(f"{PIX}/hr")
           if f.endswith(".npy") and not f.endswith(".tmp.npy")}
    _lr = {f[:-4] for f in os.listdir(f"{PIX}/lr")
           if f.endswith(".npy") and not f.endswith(".tmp.npy")}
    ids = sorted(_hr & _lr)
    random.Random(1234).shuffle(ids)
    val = Clips(ids[:VAL_CLIPS], args.lr_source)
    train = Clips(ids[VAL_CLIPS:], args.lr_source)
    print(f"train {len(train.ids)} val {len(val.ids)} clips  window={args.window} "
          f"taps={taps} lr_source={args.lr_source}", flush=True)

    model = PixelSR(args.width, args.blocks, 3, taps,
                    text_dim=TEXT_DIM if args.text else 0,
                    text_sites=max(args.text, 1),
                    keyframe_ref=("latent" if args.ref_latent else bool(args.ref_max)),
                    local_gate=bool(args.local_gate)).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"params {n/1e6:.2f}M", flush=True)

    perceptual = None
    if args.lpips > 0:
        import lpips as lpips_pkg
        perceptual = lpips_pkg.LPIPS(net="vgg").to(device).eval()
        for q in perceptual.parameters():
            q.requires_grad_(False)
        print(f"LPIPS enabled, weight {args.lpips}", flush=True)

    disc = d_opt = None
    if args.gan > 0:
        from gan_parts import PatchDiscriminator, TemporalPatchDiscriminator, d_loss, g_loss
        if args.gan_temporal:
            assert args.gan_temporal == args.window, \
                "the temporal discriminator consumes whole sampled windows"
            assert taps == 1, "the generator must stay memoryless; pass --temporal-taps 1"
            disc = TemporalPatchDiscriminator(args.gan_temporal).to(device)
        else:
            disc = PatchDiscriminator().to(device)
        d_opt = torch.optim.AdamW(disc.parameters(), lr=args.d_lr, betas=(0.5, 0.9))
        print(f"GAN enabled: weight {args.gan} from step {args.gan_start}, "
              f"temporal={args.gan_temporal or 'no'}, "
              f"D {sum(q.numel() for q in disc.parameters())/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.03, final_div_factor=50)
    start_step = 1
    if args.resume and args.resume.exists():
        rb = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(rb["model"])
        start_step = rb["step"] + 1
        for _ in range(rb["step"]):
            sched.step()
        print(f"resumed from {args.resume} at step {rb['step']}", flush=True)
    rng = random.Random(args.seed)
    (args.out / "config.json").write_text(json.dumps({**vars(args), "taps": taps,
                                                      "params": n}, indent=2, default=str))
    log = (args.out / "metrics.jsonl").open("a")
    t0, best = time.time(), -1e9

    for step in range(start_step, args.steps + 1):
        lr, hr, txt, ref, dist, chg = batch(train, rng, args.clips_per_step,
                                            args.window, device,
                                            want_text=bool(args.text),
                                            ref_max=args.ref_max,
                                            ref_latent=bool(args.ref_latent))
        with torch.autocast("npu", dtype=torch.bfloat16):
            rf = _enc(model, ref, lr.shape[-2:]) if model.kref is not None else None
            out = model(lr, frames=args.window, text=txt, ref_feat=rf, dist=dist,
                        change=chg)
            out_f = out.float()
            loss = F.l1_loss(out_f, hr)
            if perceptual is not None:
                # Inputs are already in [-1,1], the range LPIPS expects.
                loss = loss + args.lpips * perceptual(out_f, hr).mean()
        adv = 0.0
        if disc is not None and step >= args.gan_start:
            from gan_parts import d_loss, g_loss
            # Discriminator step on the detached sample.
            d_opt.zero_grad(set_to_none=True)
            dl = d_loss(disc, hr, out_f)
            dl.backward()
            d_opt.step()
            # Generator sees the updated discriminator.
            adv_t = g_loss(disc, out_f)
            loss = loss + args.gan * adv_t
            adv = float(adv_t.detach())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 500 == 0:
            have_hr = {f[:-4] for f in os.listdir(f"{PIX}/hr")
                       if f.endswith(".npy") and not f.endswith(".tmp.npy")}
            have_lr = {f[:-4] for f in os.listdir(f"{PIX}/lr")
                       if f.endswith(".npy") and not f.endswith(".tmp.npy")}
            avail = sorted(have_hr & have_lr)
            added = train.refresh([i for i in avail if i not in set(val.ids)])
            if added:
                print(json.dumps(dict(step=step, event="train_pool_grew",
                                      added=added, total=len(train.ids))), flush=True)

        if step % 100 == 0 or step == 1:
            rec = dict(step=step, loss=round(loss.item(), 5), grad_norm=round(float(gn), 3),
                       peak_gib=round(torch.npu.max_memory_allocated() / 2**30, 2),
                       elapsed=round(time.time() - t0, 1))
            if disc is not None:
                rec["adv"] = round(adv, 4)
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()

        if step % args.eval_every == 0 or step == args.steps:
            want = args.lpips > 0 or args.gan > 0
            ev = evaluate(model, val, args.window, device, want_lpips=want)
            if want:
                p, b, lp_o, lp_b = ev
            else:
                p, b = ev; lp_o = lp_b = None
            # Run the wrong-caption control alongside, so "does it actually use
            # the text" is answered by the same curve rather than a later guess.
            shuf = None
            if args.ref_max:
                rec_ref = evaluate_ref(model, val, device, args.ref_max)
            else:
                rec_ref = None
            if args.text:
                sv = evaluate(model, val, args.window, device, want_lpips=want,
                              shuffle_text=True)
                shuf = dict(psnr_shuftext=round(sv[0], 3))
                if want:
                    shuf["lpips_shuftext"] = round(sv[2], 4)
            rec = dict(step=step, val_psnr=round(p, 3), bicubic_psnr=round(b, 3),
                       gain=round(p - b, 3), elapsed=round(time.time() - t0, 1))
            if shuf:
                rec.update(shuf)
            if rec_ref:
                rec.update(rec_ref)
            if lp_o is not None:
                rec.update(val_lpips=round(lp_o, 4), lpips_bicubic=round(lp_b, 4),
                           lpips_gain=round(lp_b - lp_o, 4))
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            torch.save({"model": model.state_dict(), "step": step, "args": vars(args),
                        "taps": taps}, args.out / "latest.pt")
            score = -lp_o if lp_o is not None else p
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "step": step, "args": vars(args),
                            "taps": taps, "val_psnr": p}, args.out / "best.pt")

    (args.out / "COMPLETE.json").write_text(json.dumps(
        dict(complete=True, steps=args.steps, best_val_psnr=best,
             wall_seconds=time.time() - t0), indent=2))
    print(f"done {(time.time()-t0)/60:.1f} min  best val PSNR {best:.3f}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
