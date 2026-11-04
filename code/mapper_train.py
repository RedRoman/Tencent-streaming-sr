"""Train the H3->Wan latent mapper.

Supervision is the Wan VAE's own encode of the HR frames, so the target is a
real point in Wan latent space rather than a proxy of our own invention -- the
mistake that sank the earlier H3-latent route. Progress is reported both as
latent MSE and, every eval, as the PSNR obtained after decoding through the
frozen Wan VAE, which is the number the demo is judged by.

Pipeline being trained:
  H3 LR latent [24,T,16,28]
    -> Mapper (20M, learned)         -> Wan [16,T,32,56]
    -> UltraFlash upsampler (frozen) -> Wan [16,T,96,168]
  target: Wan encode of HR pixels    == [16,T,96,168]
"""
from __future__ import annotations

import argparse, importlib.util, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
PAIRS = ROOT / "data/wanpairs"
CKPT = "/temp/liusonghua/h3_to_wan_repro_32193/joint_mapper_upsampler_wan_dit_step_002000.pt"
UF_UP = "/temp/rongxiang_workspace_backup_20260921/rongxiang/UltraFlash/inference/checkpoints/2K-ultraLatentUpsampler-step_28100.pth"
VAL_CLIPS = 24
SEG = 7  # the mapper's native temporal unit: 5 main + 2 tail


def load_upsampler(device, from_joint=True):
    spec = importlib.util.spec_from_file_location("uf",
        ROOT / "h2w/vendor/UltraFlash/inference/sr/latent_upsampler/ultra_latent_up_v1.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    up = mod.UltraLatentUpsampler(in_channels=16, out_channels=16, mid_channels=128,
        num_blocks=8, upsample_scale=3, activation="relu", residual=False,
        memory_init="replicate", zero_init_final=False, default_parallel=True)
    if from_joint:
        sd = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)["model"]
        up.load_state_dict({k[16:]: v for k, v in sd.items()
                            if k.startswith("joint.upsampler.")}, strict=True)
        del sd
    else:
        up.load_state_dict(torch.load(UF_UP, map_location="cpu",
                                      weights_only=False)["model"], strict=True)
    return up.to(device)


class Pairs:
    def __init__(self, ids):
        self.ids = list(ids); self._mm = {}

    def _get(self, vid):
        if vid not in self._mm:
            self._mm[vid] = (np.load(f"{PAIRS}/h3/{vid}.npy", mmap_mode="r"),
                             np.load(f"{PAIRS}/wan/{vid}.npy", mmap_mode="r"))
        return self._mm[vid]

    def sample(self, rng):
        h, w = self._get(self.ids[rng.randrange(len(self.ids))])
        # Segments are 7 latents wide and must not straddle a 22-frame boundary.
        s = rng.randrange(0, h.shape[1] // SEG) * SEG
        return (np.asarray(h[:, s:s + SEG]), np.asarray(w[:, s:s + SEG]))


def batch(ds, rng, bs, device):
    hs, ws = zip(*(ds.sample(rng) for _ in range(bs)))
    h = torch.from_numpy(np.stack(hs)).to(device).float()
    w = torch.from_numpy(np.stack(ws)).to(device).float()
    return h, w


def run_pipeline(mapper, up, h):
    """Mapper then the frozen upsampler, with main/tail memory reset like training."""
    low = mapper(h)
    return torch.cat([up(low[:, :, :5], parallel=True),
                      up(low[:, :, 5:], parallel=True)], dim=2)


def wan_decode(wan, z):
    """Decode a 7-latent segment the way it was encoded: main 5 then tail 2."""
    outs = []
    for part in (z[:, :, :5], z[:, :, 5:]):
        wan.model.clear_cache()
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            r = wan.model.decode(part.to(torch.bfloat16), wan.scale)
        wan.model.clear_cache()
        outs.append((r.float().clamp(-1, 1) + 1) / 2)
    return torch.cat(outs, dim=2)


@torch.no_grad()
def evaluate(mapper, up, ds, device, n=16, wan=None, n_pixel=3):
    """Latent MSE on every probe, plus pixel PSNR on a few.

    The pixel number is the one that matters: a latent-space improvement that
    does not survive the decoder is worthless, which is exactly how the earlier
    H3-latent attempt went wrong.
    """
    mapper.eval()
    if not hasattr(ds, "_probe"):
        pr = random.Random(20261009)
        ds._probe = [batch(ds, pr, 1, device) for _ in range(n)]
    mse, base, psnr_pred = [], [], []
    for i, (h, w) in enumerate(ds._probe):
        with torch.autocast("npu", dtype=torch.bfloat16):
            p = run_pipeline(mapper, up, h).float()
        mse.append(F.mse_loss(p, w).item())
        base.append(F.mse_loss(w.mean().expand_as(w), w).item())
        if wan is not None and i < n_pixel:
            ch, cw = 24, 42          # latent crop -> 192x336 pixels
            y0 = (p.shape[-2] - ch) // 2
            x0 = (p.shape[-1] - cw) // 2
            pc = p[:, :, :, y0:y0 + ch, x0:x0 + cw].contiguous()
            wc = w[:, :, :, y0:y0 + ch, x0:x0 + cw].contiguous()
            px_p, px_t = wan_decode(wan, pc), wan_decode(wan, wc)
            torch.npu.empty_cache()
            # Against the Wan decode of the true latent: isolates the mapper's
            # error from the VAE round-trip loss, which no mapper can fix.
            psnr_pred.append(float(-10 * torch.log10(F.mse_loss(px_p, px_t) + 1e-12)))
    mapper.train()
    out = dict(val_latent_mse=round(sum(mse) / len(mse), 5),
               trivial_mse=round(sum(base) / len(base), 5))
    if psnr_pred:
        out["val_pixel_psnr_vs_wan_decode"] = round(sum(psnr_pred) / len(psnr_pred), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--train-upsampler", action="store_true")
    ap.add_argument("--init", choices=["joint", "scratch"], default="joint",
                    help="joint: start from the h3_to_wan mapper; scratch: random")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--no-pixel-eval", action="store_true",
                    help="skip decoding during eval (faster, but latent MSE alone "
                         "has already proven an unreliable proxy)")
    args = ap.parse_args()

    from h3wan.model import Mapper
    torch.manual_seed(args.seed)
    device = "npu:0"
    args.out.mkdir(parents=True, exist_ok=True)

    ids = sorted(f[:-4] for f in os.listdir(f"{PAIRS}/wan")
                 if f.endswith(".npy") and not f.endswith(".tmp.npy"))
    ids = [i for i in ids if (PAIRS / "h3" / f"{i}.npy").exists()]
    random.Random(1234).shuffle(ids)
    val, train = Pairs(ids[:VAL_CLIPS]), Pairs(ids[VAL_CLIPS:])
    print(f"train {len(train.ids)} val {len(val.ids)} clips", flush=True)

    mapper = Mapper().to(device)
    if args.init == "joint":
        sd = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)["model"]
        mapper.load_state_dict({k[13:]: v for k, v in sd.items()
                                if k.startswith("joint.mapper.")}, strict=True)
        del sd
        print("mapper initialized from h3_to_wan joint checkpoint", flush=True)
    up = load_upsampler(device, from_joint=True)
    up.requires_grad_(args.train_upsampler)
    if not args.train_upsampler:
        up.eval()
    params = list(mapper.parameters()) + (list(up.parameters()) if args.train_upsampler else [])
    print(f"trainable {sum(p.numel() for p in params)/1e6:.2f}M "
          f"(mapper {sum(p.numel() for p in mapper.parameters())/1e6:.2f}M)", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.03, final_div_factor=50)
    start = 1
    if args.resume and args.resume.exists():
        rb = torch.load(args.resume, map_location="cpu", weights_only=False)
        mapper.load_state_dict(rb["mapper"])
        if args.train_upsampler and "upsampler" in rb:
            up.load_state_dict(rb["upsampler"])
        start = rb["step"] + 1
        for _ in range(rb["step"]):
            sched.step()
        print(f"resumed at step {rb['step']}", flush=True)

    wan = None
    if not args.no_pixel_eval:
        from vendor.wan_vae import WanVAE
        wan = WanVAE(vae_pth="/temp/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
                     dtype=torch.bfloat16, device=device)
        wan.model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
        print("Wan VAE loaded for pixel-domain eval", flush=True)

    rng = random.Random(args.seed)
    (args.out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    log = (args.out / "metrics.jsonl").open("a")
    t0, best = time.time(), 1e9

    for step in range(start, args.steps + 1):
        h, w = batch(train, rng, args.batch, device)
        with torch.autocast("npu", dtype=torch.bfloat16):
            p = run_pipeline(mapper, up, h)
            loss = F.mse_loss(p.float(), w)
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
            ev = evaluate(mapper, up, val, device, wan=wan)
            m = ev["val_latent_mse"]
            rec = dict(step=step, **ev, elapsed=round(time.time() - t0, 1))
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            blob = {"mapper": mapper.state_dict(), "step": step, "args": vars(args)}
            if args.train_upsampler:
                blob["upsampler"] = up.state_dict()
            torch.save(blob, args.out / "latest.pt")
            if m < best:
                best = m
                torch.save({**blob, "val_latent_mse": m}, args.out / "best.pt")

    (args.out / "COMPLETE.json").write_text(json.dumps(
        dict(complete=True, steps=args.steps, best_val_latent_mse=best,
             wall_seconds=time.time() - t0), indent=2))
    print(f"done {(time.time()-t0)/60:.1f} min  best val latent MSE {best:.5f}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
