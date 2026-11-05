#!/usr/bin/env python
"""Method-2: independent causal student distilled from the frozen bidirectional
Open-H3 upscaler teacher. My own design (distinct from the earlier aligned-past
weight surgery in experiment/causal_upscaler.py).

Student: causal memory blocks (per-frame 2D convs; temporal fusion only via
current+past concat) + PixelShuffle 2x head. Parallel forward for training,
sequential cached forward for streaming; both must agree.

Objective (v2 pixel contract, adapted from songhua's frozen_mapper672 run):
latent MSE + decoded-pixel L1 + LPIPS(alex) + wavelet high-freq (lh/hl/hh) +
temporal-difference MSE. Pixel losses flow gradients through the frozen H3 VAE
decoder back into the student. Decode cost: full 37-frame decode ~0.97s+grad;
too heavy per step -> sample a random 5-latent-frame window per step (17 decoded
frames), decode it with tiling disabled (T<5 hits a torch.cat bug).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
import lpips as lpips_lib
import pytorch_wavelets

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiment.data_utils import empty_cache, select_device, tensor_load, video_tensor
from experiment.train_upscaler_trial import atomic_json, atomic_torch_save, metrics, read_pairs, wrapper_forward
from experiment.prepare_data import load_author_upscaler


def conv2d(cin, cout, k=3):
    return nn.Conv2d(cin, cout, k, padding=k // 2)


class CausalMemBlock(nn.Module):
    """Current-frame spatial convs + past-frame memory concat. Causal in T."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            conv2d(ch * 2, ch), nn.SiLU(),
            conv2d(ch, ch), nn.SiLU(),
            conv2d(ch, ch),
        )
        self.act = nn.SiLU()

    def forward(self, x, past):        # x, past: (B,C,H,W)
        return self.act(self.conv(torch.cat([x, past], dim=1)) + x)


class CausalStudentUpscaler(nn.Module):
    """(B,24,T,h,w) -> (B,24,T,2h,2w), strictly causal in T.

    Residual: each output frame adds its own pre-PixelShuffle draft frame
    (upsampled by the head's implicit interpolation? no — we add the draft
    frame bilinearly upsampled 2x), so a zero-initialized final conv makes the
    student start as a plain bilinear 2x upsampler.
    """

    def __init__(self, ch_in=24, ch=256, blocks=6, scale=2):
        super().__init__()
        self.scale = scale
        self.head = nn.Sequential(
            conv2d(ch_in, ch * scale * scale), nn.SiLU(), nn.PixelShuffle(scale), conv2d(ch, ch))
        self.blocks = nn.ModuleList([CausalMemBlock(ch) for _ in range(blocks)])
        self.norm = nn.GroupNorm(32, ch)
        self.final = conv2d(ch, ch_in)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def _frame(self, feat, resid, caches):
        # feat: head features (B,ch,2h,2w); resid: 24ch draft bilinearly upsampled 2x
        h = feat
        new_caches = []
        for i, block in enumerate(self.blocks):
            past = torch.zeros_like(feat) if caches is None else caches[i]
            h = block(h, past)
            new_caches.append(h)
        out = self.final(self.norm(h)) + resid
        return out, new_caches

    def _prep(self, x):
        b, c, t, h, w = x.shape
        xf = self.head(rearrange(x, "b c t h w -> (b t) c h w"))
        _, chn, hh, ww = xf.shape
        resid = F.interpolate(
            rearrange(x, "b c t h w -> (b t) c h w"), size=(hh, ww),
            mode="bilinear", align_corners=False)
        return x, xf.reshape(b, t, chn, hh, ww), resid.reshape(b, t, c, hh, ww)

    def forward_parallel(self, x):
        x, feats, rids = self._prep(x)
        b, t = feats.shape[:2]
        outs, caches = [], None
        for i in range(t):
            o, caches = self._frame(feats[:, i], rids[:, i], caches)
            outs.append(o)
        return rearrange(torch.stack(outs, dim=1), "b t c h w -> b c t h w")

    def forward(self, x):
        return self.forward_parallel(x)

    def forward_sequential(self, x, caches=None):
        """x: (B,C,t,h,w) streaming chunk (usually t=1). Returns (out, caches)."""
        x, feats, rids = self._prep(x)
        t = feats.shape[1]
        outs = []
        for i in range(t):
            o, caches = self._frame(feats[:, i], rids[:, i], caches)
            outs.append(o)
        return rearrange(torch.stack(outs, dim=1), "b t c h w -> b c t h w"), caches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=Path, default=Path("upscaler_5s/pairs"))
    p.add_argument("--upstream", type=Path, default=Path(__file__).resolve().parent.parent / "upstream")
    p.add_argument("--upscaler", type=Path, default=Path(
        "assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--temporal-weight", type=float, default=0.1)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--seed", type=int, default=20261104)
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--save-every", type=int, default=500)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    out = args.output
    (out / "checkpoint").mkdir(parents=True, exist_ok=True)

    pairs = read_pairs(args.pairs / "train.jsonl", "train")
    hold_pairs = read_pairs(args.pairs / "holdout.jsonl", "holdout")
    print(f"train {len(pairs)} holdout {len(hold_pairs)}", flush=True)

    # Frozen teacher: the author's bidirectional upscaler via its ComfyUI node.
    node = load_author_upscaler(args.upstream, args.upscaler, device)
    teacher = node.load_model("upscaler", device, "fp32")
    for p_ in teacher.parameters():
        p_.requires_grad_(False)

    def teacher_out(draft, target):
        # wrapper_forward convention: D(U(N(z)))
        with torch.no_grad():
            return wrapper_forward(node, teacher, draft, tuple(target))

    student = CausalStudentUpscaler().to(device=device, dtype=torch.float32)
    n_params = sum(p_.numel() for p_ in student.parameters())
    print(f"student params {n_params:,}", flush=True)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)

    # --- v2 pixel-loss stack ---
    import sys as _sys
    _sys.path.insert(0, "/home/ma-user/workspace/zzjrua/vdn-minimax-h3/diffusers/src")
    from diffusers import AutoencoderKLMiniMaxH3
    from huggingface_hub import hf_hub_download  # noqa: F401 (vendored hub path)
    from data_utils import latent_stats, pixel_stats, load_video_vae  # load_video_vae handles vendored import order
    vae = load_video_vae("/temp/h3_training_assets_20260921/VDN-MiniMax-H3/h3-base", device)
    vae.disable_tiling()          # T<5 chunk decode hits torch.cat bug with tiling
    for p_ in vae.parameters():
        p_.requires_grad_(False)
    lmean, lstd = latent_stats(vae, device)
    pmean, pstd = pixel_stats(device)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    for p_ in lpips_fn.parameters():
        p_.requires_grad_(False)
    dwt = pytorch_wavelets.DWTForward(J=1).to(device)

    def decode_latent_train(lat):
        # lat: (1,24,T,24,42) normalized like reference latents; returns decoded pixels (1,3,Td,384,672)
        return vae.decode(lat * lstd + lmean, return_dict=False)[0]

    def pixel_losses(pred_lat, te_lat, rng):
        # sample a random T=5 window; decode both; pixel-space losses with grad
        T = pred_lat.shape[2]
        assert T >= 5
        start = rng.randint(0, T - 5 + 1)
        pw = pred_lat[:, :, start:start + 5]
        tw = te_lat[:, :, start:start + 5]
        px = decode_latent_train(pw)                    # (1,3,17,H,W) raw pixels
        pt = decode_latent_train(tw).detach()
        l1 = (px - pt).abs().mean()
        # map raw pixels -> [0,1] -> [-1,1] for LPIPS (songhua contract: unclipped
        # raw for L1/wavelet, but LPIPS needs bounded input; clamp only LPIPS branch)
        def to_pm1(v):
            # (1,3,T,H,W) -> (T,3,H,W): LPIPS is per-image
            return (((v * pstd + pmean).clamp(0, 1) - 0.5) / 0.5)[0].permute(1, 0, 2, 3)
        lp = lpips_fn(to_pm1(px), to_pm1(pt)).mean()
        px4 = px[0].permute(1, 0, 2, 3)      # (17,3,H,W) for DWT
        pt4 = pt[0].permute(1, 0, 2, 3)
        yl, yh = dwt(px4)
        ylt, yht = dwt(pt4)
        wav = sum(((a - b).abs().mean() for a, b in
                   zip([yl] + list(yh), [ylt] + list(yht))))
        return l1, lp, wav

    loss_rng = __import__("random").Random(args.seed)

    def load_samples(prs):
        res = []
        for pr in prs:
            d = tensor_load(pr["path"])
            res.append({"id": pr["sample_id"], "draft": video_tensor(d["draft"]).float(),
                        "reference": video_tensor(d["reference"]).float()})
        return res

    samples = load_samples(pairs)
    hold_samples = load_samples(hold_pairs)

    log = []
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        s = samples[(step - 1) % len(samples)]
        draft = s["draft"][None].to(device)
        target = tuple(s["reference"].shape[-3:])
        te = teacher_out(draft, target)
        pred = student(draft)
        loss = (pred - te).square().mean()
        td = (pred[:, :, 1:] - pred[:, :, :-1]) - (te[:, :, 1:] - te[:, :, :-1])
        loss = loss + args.temporal_weight * td.square().mean()
        l1, lp, wav = pixel_losses(pred, te, loss_rng)
        loss = loss + 1.0 * l1 + 1.0 * lp + 0.333 * wav
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if step % 10 == 0:
            rec = {"step": step, "loss": round(float(loss), 5),
                   "l1": round(float(l1), 5), "lpips": round(float(lp), 5),
                   "wav": round(float(wav), 5),
                   "seconds": round(time.perf_counter() - t0, 1)}
            log.append(rec)
            print(json.dumps(rec), flush=True)
        if step % args.val_every == 0 or step == args.steps:
            student.eval()
            with torch.no_grad():
                for s2 in hold_samples:
                    d2 = s2["draft"][None].to(device)
                    te2 = teacher_out(d2, tuple(s2["reference"].shape[-3:]))
                    m = metrics(student(d2)[0], te2[0], args.temporal_weight)
                    print(json.dumps({"val_step": step, "id": s2["id"],
                                      "latent_mse": round(m["latent_mse"], 5)}), flush=True)
            student.train()
        if step % args.save_every == 0 or step == args.steps:
            atomic_torch_save(out / "checkpoint" / "student.pt",
                              {"model": student.state_dict(), "step": step, "args": vars(args)})

    # parallel vs sequential streaming consistency on holdout clip 0
    student.eval()
    s = hold_samples[0]
    draft = s["draft"][None].to(device)
    with torch.no_grad():
        full = student(draft)
        caches, seqs = None, []
        for i in range(draft.shape[2]):
            o, caches = student.forward_sequential(draft[:, :, i:i + 1], caches)
            seqs.append(o)
        seq = torch.cat(seqs, dim=2)
    consistency = float((seq - full).abs().max())
    print(json.dumps({"parallel_sequential_max_abs": consistency}), flush=True)

    # final teacher-relative metrics on all clips
    final = {}
    with torch.no_grad():
        for s2 in samples + hold_samples:
            d2 = s2["draft"][None].to(device)
            te2 = teacher_out(d2, tuple(s2["reference"].shape[-3:]))
            final[s2["id"]] = metrics(student(d2)[0], te2[0], args.temporal_weight)
            print(json.dumps({"final_vs_teacher": s2["id"],
                              "latent_mse": round(final[s2["id"]]["latent_mse"], 5)}), flush=True)

    atomic_torch_save(out / "checkpoint" / "student_final.pt",
                      {"model": student.state_dict(), "step": args.steps,
                       "consistency_max_abs": consistency})
    atomic_json(out / "train_summary.json", {
        "steps": args.steps, "params": n_params,
        "consistency_max_abs": consistency, "final_vs_teacher": final,
        "log_tail": log[-10:], "passed": consistency < 1e-2})
    empty_cache(device)


if __name__ == "__main__":
    main()
