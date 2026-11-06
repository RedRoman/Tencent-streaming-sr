#!/usr/bin/env python
"""v4: surgery-causalized Open-H3 upscaler trained under the v2 pixel-loss contract.

Single question: pretrained-init (aligned-past surgery on the released upscaler)
vs from-scratch distillation (CausalStudentUpscaler), under an identical loss
contract and identical expanded data. Architecture = method-1's causalized
original upscaler (frozen teacher is the untouched bidirectional original).

Losses and decode-window logic are imported verbatim from
experiment/train_student_pixel_v2.py so the contract is bit-identical to v2.
Init options:
  --init fresh        apply initialize_aligned_past_ to the released upscaler
  --init aligned50    reload upscaler_causal_finetune/aligned_past_steps50
                      checkpoint (weights already transformed; NEVER re-surgery)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiment.data_utils import empty_cache, select_device, tensor_load, video_tensor
from experiment.train_upscaler_trial import atomic_json, atomic_torch_save, metrics, read_pairs, wrapper_forward
from experiment.prepare_data import load_author_upscaler
from experiment.causal_upscaler import CausalUpscalerAdapter, initialize_aligned_past_

# v2 loss contract: pixel-loss stack below is verified line-by-line against
# train_student_pixel_v2.py:184-206 (L1 on raw unclipped pixels, LPIPS clamped,
# 1-level DWT on all bands, random T=5 window).


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, default=Path("upscaler_5s_v4exp/pairs"))
    p.add_argument("--upstream", type=Path, default=Path(__file__).resolve().parent.parent / "upstream")
    p.add_argument("--upscaler", type=Path, default=Path(
        "assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"))
    p.add_argument("--init", choices=("fresh", "aligned50"), default="fresh")
    p.add_argument("--aligned50-checkpoint", type=Path,
                   default=Path("upscaler_causal_finetune/aligned_past_steps50/checkpoint/model.pt"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--temporal-weight", type=float, default=0.1)
    p.add_argument("--motion-weight", type=float, default=0.0,
                   help="extra motion-boundary-weighted loss weight (0 = identical to v4)")
    p.add_argument("--motion-threshold", type=float, default=0.05,
                   help="mean-channel abs adjacent RGB change marking a moving edge")
    p.add_argument("--resume", type=Path, default=None,
                   help="checkpoint .pt to warm-start weights from; must match adapter_config")
    p.add_argument("--start-step", type=int, default=0,
                   help="resume step counter (logs/checkpoints continue from here)")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--seed", type=int, default=20261104)
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--save-every", type=int, default=500)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    out = args.output
    if out.exists():
        raise FileExistsError(f"{out}: use a fresh output directory")
    (out / "checkpoint").mkdir(parents=True)

    pairs = read_pairs(args.pairs / "train.jsonl", "train")
    hold_pairs = read_pairs(args.pairs / "holdout.jsonl", "holdout")
    print(f"train {len(pairs)} holdout {len(hold_pairs)}", flush=True)

    # --- surgery model (trainable) ---
    adapter_config = {
        'version': 1, 'initialization': 'aligned_past',
        'temporal_padding': 'left_only_zero', 'groupnorm': 'per_frame',
        'normalization': 'D(U(N(native_normalized_H3_latent)))',
        'spatial_scale': 2., 'temporal_chunking': False, 'dtype': 'float32',
        'saved_weights_already_initialized': True,
        'reload_must_not_reinitialize_weights': True}
    node = load_author_upscaler(args.upstream, args.upscaler, device)
    model = node.load_model('upscaler', device, 'fp32')
    if args.init == "fresh":
        init_report = initialize_aligned_past_(model)
        print(json.dumps({"init": "fresh", "report_kind": init_report.get('kind', 'aligned_past')}), flush=True)
    else:
        saved = tensor_load(args.aligned50_checkpoint)
        if saved['adapter_config'] != adapter_config:
            raise ValueError('aligned50 checkpoint adapter config mismatch')
        model.load_state_dict(saved['model_state_dict'], strict=True)
        model._aligned_past_initialized = True
        del saved
        print(json.dumps({"init": "aligned50", "source": str(args.aligned50_checkpoint)}), flush=True)
    if args.resume is not None:
        saved = tensor_load(args.resume)
        if saved.get('adapter_config', adapter_config) != adapter_config:
            raise ValueError('resume checkpoint adapter config mismatch')
        model.load_state_dict(saved['model_state_dict'], strict=True)
        model._aligned_past_initialized = True
        del saved
        print(json.dumps({"resume": str(args.resume)}), flush=True)
    adapter = CausalUpscalerAdapter(node, model)
    # 193M fp32 full-clip forward: checkpoint activations to fit 61GB HBM
    # alongside the frozen fp32 teacher and the VAE decode-with-grad path.
    try:
        from experiment.train_upscaler_trial import enable_activation_checkpointing
        enable_activation_checkpointing(model)
        print(json.dumps({"activation_checkpointing": True}), flush=True)
    except ImportError:
        print(json.dumps({"activation_checkpointing": False, "reason": "import failed"}), flush=True)
    # Original upscaler recipe: dropout stays OFF for the entire run
    # (train_upscaler_trial: "eval mode keeps dropout off"). Never call
    # model.train(): ResBlocks carry dropout=0.1 and this run must be
    # deterministic and regime-consistent end to end.
    model.eval().requires_grad_(True)
    parameters = [v for v in model.parameters() if v.requires_grad]
    n_params = sum(v.numel() for v in parameters)
    print(json.dumps({"trainable_params": n_params}), flush=True)
    opt = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)

    # --- frozen teacher: untouched bidirectional original ---
    tnode = load_author_upscaler(args.upstream, args.upscaler, device)
    teacher = tnode.load_model("upscaler", device, "fp32")
    for p_ in teacher.parameters():
        p_.requires_grad_(False)

    def teacher_out(draft, target):
        with torch.no_grad():
            return wrapper_forward(tnode, teacher, draft, tuple(target))

    import lpips as lpips_lib
    import pytorch_wavelets

    # --- v2 pixel-loss stack (same construction as v2 main) ---
    import sys as _sys
    _sys.path.insert(0, "/home/ma-user/workspace/zzjrua/vdn-minimax-h3/diffusers/src")
    from diffusers import AutoencoderKLMiniMaxH3  # noqa: F401
    from data_utils import latent_stats, load_video_vae, pixel_stats
    vae = load_video_vae("/temp/h3_training_assets_20260921/VDN-MiniMax-H3/h3-base", device)
    vae.disable_tiling()
    for p_ in vae.parameters():
        p_.requires_grad_(False)
    lmean, lstd = latent_stats(vae, device)
    pmean, pstd = pixel_stats(device)
    lpips_fn = lpips_lib.LPIPS(net='alex').to(device)
    lpips_fn.eval(); lpips_fn.requires_grad_(False)
    lpips_spatial = lpips_lib.LPIPS(net='alex', spatial=True).to(device)
    lpips_spatial.eval(); lpips_spatial.requires_grad_(False)
    dwt = pytorch_wavelets.DWTForward(J=1).to(device)

    def decode_latent_train(lat):
        return vae.decode(lat * lstd + lmean, return_dict=False)[0]

    # pixel_losses verified line-by-line against train_student_pixel_v2.py:184-206
    loss_rng = random.Random(args.seed)

    def pixel_losses(pred_lat, te_lat, td_lat=None, td_shape=None):
        T = pred_lat.shape[2]
        start = loss_rng.randint(0, T - 5)
        pw = pred_lat[:, :, start:start + 5]
        tw = te_lat[:, :, start:start + 5]
        px = decode_latent_train(pw)
        pt = decode_latent_train(tw).detach()
        l1 = (px - pt).abs().mean()

        def to_pm1(v):
            return (((v * pstd + pmean).clamp(0, 1) - 0.5) / 0.5)[0].permute(1, 0, 2, 3)
        lp = lpips_fn(to_pm1(px), to_pm1(pt)).mean()
        px4 = px[0].permute(1, 0, 2, 3)
        pt4 = pt[0].permute(1, 0, 2, 3)
        yl, yh = dwt(px4)
        ylt, yht = dwt(pt4)
        wav = sum(((a - b).abs().mean() for a, b in zip([yl] + list(yh), [ylt] + list(yht))))
        # Motion-boundary weight from the detached teacher window: a pixel pair
        # whose mean-channel abs RGB change >= threshold is a moving edge; a
        # frame counts as motion if either adjacent pair touches it (same
        # construction as the v5 motion arm, but from teacher output since v4
        # has no GT pixels in the loop).
        import torch.nn.functional as F
        with torch.no_grad():
            # pt: (1,3,T,H,W). Mean-channel abs change over channels -> (1,T-1,H,W).
            pair_mask = ((pt[:, :, 1:] - pt[:, :, :-1]).abs().mean(dim=1)
                         >= args.motion_threshold).float()
            # frame_w: (1,T,H,W); frame t is motion if either adjacent pair moves.
            frame_w = torch.cat([pair_mask[:, :1],
                                 torch.maximum(pair_mask[:, :-1], pair_mask[:, 1:]),
                                 pair_mask[:, -1:]],
                                dim=1)
        if args.motion_weight > 0:
            # All masks live in (1,1,T,h,w); resize adaptively to whatever the
            # target tensor's (T,H,W) is, so frame-count mismatches between
            # pixel teacher / pred / latent tensors cannot break broadcasting.
            fw5 = frame_w.unsqueeze(1)                               # (1,1,Tm,h,w)
            w_px = F.interpolate(fw5, size=px.shape[-3:], mode='trilinear')
            l1_m = ((px - pt).abs() * w_px).sum() / (w_px.sum() * 3 + 1e-6)
            # spatial LPIPS frame-by-frame (full-window spatial maps OOM the NPU)
            maps = torch.cat([lpips_spatial(to_pm1(px[:, :, t:t+1]),
                                            to_pm1(pt[:, :, t:t+1]))
                              for t in range(px.shape[2])], dim=0)   # (T,1,h,w)
            maps = maps.permute(1, 0, 2, 3)[None]                    # -> (1,1,T,h,w)
            w_s = F.interpolate(fw5, size=maps.shape[-3:], mode='trilinear')
            lp_m = (maps * w_s).sum() / (w_s.sum() + 1e-6)
            w_td = F.interpolate(fw5, size=td_lat.shape[-3:], mode='trilinear')
            td_m = (td_lat.abs() * w_td).sum() / (w_td.sum() * 3 + 1e-6)
        else:
            l1_m = lp_m = td_m = torch.zeros((), device=px.device)
        return l1, lp, wav, l1_m, lp_m, td_m

    def load_samples(prs):
        res = []
        for pr in prs:
            d = tensor_load(pr["path"])
            res.append({"id": pr["sample_id"], "draft": video_tensor(d["draft"]).float(),
                        "reference": video_tensor(d["reference"]).float()})
        return res

    samples = load_samples(pairs)
    hold_samples = load_samples(hold_pairs)

    # streaming-consistency gate must pass before any training step
    with torch.no_grad():
        s0 = hold_samples[0]["draft"][None].to(device)
        full = adapter.native_forward(s0, tuple(hold_samples[0]["reference"].shape[-3:]))
        caches, seqs = [], []
        adapter.reset_stream()
        for i in range(s0.shape[2]):
            o = adapter.stream_step(s0[:, :, i:i + 1], (s0.shape[3] * 2, s0.shape[4] * 2))
            seqs.append(o)
        seq = torch.cat(seqs, dim=2)
        cons = float((seq - full).abs().max())
    print(json.dumps({"streaming_consistency_max_abs": cons}), flush=True)
    if not cons < 1e-2:
        raise RuntimeError(f'streaming consistency failed pre-gate: {cons}')

    log = []
    t0 = time.perf_counter()
    assert args.start_step <= args.steps
    for step in range(args.start_step + 1, args.steps + 1):
        s = samples[(step - 1) % len(samples)]
        draft = s["draft"][None].to(device)
        te = teacher_out(draft, tuple(s["reference"].shape[-3:]))
        pred = adapter.native_forward(draft, tuple(s["reference"].shape[-3:]))
        loss = (pred - te).square().mean()
        td = (pred[:, :, 1:] - pred[:, :, :-1]) - (te[:, :, 1:] - te[:, :, :-1])
        loss = loss + args.temporal_weight * td.square().mean()
        l1, lp, wav, l1_m, lp_m, td_m = pixel_losses(pred, te, td, td.shape[-2:])
        loss = loss + 1.0 * l1 + 1.0 * lp + 0.333 * wav
        loss = loss + args.motion_weight * (l1_m + lp_m + td_m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        opt.step()
        if step % 10 == 0 or step == args.start_step + 1:
            rec = {"step": step, "loss": round(float(loss), 5),
                   "l1": round(float(l1), 5), "lpips": round(float(lp), 5),
                   "wav": round(float(wav), 5),
                   "l1_m": round(float(l1_m), 5), "lp_m": round(float(lp_m), 5),
                   "td_m": round(float(td_m), 5),
                   "seconds": round(time.perf_counter() - t0, 1)}
            log.append(rec)
            print(json.dumps(rec), flush=True)
        if step % args.val_every == 0 or step == args.steps:
            with torch.no_grad():
                for s2 in hold_samples:
                    d2 = s2["draft"][None].to(device)
                    te2 = teacher_out(d2, tuple(s2["reference"].shape[-3:]))
                    m = metrics(adapter.native_forward(d2, tuple(s2["reference"].shape[-3:]))[0],
                                te2[0], args.temporal_weight)
                    print(json.dumps({"val_step": step, "id": s2["id"],
                                      "latent_mse": round(m["latent_mse"], 5)}), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            atomic_torch_save(out / "checkpoint" / "v4.pt",
                              {"model_state_dict": model.state_dict(), "step": step,
                               "adapter_config": adapter_config, "init": args.init,
                               "args": {k: str(v) for k, v in vars(args).items()}})

    # causality gates on holdout: future_cut + prefix_cut + streaming
    model.eval()
    checks = {}
    with torch.no_grad():
        s = hold_samples[0]
        draft = s["draft"][None].to(device)
        full = adapter.native_forward(draft, tuple(s["reference"].shape[-3:]))
        for cut in (1, 13, 25):
            changed = draft.clone()
            changed[:, :, cut:] = 3 * torch.randn(changed[:, :, cut:].shape,
                generator=torch.Generator().manual_seed(20261104 + cut)).to(device)
            actual = adapter.native_forward(changed, tuple(s["reference"].shape[-3:]))
            checks[f"future_cut{cut}_max_abs"] = float((actual[:, :, :cut] - full[:, :, :cut]).abs().max())
            prefix = adapter.native_forward(draft[:, :, :cut].contiguous(), tuple(s["reference"].shape[-3:]))
            checks[f"prefix_cut{cut}_max_abs"] = float((prefix - full[:, :, :cut]).abs().max())
        adapter.reset_stream()
        seqs = []
        for i in range(draft.shape[2]):
            seqs.append(adapter.stream_step(draft[:, :, i:i + 1], (draft.shape[3] * 2, draft.shape[4] * 2)))
        checks["streaming_max_abs"] = float((torch.cat(seqs, dim=2) - full).abs().max())
    checks["passed"] = all(v < 1e-2 for k, v in checks.items() if isinstance(v, float))
    print(json.dumps({"causality": checks}), flush=True)

    final = {}
    with torch.no_grad():
        for s2 in samples + hold_samples:
            d2 = s2["draft"][None].to(device)
            te2 = teacher_out(d2, tuple(s2["reference"].shape[-3:]))
            final[s2["id"]] = metrics(
                adapter.native_forward(d2, tuple(s2["reference"].shape[-3:]))[0], te2[0],
                args.temporal_weight)
            print(json.dumps({"final_vs_teacher": s2["id"],
                              "latent_mse": round(final[s2["id"]]["latent_mse"], 5)}), flush=True)

    atomic_torch_save(out / "checkpoint" / "v4_final.pt",
                      {"model_state_dict": model.state_dict(), "step": args.steps,
                       "adapter_config": adapter_config, "init": args.init,
                       "consistency_max_abs": checks["streaming_max_abs"]})
    atomic_json(out / "train_summary.json", {
        "steps": args.steps, "params": n_params, "init": args.init,
        "train_clips": len(samples), "holdout_clips": len(hold_samples),
        "causality": checks, "final_vs_teacher": final,
        "log_tail": log[-10:], "passed": checks["passed"]})
    empty_cache(device)


if __name__ == "__main__":
    main()
