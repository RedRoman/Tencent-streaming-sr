# h3 v4b: motion-boundary-weighted loss + aligned50 warm start (2026-11-06)

Status: RUNNING on huawei-modelarts-32193, NPU 7 (Ascend 910B3), 3000 steps.
This snapshot captures work-in-progress; final numbers come after the run and
the standard eval_v4_5s.py protocol.

## Motivation (user-reported failure mode)

v4 shows severe overall degradation, worst at regions with large
adjacent-frame change (motion boundaries). Two levers combined in v4b:

1. `--init aligned50`: warm start from the causal-finetune aligned-past-50
   checkpoint instead of fresh init — targets the generalization gap
   (v4 holdout 27.10 dB vs conv3d_hybrid 29.24 dB on the same 5s protocol;
   v4 train latent_mse 0.0146 vs holdout 0.0956, i.e. the deficit is mostly
   generalization, not fitting).
2. Motion-boundary weighted loss (`--motion-weight 0.5`): masks are derived
   from the detached teacher RGB window (mean-channel abs adjacent-frame
   change >= 0.05); a frame is "motion" if either adjacent pair moves.
   Adds masked L1, masked spatial LPIPS (per-frame to avoid NPU OOM), and
   masked latent temporal-difference L1 on top of the unchanged v2 pixel
   contract (latent MSE + 0.1 temporal + 1.0 L1 + 1.0 LPIPS + 0.333 wavelet).

## Prior measured evidence (expectation setting)

The parallel v5 control-vs-motion experiment
(`experiments/h3_v5_motion_20261106`, completed, gate passed) measured that
naive motion-region loss weighting alone at matched budget (2000 steps,
warm-started from conv3d hybrid best_step_001800) yields only ~2% improvement:
delta_l1 0.01803 vs 0.01843, psnr 25.85 vs 25.65, lpips 0.1283 vs 0.1296;
motion arm wins all 16 test clips but margins are 0.0002-0.0008.
Therefore v4b's primary expected gain is from the aligned50 warm start;
motion weighting is a secondary lever stacked on top.

## Script

`train_v4_motion_pixel.py` — patch of `train_v4_surgery_pixel.py` (v2 pixel
contract preserved exactly when `--motion-weight 0`).

New args:
- `--motion-weight` (default 0.0 = identical to v4)
- `--motion-threshold` (default 0.05)

Key implementation notes (bugs fixed during bring-up on NPU):
- Teacher pixel window is 16 frames, adapter/latent tensors 17 frames; masks
  are stored as (1,1,T,h,w) and adaptively resized (trilinear) to whatever
  target tensor's (T,H,W) they weight, so frame-count mismatch cannot break
  broadcasting.
- `lpips_spatial` (alex, spatial=True, frozen) computed frame-by-frame —
  full-window spatial maps OOM the 61.96 GiB NPU.
- frame_w built by extending the (T-1)-length pair mask with a copy of the
  last pair so T matches the pixel window.

Launch command (on 32193):

```bash
cd /cache/songhua/h3-regenerator-self-forcing-dmd-20261104 && source experiment/env.sh && \
export PYTHONPATH="/home/ma-user/.local/lib/python3.12/site-packages:$PYTHONPATH" && \
export ASCEND_RT_VISIBLE_DEVICES=7 && \
python experiment/train_v4_motion_pixel.py \
  --pairs $EXP_ROOT/upscaler_5s_v4exp/pairs \
  --upstream $EXP_ROOT/upstream \
  --upscaler $EXP_ROOT/assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors \
  --init aligned50 \
  --aligned50-checkpoint $EXP_ROOT/upscaler_causal_finetune/aligned_past_steps50/checkpoint/model.pt \
  --output $EXP_ROOT/runs/v4b_motion_aligned50_20261106 \
  --steps 3000 --motion-weight 0.5
```

`train_log_steps.jsonl` in this directory is a snapshot of the step log
(through step 230); it is not the complete run.

## Measured so far (same machine, shared with another user's 4-NPU job — speed ~5.5-6.5 s/step, ETA ~5h for 3000 steps)

- step 1: loss 1.89454 (l1 0.20194, lpips 0.19198, wav 0.43919, l1_m 0.2397, lp_m 0.21748, td_m 1.85592)
- step 230: loss 0.78560 (l1 0.10974, lpips 0.08500, wav 0.23693, l1_m 0.12971, lp_m 0.09126, td_m 0.72231)
- streaming-consistency pre-gate: max_abs 0.0 (exact-zero causality, passed)

## Next steps after completion

1. Causality post-checks + eval_v4_5s.py protocol vs decoded reference
   (holdout 0380588 + train-10 mean), compare against v4 (27.10) and
   conv3d_hybrid (29.24).
2. Report motion-region metrics (delta_l1-style, as in h3_v5_motion_20261106)
   in addition to global PSNR — motion-boundary artifacts are the stated
   priority.
3. If gains are marginal, candidate next arms (to be discussed before adding):
   higher motion-weight, multi-lag teacher masks, SF2-style (arXiv 2610.35594)
   EMA-teacher representation alignment.

Not pushed: checkpoints, datasets, pair caches. Logs are partial snapshots.
