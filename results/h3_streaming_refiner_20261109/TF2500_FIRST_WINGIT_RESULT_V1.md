# First TF2500 full-sequence result

The trained2500 checkpoint completed the first held-out WingIt clip: all67 latents and all226 frames at24fps. Its native-HR fidelity gate **fails**: L1 is4.737341 times and LPIPS5.318260 times the frozen no-refiner baseline. This is a negative result for this checkpoint on this clip; the remaining fixed2500/1000/2000 cohort continues. No refiner is selected and no self-generated-history or DMD promotion follows.

Training stopped at exactly2500 real optimizer updates with the adapter, AdamW state, saved RNG and next schedule verified. The accepted upsampler remains teacherstep2000 SHA256 `7b9f43c17462fb2d0554645bc7889dc7abdebc052ee9f18f6780624de912e0b5`. This does not implement the previously discussed step0 identity requirement or strict zero-regression guarantee.

## Same-clip comparison

All rows use the same frozen upsampler, no-text/known-silence conditioning, fixed global noise and geometry. Metrics use all226 decoded frames against actual native-HR GT. Lower is better.

| Arm | L1 / no-refiner L1 | LPIPS / no-refiner LPIPS |
| --- | ---: | ---: |
| No refiner | 1.000000 | 1.000000 |
| Untrained causal step0 | 6.208725 | 5.938129 |
| TF200 streaming | 4.620512 | 5.604026 |
| TF2500 streaming | 4.737341 | 5.318260 |
| Original bidirectional refiner | 2.833940 | 3.218384 |

TF2500 LPIPS is lower than TF200 on this clip, while L1 is slightly higher. This is not an overall-improvement claim. The original bidirectional arm also fails the frozen fidelity gate. The full-cohort equal-clip result is pending.

TF2500 absolute metrics: L1 `0.10194905979179703`, LPIPS `0.5666889469180487`, PSNR `16.68194943952265` dB, adjacent-frame delta L1 `0.03462661929228387`. No-refiner values are L1 `0.02152031323814814`, LPIPS `0.10655533792698278`, PSNR `27.72829259995797` dB, delta L1 `0.01659603923696358`.

## Correctness and scope

The actual trained full33B two-latent cache, future-video, future-condition and save/reload checks report exact0 error. The independent first-four-latent chunk replay is also exact0. Full67 inference performs268 model forwards: three denoising forwards plus one finalized clean-cache write per latent. Cache storage reaches6,930,022,400 bytes after the first latent and remains bounded through67. This does not validate the full model beyond67 latents.

The pipeline latent geometry is `[1,24,67,68,120]`; the saved unbatched array is `[24,67,68,120]`. The visible pixel array is `[226,1080,1920,3]`, using the fixed1920×1088 decode and crop y=4:1084. Pixel decoding uses the frozen official overlap decoder, which remains provisional for strict streaming. Production text conditioning and the strict decoder remain unvalidated.

The CPU encoder produced a native1080p video, a main three-column streaming-TF2500 / original-bidirectional / actual-GT comparison, a compact version, and a separate streaming-TF2500 / no-refiner / actual-GT ablation. Every video passed226-frame count,24fps, all-frame timestamp and full-decode checks. Technical video validity is separate from visual quality and continuous visual review.

The completed latent and pixel receipts, summary and video audit preserve source payload SHA256 values. Private cache arrays, optimizer/RNG state and weights are not included. The remaining evaluation uses the existing single launcher under all three NPU locks; no duplicate job or monitor was added.

## Evidence

- `TF2500_first_WingIt_summary.json`: full-frame metrics, payload lineage and gate scope.
- `TF2500_first_WingIt_latent_result.json`: complete67 latent result and chunk/cache checks.
- `TF2500_first_WingIt_pixels_result.json`: full226 metrics and per-frame results.
- `TF2500_video_audit.json`: encoded-file SHA256 and complete technical audits.
- `TF2500_second_clip_snapshot_v1.json`:04:00:10UTC second-clip progress and resource ownership.

Original versus public file SHA256 values are recorded in `evidence_sha_mapping.json`; deployment paths are replaced with placeholders.
