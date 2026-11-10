# Strict pretrained decoder: technical checks pass, fidelity gate fails

This independent ablation decodes the same frozen teacherstep2000 upsampler output using either strict incremental decoding or the official overlap decoder. Neither arm applies a refiner. Metrics use all226 native1080 RGB frames and trueHR. Existing six refiner video sets remain unchanged.

| Metric | Strict / official error | Change |
|---|---:|---:|
| pixel_l1 | 1.056388590 | +5.639% |
| mse | 1.104118796 | +10.412% |
| lpips | 1.086851222 | +8.685% |
| delta_l1 | 1.088515253 | +8.852% |
| temporal_difference_mse | 1.119399327 | +11.940% |

The unchanged first-clip threshold is at most1.02 for both L1 andLPIPS; both fail. No strict decoder or refiner model is accepted.

Per-frame comparisons (improved/worsened): L1 3/223; MSE 8/218; LPIPS 8/218.

Technical evidence: the full pretrained36-layer VAE matched the pinned upstream incremental reference exactly on nine16x16 latent crops. Full68x120 first-eight reset/chunk replay and future-prefix tests were exact0; changing the eighth input changed its own output. The137-latent looped-input stress completed464frames with constant63,594,823,680-byte CPU KV, at most7 retained inputs and190 slot forwards including53 bounded past-slot refreshes. This is not full-prefix recomputation, nor a realtime result.

The unique job naturally exited0, its child was reaped, and both original processes are absent. Successful process completion does not override the failed quality gate. Frozen sources, precision, compensation=false and thresholds are preserved. Reset replay is not serialized decoder-cache restoration. The137 inputs repeat67 authorized source latents and do not establish quality on a real long video.

Continuous visual review is pending. Video encoding is a separate CPU-only packaging step; no optimizer updates or accelerator forwards are added. The separate real33B refiner long-stream job continues under its original process and locks. Its result cannot substitute for decoder fidelity.

The full226 diagnostic comparison is [strict decoder / official overlap decoder / nativeHR](strict_official_gt_full226.mp4). Neither decoder arm applies a refiner. Local original1080 output and comparison hashes,226timestamps and full decode are verified. Four stills (0,72,144,192) retain broad character/layout structure but show tonal differences, notably frame0, and soft detail; this limited inspection does not assess flicker or drift continuously.
