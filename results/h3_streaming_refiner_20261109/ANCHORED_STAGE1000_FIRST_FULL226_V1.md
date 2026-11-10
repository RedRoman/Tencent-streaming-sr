# Anchored stage1000: first complete226-frame result

The fixed new-stage1000/cumulative3500 candidate completed WingIt_dev_3586_3811 with all67 incremental latents and226 frames at24fps. The checkpoint was not replaced by later training weights. The original trainer continues its unchanged2000-new-update plan. The first evaluator exited0 and its parent/child were reaped before the second development clip started on the released device with allthree locks.

| NativeHR metric, full226 | Candidate / frozen no-refiner | Direction |
|---|---:|---|
| Pixel L1 | 0.946148 | improved5.39% |
| LPIPS | 1.076976 | worse7.70% |
| Pixel MSE | 0.953731 | improved4.63% |
| Adjacent-frame difference L1 | 1.001610 | almost unchanged |
| Temporal difference MSE | 0.999900 | almost unchanged |

All226 frames improve L1 and all226 worsen LPIPS. This is a consistent metric tradeoff on one clip, not a few outlier frames. It supports inspecting perceptual detail and the reconstruction objective, but does not establish an exclusive root cause or generalization. This candidate is not accepted. Frozen thresholds remain unchanged, and the other two development clips remain required.

The67-latent generation used268 forwards; KV memory stayed6930022400 bytes after the second latent. Realfour-latent cache/future-input/future-condition gates and first4 reloaded-checkpoint/chunk replay have maxabs0. Longer-than67 full-model streams remain unverified.

The original three H.264 outputs passed all226 PTS checks,24fps and complete error-checked decode. Candidate:1920x1080; comparisons:2880x620. The visible image uses the prescribed1088→1080 crop. Smaller viewing copies preserve all226 frames; all quantitative metrics use the full-resolution lossless RGB, not H.264 or the previews.

Static inspection covers main frame0 and ablation frames0/72/144/192. Subject poses and scene layout remain recognizable across these four sampled times, with softened contours and small details relative to nativeHR. The candidate retains recognizable subjects and scene geometry in this frame and is much closer to nativeHR than the bidirectional reference with conspicuous texture/shape changes. It is close to the no-refiner draft, with some slightly softer edges. This is not continuous visual review and does not resolve flicker or drift. Continuous review and the strict streaming decoder remain pending. The decoder used for these comparisons is the frozen official overlap decoder.

The original TF2500 negatives, numerical parity failure and aligned-reference diagnosis are preserved. No optimizer/RNG/cache payload is published. Large inference weights are not selected for release.

Videos: [candidate / bidirectional / GT](anchored_stage1000_dense_gt_preview.mp4), [candidate / no-refiner / GT](anchored_stage1000_no_refiner_gt_preview.mp4). These are downscaled playback copies, not native1080 quality evidence.
