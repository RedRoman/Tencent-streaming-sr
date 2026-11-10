# Anchored stage1000: ElephantsDream full226 result

Exact anchoredstage1000/cumulative3500 completed67 latents,226 frames and allthree videos for ElephantsDream_dev_13070_13295. The evaluator naturally exited0, its child was reaped and both parent/dispatcher were absent before NPU4 was reused. The original stage1000 checkpoint and frozen upsampler are unchanged.

| NativeHR metric, full226 | Candidate / no-refiner | Original bidirectional / no-refiner |
|---|---:|---:|
| Pixel L1 | 0.908707964 | 2.570817796 |
| LPIPS | 0.996802589 | 2.491357221 |

The candidate improves mean L1 by9.13% and mean LPIPS by0.32%. L1 and MSE improve on all226 frames; LPIPS improves on142 and regresses on84. The final18-frame LPIPS ratio is1.004205914, so the aggregate improvement does not imply every frame improves. Candidate temporal-difference L1 and MSE ratios are0.978679523 and0.935147686. These are lossless1080p RGB metrics, not measurements of the compressed playback files.

The three original videos and three lightweight playback copies each passed226-frame,24fps, exact timestamp sequence and complete-file decoding checks. The local playback copies additionally matched all file SHA256 values. The native single video is1920x1080; its playback derivative is960x540. Main and ablation originals are2880x620 and their playback derivatives1440x310. Both preserve the entire9.416667-second clip, rather than sampled frames.

Static inspection covers zero-based frames0 and144 only. The original bidirectional reference adds denser tube surface grids/texture and changes some wall markings relative to GT. Candidate and no-refiner retain closer large-scale hand/tube/wall structure; some candidate texture remains softer. This supports the previously reported content-deviation concern on another clip, but is not proof of hallucination absence or temporal stability throughout the video. Continuous visual review remains pending.

The previous WingIt3586 stage1000 result remains mixed (L1ratio0.946147846, LPIPSratio1.076976347). Stage1000 has completed2/3 development videosets, not the entire cohort. Stage2000 WingIt3586 has now launched independently and passed the real33B four-latent same-history/future-input/future-condition gates at exact0; its full67/226 outputs remain pending. No model is selected or accepted. Strict streaming decoder and real33B longer-than67 validation remain separate outstanding gates. No additional training was started.

Playback files: [candidate](anchored_stage1000_ElephantsDream_native540_preview.mp4), [candidate / original bidirectional / realGT](anchored_stage1000_ElephantsDream_dense_gt_preview.mp4), [candidate / no-refiner / realGT](anchored_stage1000_ElephantsDream_no_refiner_gt_preview.mp4).
