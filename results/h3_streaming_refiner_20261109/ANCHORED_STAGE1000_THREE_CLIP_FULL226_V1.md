# Anchored stage1000: complete three-clip development evaluation

All three fixed stage1000 candidates completed 67 latents, 226 native-resolution decoded frames and three technically audited videos per clip. Stage1000 means 1000 new anchored updates after the sealed TF2500 baseline, or 3500 updates across two recipes. The adapter remains f16119620e6bcbce0c6da7ce00560dc1733e9d0b3c0d6d4107c38974d43c50a6. No model is accepted.

Ratios below compare candidate error to the same frozen upsampler without a refiner; lower than 1 is better. Metrics use original 1920x1080 lossless RGB, not compressed previews.

| Clip | L1 ratio | LPIPS ratio | Interpretation |
| --- | ---: | ---: | --- |
| WingIt 3586–3811 | 0.946148 | 1.076976 | Better pixel fidelity; worse perceptual error |
| ElephantsDream 13070–13295 | 0.908708 | 0.996803 | Better pixel fidelity; small mean perceptual gain with frame-level regressions |
| WingIt 300–525 | 0.924426 | 1.079811 | Better mean pixel fidelity; worse perceptual and temporal errors |

## WingIt 300–525 full-frame result

All 226 frames are included. L1 improves in 220 frames and worsens in 6; MSE improves in 217 and worsens in 9. LPIPS worsens in all 226. Full-clip temporal-difference L1 and MSE error ratios are 1.034530 and 1.070773, respectively. Tail18 LPIPS ratio is 1.094055. These metrics do not establish the presence or absence of visible flicker.

The original bidirectional reference has L1 ratio 2.958245 and LPIPS ratio 3.543048 against the same no-refiner baseline. Its unchanged upstream equivalence control passed. It uses the same pinned upsampler, no-text/silence conditions, noise and geometry; production prompts have not been validated.

Sampled still-frame inspection at frames 0 and 144 shows changed control-panel markings/controls and vehicle details in the bidirectional output. Candidate and no-refiner preserve the broad GT layout more closely, while both retain soft fine detail. These observations support localized content changes in the dense reference; they do not certify universal absence of hallucination or a full continuous visual review.

## Videos and execution evidence

- [Candidate single playback](anchored_stage1000_WingIt300_native540_preview.mp4)
- [Candidate / original bidirectional / actual GT](anchored_stage1000_WingIt300_dense_gt_preview.mp4)
- [Candidate / no refiner / actual GT](anchored_stage1000_WingIt300_no_refiner_gt_preview.mp4)

Public playback copies are downscaled to 960x540 or 1440x310 and preserve all226 frames at24fps. Original outputs are1920x1080 single and2880x620 comparison videos. Each original and playback copy passes complete frame count, PTS spacing and full decoding checks. The WingIt300 evaluator exited0 and its parent/dispatcher were absent before reusing NPU7. Three locks protect the new independent stage2000 ElephantsDream evaluation.

WingIt300 generation used268 real model forwards with KV cache bounded at6930022400 bytes. Real four-latent causal/cache checks and first4 checkpoint-reload/chunk checks pass at exact0. This is not full67 parity or validation beyond67 latents. 24fps is playback rate, not real-time inference.

## Still pending

Independent stage2000 full evaluation of the same three clips continues; final weights are not automatically promoted over stage1000. Continuous temporal visual review, strict pretrained streaming decoder validation and real33B longer-stream stress remain unpassed. Current quality videos use the frozen official overlap decoder. No extra optimizer update was performed, and no new training is scheduled by this report.
