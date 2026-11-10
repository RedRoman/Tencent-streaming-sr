# Anchored stage2000 WingIt3586: full226 comparison

The exact final stage2000 checkpoint (2000 new anchored updates;4500 cumulative across two recipes) completed67latents,226frames and three technically audited videos, followed by naturalexit0 and process reap. The candidate is not accepted. Original first4 results did not determine this full-frame conclusion.

| Error ratio vs same no-refiner baseline | Stage1000 | Stage2000 |
| --- | ---: | ---: |
| Pixel L1 | 0.946148 | 0.971926 |
| LPIPS | 1.076976 | 1.070931 |

Stage2000 has2.7246% more L1 error than stage1000 and0.5614% less LPIPS error. Relative to no-refiner, L1 improves215/226frames and worsens11; LPIPS worsens all226frames. MSE ratio is0.960118, temporal-difference L1ratio1.000821 and temporal-difference MSEratio1.004534. These mixed measurements do not promote the final checkpoint overstage1000.

- [Candidate playback](anchored_stage2000_WingIt3586_native540_preview.mp4)
- [Candidate / original bidirectional / trueGT](anchored_stage2000_WingIt3586_dense_gt_preview.mp4)
- [Candidate / no-refiner / trueGT](anchored_stage2000_WingIt3586_no_refiner_gt_preview.mp4)

Playback copies preserve226frames/24fps at960x540(single) or1440x310(comparison). Originalvideos and playback copies pass framecount/PTS/full-decode checks. Metrics use original1920x1080 losslessRGB, not these compressed copies. The same completed stage1000 dense reference is reused after SHA/conditions/geometry verification. Fixed teacherstep2000 upsampler, known silence/no-text and global-position noise are unchanged.

Continuous visual review has not been completed for this checkpoint. Strict pretrained streamingdecoder and real33B longer-than67 acceptance remain pending; the comparison uses the official overlap decoder. Stage2000 ElephantsDream and WingIt300 full evaluation remain active under separate protected devices. No extra training or acceptance claim is made.
