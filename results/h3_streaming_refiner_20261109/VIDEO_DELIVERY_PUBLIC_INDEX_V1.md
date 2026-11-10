# Six fixed-checkpoint video sets

All six sets contain226frames at24fps. Main columns: candidate / original bidirectional / actual GT. Ablation columns: candidate / frozen upsampler withoutrefiner / actualGT. These playback files are downscaled; all18 original-resolution videos are SHA-verified locally. Models are not accepted; continuous visual review and full streaming engineering validation remain pending.

| Sample | Stage step | Main comparison | No-refiner ablation | Candidate |
|---|---:|---|---|---|
| WingIt3586 | 1000 | [Main](anchored_stage1000_dense_gt_preview.mp4) | [Ablation](anchored_stage1000_no_refiner_gt_preview.mp4) | [Candidate](anchored_stage1000_native1080_preview.mp4) |
| ElephantsDream13070 | 1000 | [Main](anchored_stage1000_ElephantsDream_dense_gt_preview.mp4) | [Ablation](anchored_stage1000_ElephantsDream_no_refiner_gt_preview.mp4) | [Candidate](anchored_stage1000_ElephantsDream_native540_preview.mp4) |
| WingIt300 | 1000 | [Main](anchored_stage1000_WingIt300_dense_gt_preview.mp4) | [Ablation](anchored_stage1000_WingIt300_no_refiner_gt_preview.mp4) | [Candidate](anchored_stage1000_WingIt300_native540_preview.mp4) |
| WingIt3586 | 2000 | [Main](anchored_stage2000_WingIt3586_dense_gt_preview.mp4) | [Ablation](anchored_stage2000_WingIt3586_no_refiner_gt_preview.mp4) | [Candidate](anchored_stage2000_WingIt3586_native540_preview.mp4) |
| ElephantsDream13070 | 2000 | [Main](anchored_stage2000_ElephantsDream13070_dense_gt_preview.mp4) | [Ablation](anchored_stage2000_ElephantsDream13070_no_refiner_gt_preview.mp4) | [Candidate](anchored_stage2000_ElephantsDream13070_native540_preview.mp4) |
| WingIt300 | 2000 | [Main](anchored_stage2000_WingIt300_dense_gt_preview.mp4) | [Ablation](anchored_stage2000_WingIt300_no_refiner_gt_preview.mp4) | [Candidate](anchored_stage2000_WingIt300_native540_preview.mp4) |

[Full226 metrics and limitations](ANCHORED_SIX_VIDEO_SETS_FULL226_V1.md)
