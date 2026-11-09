# TF step200 first dev clip: fidelity failure

The first full226 output for `WingIt_dev_3586_3811` fails the frozen fidelity criteria. Compared with the exact frozen upsampler without a refiner, native-HR L1 is **4.621×** and LPIPS is **5.604×**. Both are well above the1.02 limit. These are single-clip results, not the pending three-clip equal-weight aggregate.

| Full226 metric | Step200 streaming | No refiner |
|---|---:|---:|
| Native-HR L1 | 0.09943487 | 0.02152031 |
| Native-HR LPIPS | 0.59713884 | 0.10655534 |
| PSNR dB | 16.8654 | 27.7283 |
| Temporal delta L1 | 0.03928526 | 0.01659604 |

Sampled-frame inspection at frames0,75,150,225 shows severe changes to character shapes and scene details, with ghosting and blur. The head/face geometry already differs strongly at frame0. This is enough to reject this checkpoint under the no-structural-drift rule; it is **not** a completed continuous-motion review. The full video remains a retained negative result, not an accepted model demonstration.

![Step200 / no-refiner / actual nativeHR GT](step200_contactsheet.jpg)

All226 frames were decoded at1920×1088 and cropped y=4:1084 to1920×1080. The provisional official overlap VAE was used for this quality comparison; strict per-latent VAE validation remains pending. The explicit no-text/known-silence engineering conditioning and two-clip TF training limitations are unchanged. No quality threshold was relaxed and no test data were used.

The refiner's full67 incremental-cache correctness result remains valid within its measured scope; it does not imply output fidelity. No self-generated-history or DMD promotion is permitted from this failed checkpoint. The original dense reference and remaining frozen checkpoint/dev evaluations still need completion.
