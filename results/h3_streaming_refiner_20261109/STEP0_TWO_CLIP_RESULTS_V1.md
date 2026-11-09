# Step0: two completed dev clips, conclusive L1 rejection

The full226 ElephantsDream result is complete and fails the per-clip fidelity comparison. Training remains at 200 real optimizer updates; these are evaluation results of the saved step0 adapter.

| Dev clip | L1 / no-refiner | LPIPS / no-refiner | PSNR (dB) | Temporal delta L1 |
|---|---:|---:|---:|---:|
| WingIt_dev_3586_3811 | 6.2087 | 5.9381 | 14.5947 | 0.064915 |
| ElephantsDream_dev_13070_13295 | 5.1835 | 4.8113 | 17.0208 | 0.049543 |

The final equal-clip metric divides the mean native-HR L1 by the mean L1 of the frozen no-refiner baseline over all three dev clips. Since L1 is nonnegative, even zero error on the unfinished third clip leaves the step0 ratio at least **3.885677**, above the unchanged **1.02** gate. This is a bound proving that checkpoint0 cannot pass the final L1 gate, not a claim that all three evaluations have finished. The third clip continues for complete evidence; the other checkpoints still require their own evaluations. No failure cause or training generalization is inferred.

ElephantsDream has all67 latents and226 RGB frames at1920×1080 after the exact y=4:1084 crop. The first-four latent chunk check is exactly zero. Quantitative metrics use lossless RGB, the same frozen upsampler and conditioning, and the provisional official overlap decoder. The auxiliary streaming/no-refiner/GT video is separate from the required selected-streaming/original-dense/GT main comparison. Technical video verification, sampled images and actual continuous visual review are distinct; continuous review and the strict decoder remain pending. No refiner is accepted and there is no self-generated-history/DMD promotion.

Sampled visual inspection of frames0/75/150/225 also rejects the ElephantsDream result: the refiner introduces different clothing/shelving, plant leaves, a person/staircase, and a person/drum set in place of the GT hand and cable scene. These observations apply to four sampled frames, not a completed continuous motion review.
