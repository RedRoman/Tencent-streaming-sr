# First-clip checkpoint comparison: no accepted refiner

The complete226-frame WingIt result at step0 also fails fidelity. TF200 is better than step0 on this clip, but neither meets the frozen2% nonregression gate against the accepted upsampler without a refiner. This is a partial cohort; the other fixed checkpoints and dev clips are still being evaluated.

| Arm | L1 | LPIPS | PSNR (dB) | L1 / no-refiner | LPIPS / no-refiner |
|---|---:|---:|---:|---:|---:|
| no_refiner | 0.021520 | 0.106555 | 27.728 | 1.000 | 1.000 |
| stream_step0 | 0.133614 | 0.632739 | 14.595 | 6.209 | 5.938 |
| stream_step200 | 0.099435 | 0.597139 | 16.865 | 4.621 | 5.604 |
| released_bidirectional | 0.060987 | 0.342936 | 19.496 | 2.834 | 3.218 |

TF200 reduces L1 by25.58% and LPIPS by5.63% relative to causal step0 on this one clip. These comparisons do not establish generalization or explain whether the remaining error comes from causal adaptation, conditioning, noise level, training objective, or their interactions. No SF/DMD promotion is authorized by these results.

All three refiner arms share the exact frozen upsampler, no-text/silence conditioning and global position-indexed noise. All rendered arms use the same geometry and provisional official overlap decoder. The native bidirectional control uses its previously verified original attention implementation. Metrics use lossless226-frame RGB, not compressed previews.

Step0 completed67 latents and268 model forwards plus one static initialization; the first4-latent chunk replay is exactly0. Cache stays6,930,022,400 bytes after the second latent. The40-layer resident run peaked at56,719,139,840 allocated bytes. Steady latency is reported in the JSON; it must not be compared as a checkpoint speedup against the earlier24-layer resident TF200 run.

The five-column auxiliary video compares step0, step200, original bidirectional, no-refiner and true GT. It supplements the previously delivered three-column video. Technical frame/timestamp/decode checks are separate from continuous visual review. No strict streaming decoder or full-model >67-latent acceptance is claimed.


Four sampled rows at frames0/75/150/225 independently show severe step0 semantic substitutions, including vehicles and a garage-like setting in place of the animated GT scene. Step200 retains some coarse character-like shapes but remains distorted and smeared. This sampled inspection is a failure finding, not continuous visual acceptance.
