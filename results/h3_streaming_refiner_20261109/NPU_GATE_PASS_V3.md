# Full-model NPU gates passed; TF training started

V3 fixed per-latent projection and RMSNorm grouping eliminated the observed offline/cache discrepancy on the actual 50-block released model: max absolute error 0 and relative RMS 0, with the original frozen thresholds unchanged. The sampled v2 trace is preserved; it localized accumulation after the first block but did not independently prove the cause.

Future-frame perturbation error was 0. Full-spatial backward produced finite nonzero gradient norm 1.1025258302688599. Adapter save/reload reproduced output exactly. The gate used two 68×120 latent frames, not the complete 67-latent sequence. Peak allocated memory during gates was 37,479,535,616 bytes.

Real optimizer updates are recorded in the accompanying snapshot. The 200-step run remains active; this is not training completion or visual acceptance. All three device locks were held. The accepted upsampler and earlier task were unchanged.

Source layout now matches config paths: `experiments/h3_streaming_refiner_20261109/src/`. Historical v1/v2 source snapshots remain in prior commits; historical configs do not validate v3 sources. No private resume state, caches, full pretrained weights or selected inference weights are included.
