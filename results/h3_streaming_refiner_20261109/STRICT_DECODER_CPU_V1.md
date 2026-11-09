# Strict streaming VAE candidate — CPU validation only

An independent candidate now decodes one latent per call, preserving the frozen FP32 VAE convention and1920x1088 to1920x1080 crop. No running model worker or saved checkpoint was changed. This candidate has not run the full pretrained VAE on an NPU and is not accepted for the final pipeline.

## Implementation and provenance

The core is adapted from [SolarWM streaming.py](https://github.com/Junchao-cs/SolarWM/blob/0193da07e44def7abd661c5f623ea352cec2722b/deploy/solar_wm/shared/vae/streaming.py), git blob `fc3b402f21c84439b523b3c081a9c4783e9f3003`. The unmodified source, Apache-2.0 license and upstream notices are retained under `vendor/`. The separate modified source identifies the changes in its header.

The candidate uses eager operators without fitted pixel compensation. It keeps rotated per-layer KV in CPU memory and transfers the visible slice for each layer/tile batch. The existing FP32 convention, temporal window7/stride5 and spatial tiling256/overlap64 are preserved. Device transfer parity and actual NPU latency are still unmeasured.

The released tile splitter yields60 tiles for1088x1920. Full36-layer FP32 KV requires63,594,823,680 bytes before model memory. Keeping this cache on the NPU risks exceeding64GB; host backing avoids that allocation while preserving precision. This is a source-derived memory estimate, not an observed production allocation or performance result.

This decoder refreshes the two shared latents when its window shifts and once refreshes the first latent alongside the second. Those bounded recomputations are explicit. It does not recompute the whole prefix, and already emitted pixels are not revised. Its causal output need not equal the official bidirectional overlap decoder; that difference requires actual native-HR and visual evaluation.

## Executed CPU checks

The test instantiates the actual Diffusers H3 VAE architecture with random weights, a two-layer decoder, two heads and16 dimensions per head. It loads no pretrained weights and performs no accelerator forward or optimizer update.

- 137 latent positions at2x3 spatial resolution: pinned upstream incremental comparison, future perturbation after index66, and reset/chunk replay all have max absolute error0.
- 67 latents emit226 frames;137 emit464 frames.
- The small cache remains24,064 bytes with at most seven retained latent inputs. A projection hook observes190 slot forwards, including53 explicit past-slot refreshes and at most three slots per push. There is no prefix reprojection.
- A full68x120 first latent yields exactly one1080x1920 RGB frame. The independently computed upstream normalization and crop match byte for byte, with nontrivial latent mean/std. Its small-model cache is55,203,840 bytes, not the full-model cache size.
- Duplicate/out-of-order indices, multiple latent frames, wrong spatial shape and nonfinite values are rejected; reset replay is exact.

See `strict_decoder_cpu_v1.json` and `config/strict_decoder_validation_v1.json`. The NPU plan requires actual pretrained parity, full-spatial replay/future checks,137 latent positions, bounded cache, full226 native-HR metrics and separate continuous visual review. The long input deliberately loops the authorized67-latent clip and must be labeled as a stress test. The separate full-model refiner long-stream test is still pending.
