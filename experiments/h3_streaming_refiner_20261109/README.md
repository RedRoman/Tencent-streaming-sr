# Open H3 strict streaming refiner: implementation milestone

2026-11-09. **Optimizer updates: 0. No trained refiner or quality improvement is claimed.** The first accelerator admission was refused by a live device lock. No existing training process was interrupted. The implementation is being supervised independently.

Frozen accepted upsampler: `h3_limiteddev_teacher_step2000_fp32.pt`, SHA256 `7b9f43c17462fb2d0554645bc7889dc7abdebc052ee9f18f6780624de912e0b5`. Historical native-HR quality-guard failure remains recorded. This task does not substitute a newer training checkpoint.

The production interface remains LR `[1,24,67,44,80]` to refiner input/output `[1,24,67,68,120]`, 226 frames / 24 fps, decoded 1920×1088 cropped y=4:1084 to 1080p. This milestone has **not** validated all 67 latent frames or the decoder.

## Implementation

`causal_attention.py` implements strict per-latent attention and bounded detached K/V. Static text/audio cannot read video; video sees static, first-frame sink, previous frame, and current spatial rows. Global positions use H3's nonuniform temporal RoPE. `streaming_refiner.py` accepts arbitrary user chunk boundaries and fixes noise by absolute latent position. Each latent uses the original three denoising steps **plus one clean cache-write forward**. Old video activations are not recomputed during streamed inference. Host paging is not a real-time performance claim.

`paged_runtime.py` preserves the frozen released 33B transformer, 312-target merged LoRA and 50 Sharp branches, using original BF16/FP32 islands. It adds rank-8 Q/K/V/out adapters and a zero-initialized clean-draft projection. CPU layer paging and custom recompute backward support a single 64 GB NPU; full-model NPU validation is pending.

`train_pilot.py` contains the gated 200-step Teacher Forcing pilot. It uses two existing paired native-HR clips (Charge/Caminandes), 67 absolute latent positions, full 68×120 spatial latents, and bounded clean-history windows. These windows are recomputed after optimizer updates and are explicitly **truncated training context**, not incremental inference. Native-HR flow matching is combined with draft x0 preservation. Original sigma schedule, quality gates, dev splits and negative results are retained. No test-set access, DMD run, or optimizer update has happened here.

These public proxies have synthetic LR degradation. Caption embeddings for these pairs are unavailable, so the initial engineering pilot explicitly omits text and uses known full-duration encoded silence. It does not reuse another clip's caption or pretend a zero tensor is an encoded empty prompt. Production text conditioning and full reference-refiner comparison remain unvalidated.

## Validation scope

- Synthetic attention: future error 0, offline/cache error 0, nonzero gradient, bounded cache, exactly chunk-invariant noise.
- Actual released full-width block + Sharp: 645,973,248 parameters, six token rows on CPU; ordinary vs recomputed backward input/adapter gradients identical, future and cache error 0.
- Tiny random H3: full 68×120 spatial input, two latents; runtime cache error 0, nonzero condition gradient, reload exact, stale-cache rejection, nonuniform positions.

See [evidence](../../results/h3_streaming_refiner_20261109). These are different tests and must not be described as full-model NPU, 67-latent, 226-frame or visual-quality passes. The large-block random-activation gradient norm is an operator diagnostic, not a training metric.

## Run prerequisites

Supply the existing Open H3 asset root, the audited production67 data/factory stage, a completed CPU asset SHA inventory, and the exact three device lock paths. The launcher refuses a busy card; it does not wait or terminate another process. `train_pilot.py --help` and `run_locked.py --help` document the arguments. The earlier production67 helper sources live in this repository's prior experiment snapshot; their original frozen stage layout must be restored and verified before this new pipeline is usable. No private cache, optimizer state, credentials, or full pretrained weights are included.

Required next gates: full actual model/cache parity, future perturbation, backward, save/reload; then 200 optimizer updates and held-out full226 native-HR evaluation. Main future video: streaming refiner pipeline | original bidirectional refiner | real GT; retain no-refiner ablation. Strict streaming decoder, long streams, cache limits, timing, boundaries, continuous visual review, and selected inference-weight Release remain pending.
