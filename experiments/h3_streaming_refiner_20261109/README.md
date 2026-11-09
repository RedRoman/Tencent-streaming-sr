# Open H3 strict streaming refiner

2026-11-09: full-model NPU cache/future/backward/reload gates passed and the 200-update Teacher Forcing pilot completed and passed the checkpoint audit. See the timestamped [status](../../results/h3_streaming_refiner_20261109/STATUS.json) and [v3 evidence](../../results/h3_streaming_refiner_20261109/NPU_GATE_PASS_V3.md). Full67/full226 and visual acceptance remain pending.

Frozen accepted upsampler: `h3_limiteddev_teacher_step2000_fp32.pt`, SHA256 `7b9f43c17462fb2d0554645bc7889dc7abdebc052ee9f18f6780624de912e0b5`. Historical native-HR quality-guard failure remains recorded. This task does not substitute a newer training checkpoint.

The production interface remains LR `[1,24,67,44,80]` to refiner input/output `[1,24,67,68,120]`, 226 frames / 24 fps, decoded 1920×1088 cropped y=4:1084 to 1080p. This milestone has **not** validated all 67 latent frames or the decoder.

## Implementation

`src/causal_attention.py` implements strict per-latent attention and bounded detached K/V. Static text/audio cannot read video; video sees static, first-frame sink, previous frame, and current spatial rows. Global positions use H3's nonuniform temporal RoPE. `src/streaming_refiner.py` accepts arbitrary user chunk boundaries and fixes noise by absolute latent position. Each latent uses the original three denoising steps **plus one clean cache-write forward**. Old video activations are not recomputed during streamed inference. Host paging is not a real-time performance claim.

`src/paged_runtime.py` preserves the frozen released 33B transformer, 312-target merged LoRA and 50 Sharp branches, using original BF16/FP32 islands. It adds rank-8 Q/K/V/out adapters and a zero-initialized clean-draft projection. CPU layer paging and custom recompute backward support a single 64 GB NPU; full-model NPU two-latent validation passed. V3 groups token-local projections and RMSNorm by latent frame to make their numerical shapes invariant to arrival batching. The offline reference still computes independent full K/V under the same causal mask.

`src/train_pilot.py` contains the gated 200-step Teacher Forcing pilot. It uses two existing paired native-HR clips (Charge/Caminandes), 67 absolute latent positions, full 68×120 spatial latents, and bounded clean-history windows. These windows are recomputed after optimizer updates and are explicitly **truncated training context**, not incremental inference. Native-HR flow matching is combined with draft x0 preservation. Original sigma schedule, quality gates, dev splits and negative results are retained. All 200 optimizer updates completed; no test-set access or DMD run has occurred.

These public proxies have synthetic LR degradation. Caption embeddings for these pairs are unavailable, so the initial engineering pilot explicitly omits text and uses known full-duration encoded silence. It does not reuse another clip's caption or pretend a zero tensor is an encoded empty prompt. Production text conditioning and full reference-refiner comparison remain unvalidated.

## Validation scope

- Synthetic attention: future error 0, offline/cache error 0, nonzero gradient, bounded cache, exactly chunk-invariant noise.
- Actual released full-width block + Sharp: 645,973,248 parameters, six token rows on CPU; ordinary vs recomputed backward input/adapter gradients identical, future and cache error 0.
- Tiny random H3: full 68×120 spatial input, two latents; runtime cache error 0, nonzero condition gradient, reload exact, stale-cache rejection, nonuniform positions.

See [evidence](../../results/h3_streaming_refiner_20261109). These CPU tests have separate scopes. The v3 NPU evidence adds actual full-model two-latent checks; 67-latent, 226-frame and visual-quality passes remain pending. The large-block random-activation gradient norm is an operator diagnostic, not a training metric.

## Run prerequisites

Use `src/` and `config/pilot_v3.json` as the staged layout. Historical v2 configs refer to prior-commit sources. Supply the existing Open H3 asset root, the audited production67 data/factory stage, a completed CPU asset SHA inventory, and the exact three device lock paths. The launcher refuses a busy card; it does not wait or terminate another process. `src/train_pilot.py --help` and `src/run_locked.py --help` document the arguments. The earlier production67 helper sources live in this repository's prior experiment snapshot; their original frozen stage layout must be restored and verified before this new pipeline is usable. No private cache, optimizer state, credentials, or full pretrained weights are included.

Next: recheck trained cache/chunking behavior and run held-out full226 native-HR evaluation. Main future video: streaming refiner pipeline | original bidirectional refiner | real GT; retain no-refiner ablation. Strict streaming decoder, long streams, cache limits, timing, boundaries, continuous visual review, and selected inference-weight Release remain pending.

## Full-sequence evaluation workers

`src/prepare_dev_cpu.py` verified all three dev pairs and reuses exact accepted-upsampled predictions plus lossless native-HR/decoded-HR/no-refiner RGB. Full SHA evidence is retained; this preparation performs no new accelerator inference.

`src/run_evaluation_batch.py` refuses incomplete training, verifies staged sources and five checkpoint hashes, then serially runs every fixed checkpoint on the three dev clips. `src/evaluate_streaming.py` rechecks the trained adapter cache/future/reload gates before full67 generation and verifies a four-latent chunk rerun plus cache bounds. `src/evaluate_dense_reference.py` uses all-video noisy timesteps and unmasked attention, with new adapters disabled; it checks against the upstream model forward before the full reference run. A tiny full-spatial CPU comparison is exact0; full33B NPU reference equivalence is still pending.

`src/evaluate_pixels.py` decodes the exact226 frames with the frozen official overlap decoder and computes native-HR metrics, including fixed regions and temporal/ROI measures. This decoder is provisional for comparing refiner quality and does not establish a strict streaming pipeline. The batch records metric-passing candidates but never automatically marks visual acceptance or selects release weights. The worker manifest is `config/evaluation_workers_v4.json`; no full67 or pixel evaluation has yet completed.

Evaluation v4 first checks all five saved adapters (latest first) on two full-spatial latent frames, including separate future-video and future-draft perturbations. These gates-only runs are labeled explicitly and are not full67/full226 results. The full metric evaluation order remains 0/50/100/150/200. Disk array shapes and batch-inclusive pipeline shapes are recorded separately; initialization latency is included in the first-latent timing.

The [completed TF pilot audit](../../results/h3_streaming_refiner_20261109/TF_PILOT_COMPLETE.md) verifies all 200 real updates and five checkpoints. Training completion is separate from inference correctness and quality acceptance.
