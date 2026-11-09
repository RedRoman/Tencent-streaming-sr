# Open H3 strict streaming refiner

Current direction: continue Teacher Forcing to **3000 total real optimizer updates** from the saved step200 adapter and AdamW state. Real NPU resume gates passed and new training updates have started. Step0 is an untrained baseline and200 steps are a short pilot; their failed outputs do not reject TF or stop this extension. The unfinished low-step evaluation grid is deferred. See the [training extension and evidence](../../results/h3_streaming_refiner_20261109/TF_EXTENSION_3000_V1.md) and timestamped [status](../../results/h3_streaming_refiner_20261109/STATUS.json).

2026-11-09: full-model NPU cache/future/backward/reload gates passed and the 200-update Teacher Forcing pilot completed and passed the checkpoint audit. See the timestamped [status](../../results/h3_streaming_refiner_20261109/STATUS.json) and [v3 evidence](../../results/h3_streaming_refiner_20261109/NPU_GATE_PASS_V3.md). The first step200 dev clip completed full67/full226 and failed fidelity: L1 4.621× and LPIPS 5.604× versus no-refiner, with severe sampled-frame structural drift. See the [negative result](../../results/h3_streaming_refiner_20261109/STEP200_NEGATIVE_RESULT.md).

Frozen accepted upsampler: `h3_limiteddev_teacher_step2000_fp32.pt`, SHA256 `7b9f43c17462fb2d0554645bc7889dc7abdebc052ee9f18f6780624de912e0b5`. Historical native-HR quality-guard failure remains recorded. This task does not substitute a newer training checkpoint.

The production interface remains LR `[1,24,67,44,80]` to refiner input/output `[1,24,67,68,120]`, 226 frames / 24 fps, decoded 1920×1088 cropped y=4:1084 to 1080p. The first step200 dev clip has completed all 67 latent frames. Decoder and quality acceptance remain pending.

Initialization limitation: the current step0 refiner is **not an identity mapping** of the upsampler output. Zero new adapters do not disable noisy three-step denoising. The frozen teacher step2000 and v5 step1800 are distinct checkpoints, and the historical 2% metric allowance is not strict zero regression. See the [first continuation checkpoint audit and identity scope](../../results/h3_streaming_refiner_20261109/TF3000_STEP300_AND_IDENTITY_SCOPE_V1.md).

## Implementation

`src/causal_attention.py` implements strict per-latent attention and bounded detached K/V. Static text/audio cannot read video; video sees static, first-frame sink, previous frame, and current spatial rows. Global positions use H3's nonuniform temporal RoPE. `src/streaming_refiner.py` accepts arbitrary user chunk boundaries and fixes noise by absolute latent position. Each latent uses the original three denoising steps **plus one clean cache-write forward**. Old video activations are not recomputed during streamed inference. Host paging is not a real-time performance claim.

`src/paged_runtime.py` preserves the frozen released 33B transformer, 312-target merged LoRA and 50 Sharp branches, using original BF16/FP32 islands. It adds rank-8 Q/K/V/out adapters and a zero-initialized clean-draft projection. CPU layer paging and custom recompute backward support a single 64 GB NPU; full-model NPU two-latent validation passed. V3 groups token-local projections and RMSNorm by latent frame to make their numerical shapes invariant to arrival batching. The offline reference still computes independent full K/V under the same causal mask.

`src/train_pilot.py` contains the gated 200-step Teacher Forcing pilot. It uses two existing paired native-HR clips (Charge/Caminandes), 67 absolute latent positions, full 68×120 spatial latents, and bounded clean-history windows. These windows are recomputed after optimizer updates and are explicitly **truncated training context**, not incremental inference. Native-HR flow matching is combined with draft x0 preservation. Original sigma schedule, quality gates, dev splits and negative results are retained. All 200 optimizer updates completed; no test-set access or DMD run has occurred.

These public proxies have synthetic LR degradation. Caption embeddings for these pairs are unavailable, so the initial engineering pilot explicitly omits text and uses known full-duration encoded silence. It does not reuse another clip's caption or pretend a zero tensor is an encoded empty prompt. Production text conditioning remains unvalidated. The native dense reference completed its first full67/full226 clip after an exact full-model two-latent upstream gate. It also failed the first-clip fidelity metrics; see the current status.

## Validation scope

- Synthetic attention: future error 0, offline/cache error 0, nonzero gradient, bounded cache, exactly chunk-invariant noise.
- Actual released full-width block + Sharp: 645,973,248 parameters, six token rows on CPU; ordinary vs recomputed backward input/adapter gradients identical, future and cache error 0.
- Tiny random H3: full 68×120 spatial input, two latents; runtime cache error 0, nonzero condition gradient, reload exact, stale-cache rejection, nonuniform positions.

See [evidence](../../results/h3_streaming_refiner_20261109). These CPU tests have separate scopes. The v3 NPU evidence adds actual full-model two-latent checks; the first full67 step200 clip is separately recorded in the current status. The frozen full226 cohort and visual-quality acceptance remain pending. The large-block random-activation gradient norm is an operator diagnostic, not a training metric.

## Run prerequisites

Use `src/` and `config/pilot_v3.json` as the staged layout. Historical v2 configs refer to prior-commit sources. Supply the existing Open H3 asset root, the audited production67 data/factory stage, a completed CPU asset SHA inventory, and the exact three device lock paths. The launcher refuses a busy card; it does not wait or terminate another process. `src/train_pilot.py --help` and `src/run_locked.py --help` document the arguments. The earlier production67 helper sources live in this repository's prior experiment snapshot; their original frozen stage layout must be restored and verified before this new pipeline is usable. No private cache, optimizer state, credentials, or full pretrained weights are included.

Next: finish the user-directed TF3000 continuation, then evaluate trained1000/2000/3000 checkpoints on held-out full226 native-HR data. Main future video: streaming refiner pipeline | original bidirectional refiner | real GT; retain no-refiner ablation. Strict streaming decoder, long streams, cache limits, timing, boundaries, continuous visual review, and selected inference-weight Release remain pending.

## Full-sequence evaluation workers

`src/prepare_dev_cpu.py` verified all three dev pairs and reuses exact accepted-upsampled predictions plus lossless native-HR/decoded-HR/no-refiner RGB. Full SHA evidence is retained; this preparation performs no new accelerator inference.

`src/run_evaluation_batch.py` refuses incomplete training, verifies staged sources and five checkpoint hashes, then serially runs every fixed checkpoint on the three dev clips. `src/evaluate_streaming.py` rechecks the trained adapter cache/future/reload gates before full67 generation and verifies a four-latent chunk rerun plus cache bounds. `src/evaluate_dense_reference.py` uses all-video noisy timesteps and unmasked attention, with new adapters disabled; it checks against the upstream model forward before the full reference run. The original query-sliced v4 dense implementation failed the full33B NPU gate (maxabs0.254151, relative RMS0.017853), before full inference. Its source and failure remain preserved. The separate `stages/dense_native_v1` worker restores the original native attention processor and full-row output head. The unchanged NPU gate is exact0, and the first full67 three-step output completed; no streaming worker or trained checkpoint changed.

`src/evaluate_pixels.py` decodes the exact226 frames with the frozen official overlap decoder and computes native-HR metrics, including fixed regions and temporal/ROI measures. This decoder is provisional for comparing refiner quality and does not establish a strict streaming pipeline. The batch records metric-passing candidates but never automatically marks visual acceptance or selects release weights. The worker manifest is `config/evaluation_workers_v4.json`; the first step200 full67 inference has completed; the first full226 evaluation is complete and failed fidelity.

Evaluation v4 first checks all five saved adapters (latest first) on two full-spatial latent frames, including separate future-video and future-draft perturbations. These gates-only runs are labeled explicitly and are not full67/full226 results. The historical short-pilot metric order was0/50/100/150/200; its unfinished jobs are now deferred by the user-directed TF3000 extension. Disk array shapes and batch-inclusive pipeline shapes are recorded separately; initialization latency is included in the first-latent timing.

The [completed TF pilot audit](../../results/h3_streaming_refiner_20261109/TF_PILOT_COMPLETE.md) verifies all 200 real updates and five checkpoints. Training completion is separate from inference correctness and quality acceptance.

## Prioritized preview and continuation

The first dev clip at TF step200 completed all67 latents with 268 forwards, stable cache after the second latent, and an exact four-latent chunk replay. See [full67 evidence](../../results/h3_streaming_refiner_20261109/STEP200_FULL67.md). The requested interim preview uses step200 / no-refiner / actual native-HR columns; the final selected-streaming / released-dense / actual-GT comparison is still required.

`src/run_priority_preview.py` uses the unchanged v4 workers and explicit checkpoint SHA. `src/encode_preview_cpu.py` verifies source RGB, encodes all226 frames, checks every PTS and performs a complete decode audit; this is distinct from continuous visual review. The bounded CPU postprocessor starts no accelerator job. `src/run_evaluation_continuation.py` requires exited/reaped predecessors, verifies completed receipts and payloads, reuses five short preflights plus the completed preview, and preserves the interrupted partial step0. It prioritizes the dense reference clip before the remaining frozen cohort. Quality thresholds, held-out samples and selection remain unchanged; continuation is prepared for the remaining cohort after the preview launcher exits and resource admission succeeds.

## Native dense continuation

`src/run_evaluation_continuation_v2.py` verifies both immutable worker manifests and permits only the pinned dense worker change. It requires all predecessor children to be reaped, revalidates completed latent/RGB payload hashes, and reuses the five preflights plus the first streaming and native-dense previews. `--admission-only` checks reuse on CPU without starting any model worker. The remaining cohort and quality gates are unchanged. Optional resident-layer count changes paging only; every streaming job still repeats the real cache, future-video, future-condition, reload and chunk gates. Failure stops the finite batch. No device waiter or automatic training promotion is added.

`src/encode_dense_comparison_cpu.py` creates the interim step200 (rejected) / released bidirectional / actual-GT video only from verified completed results. It audits all226 frames and timestamps at24fps. The prior no-refiner video remains the ablation. Neither video is a selected-model or strict-decoder acceptance claim.

## CPU long-index diagnostic

[137-latent CPU evidence](../../results/h3_streaming_refiner_20261109/LONG_STREAM_CPU_V2.md) exercises the actual cache kernel on small synthetic rows and verifies full-spatial positions/noise across the67-latent boundary. Observed parity and future errors are0 and the synthetic cache stays bounded. The first test-reference rounding failure and its correction are preserved. This does not validate the full33B model beyond67 latents or the strict streaming decoder.

## Strict streaming decoder candidate

`src/strict_streaming_decoder.py` and the independent `stages/strict_decoder_v1` implement a candidate using CPU-backed per-layer KV with bounded shared-slot refresh. The frozen FP32 convention and1080p crop are retained; no fitted pixel compensation or compiled kernels are enabled. The candidate passed the [random small-model CPU checks](../../results/h3_streaming_refiner_20261109/STRICT_DECODER_CPU_V1.md), including137 latent positions and one full-spatial1080p output. Full pretrained NPU execution, fidelity and visual acceptance remain pending. This candidate was not used in the completed pilot previews; full pretrained validation remains pending after prioritized TF training.

## First-clip checkpoint comparison

The untrained causal step0 now completed all67 latents and226 frames, with exact first4 chunk replay, and failed fidelity: L1 is6.209x and LPIPS5.938x the no-refiner baseline. TF200 improves both metrics relative to step0 on this one clip but remains well outside the frozen quality gate. Four sampled rows show severe semantic substitutions at step0 and persistent distortion at step200. The [partial comparison](../../results/h3_streaming_refiner_20261109/WINGIT_CHECKPOINT_COMPARISON_V1.md) preserves the full metrics and auxiliary five-column video. The unfinished low-step dev cohort is deferred while TF continues toward3000 updates; no checkpoint has been accepted or promoted to SF/DMD.

## TF3000 continuation

`src/train_tf_extended.py` restores the exact step200 adapter and401 AdamW states, validates their parameter ordering and moments, and repeats real backward/causality/cache/reload gates before update201. `src/tf_resume_utils.py` preserves the absolute training schedule and reads the known legacy NPU optimizer format on CPU with a restricted weights-only loader. `config/tf_extension_3000_v1.json` pins the continuation sources and inputs. New checkpoints are saved every100 updates with private CPU optimizer/RNG state; only source and sanitized evidence are published here. The acceptance thresholds remain unchanged, while short-pilot failures do not halt this user-directed longer training run.
