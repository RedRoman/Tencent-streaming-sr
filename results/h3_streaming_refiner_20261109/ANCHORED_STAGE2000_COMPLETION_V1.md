# Anchored stage2000 completion and evaluation scope

The separate anchored residual training stage completed exactly2000 consecutive real optimizer updates, following the sealed originalTF2500 checkpoint. The cumulative4500 count spans two different recipes. All2000 losses are finite. The unchanged worker saved its endpoint, performed the planned three first4-latent development probes, then naturally exited0; the parent and child are absent and the launcher records the child reaped. No training extension, new training controller or optimizer reset occurred.

The final adapter SHA is `5e95b0826db51003968597b8ffbc6159f42d0c784ad8c0a10843734fae3dc485`. The private resume payload SHA is `5541609e21faa4ad4cd97b09a1455fd9a81ba08236465b0359f8cc8077b93549`; the checkpoint receipt SHA is `b6f398f549de1a4dbb0c606dccf9236a41441d4ad37cc7aa873309ca6e7c042e`. Payloads remain private. The CPU audit validates all401 original AdamW slots at4500 and four head slots at2000, parameter names/shapes, finite moments, learning rates, CPU/NPU RNG byte tensors, and the exact next schedule. CPU RNG roundtrip is exact; the stored NPU RNG was checked on CPU, not restored into a live NPU by this audit.

| Development sample | Stage500 MSE ratio | Stage1000 | Stage2000 |
|---|---:|---:|---:|
| WingIt_dev_3586_3811 | 1.201447 | 1.096375 | 1.102577 |
| ElephantsDream_dev_13070_13295 | 0.948203 | 0.922913 | 0.947870 |
| WingIt_dev_300_525 | 1.282472 | 1.161290 | 1.166400 |

Ratios use nativeHR latent MSE divided by the same frozen no-refiner draft MSE; lower is better. These are only the first4 latents. Allthree stage2000 ratios are worse than stage1000, while both WingIt ratios remain above1. This does not establish full-video ranking or a unique cause, and no candidate is accepted. The fixedstage1000 full67/full226 video cohort continues with its pinned adapter; finalstage2000 does not replace it. Fullstage2000 pixel/video evaluation remains pending.

The first fixedstage1000 WingIt full226 result still has improved pixelL1 and worseLPIPS relative to no-refiner. User viewing feedback and static reinspection distinguish conspicuous added content in the original bidirectional reference from the more content-preserving candidate and no-refiner results. This is not an exhaustive no-hallucination or continuous temporal pass, and no-refiner already shares that content-preserving behavior. Added value over no-refiner remains an acceptance requirement.

True per-latent incrementalKV is implemented and the completed67-latent run has bounded6.93GB KV; it is not a realtime performance claim. The first full67 generation took66.34minutes with approximately60.24seconds per steady-state latent. Strict pretrained streaming-decoder validation, longer-than67 full33B runs and continuous temporal quality review remain pending. Earlier failures, original and public SHA lineage, and the frozen teacherstep2000 baseline are retained.

After verified natural trainer exit/reap, the third fixedstage1000 clip WingIt_dev_300_525 was admitted on released NPU7 using a fresh npu-smi observation and allthree device locks. ElephantsDream continues on NPU4. Both evaluators retain the exactstage1000 adapter SHA; no duplicate job or waiter was created.
