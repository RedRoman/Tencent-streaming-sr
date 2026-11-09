# First continuation checkpoint and step0 identity scope

The TF continuation remains on the original step200-to3000 schedule. At the 2026-11-09T15:31:51Z observation, 328 real updates were recorded, including 128 new updates; parent/child identities and all three kernel locks matched. All new logged losses and gradient norms were finite, with nonzero gradients. No second NPU job was launched.

The first new saved checkpoint at step300 passed an independent read-only CPU audit: both payload SHA256 values matched the receipt; all401 parameter/AdamW states retained their ordering, shape, FP32 dtype and optimizer configuration; every AdamW counter was300; moments were finite; CPU and NPU RNG states were saved as CPU byte tensors. All401 trainable tensors changed from step200. All100 new event sample/position/history/sigma/noise schedules matched the preserved schedule. This audit performed zero optimizer updates, imported no torch_npu, and did not validate a full NPU resume or inference quality.

See [checkpoint audit](TF3000_step0300_CPU_audit_v1.json) and [running snapshot](TF3000_heartbeat_20261109T153151Z.json). Optimizer/RNG payloads remain private.

## Initialization and acceptance are separate requirements

The user asked whether step0 should exactly reproduce v5 step1800 and training should not degrade the baseline. The current refiner does **not** guarantee identity at step0. Zero new LoRA outputs and a zero draft projection leave the frozen refiner active; the rollout starts by mixing the draft with noise and then executes three denoising steps under causal attention. Same-mask offline/incremental parity does not prove equality to the upsampler-only baseline. More TF updates do not retroactively establish an identity initialization.

A future identity-preserving design would require direct latent and decoded-output parity against its exact baseline. The currently frozen teacher step2000 (`7b9f43c17462fb2d0554645bc7889dc7abdebc052ee9f18f6780624de912e0b5`) differs from v5 step1800 (`c7aaaa396d877a84c3fed2a5536189b7ed5e61226a82ef1a9e6be1e831f7f914`). The baseline was not switched and the active run was not modified.

The historical 2% metric allowance is not a strict zero-regression guarantee. Finite training gradients and lower training loss also do not guarantee per-step or per-sample quality improvement. Held-out full226 evaluation, full pretrained strict-decoder/longstream checks, and continuous visual review remain pending. No inference model is accepted and no SF/DMD promotion has occurred. See the [scope record](step0_identity_scope_clarification_v1.json).
