# Full-pretrained decoder validator: CPU admission complete

The executable validator is prepared in a new immutable stage. It has **not run the full pretrained VAE on NPU**. The existing evaluation batch keeps its three locks; no second accelerator job or waiter was launched. Training remains at 200 real optimizer updates.

The CPU admission pass verified the exact candidate source, the earlier CPU evidence, the predeclared validation plan, the frozen pretrained VAE assets and implementation, and the authorized first-WingIt input and RGB payloads by full SHA-256. The predecessor was observed active, so accelerator execution was ineligible. An actual invocation without `--admission-only` then failed at that guard before importing either Torch or torch_npu.

Negative tests reject corrupted payloads, changed validation-plan SHA, missing process identities, live PIDs even when a receipt says reaped, unreaped failures, and an invalid physical device. An initial macOS development test exposed that missing `/proc` could be mistaken for absent processes. Its failed evidence and source are retained. The corrected helper refuses admission unless Linux process state is available; the tests passed on the authorized Linux host.

After the active batch has exited and been reaped, a future one-shot launch must hold all three device locks, pass fresh device identity/idle checks, and verify the pinned launcher. The frozen validation plan requires nine-latent full-VAE parity, full-spatial reset/replay and future checks, 137-latent decoder stress, bounded FP32 host KV, and first-67/226-frame native-HR and official-decoder comparisons. The stress input loops the authorized 67-latent clip; it is not 137 unique video latents. Video technical audit and continuous visual review remain separate. Refiner validation beyond 67 latents is also still pending.

Public copies redact private lock-owner paths. The source/public SHA map records this; deployed stage and evidence hashes refer to the original authorized files. Public source snapshots with redacted deployment paths are review artifacts, not an assertion of execution under another lock convention. The frozen decoder candidate itself was not changed.

A separate evaluation milestone is complete: step0 ElephantsDream produced all 67 full-spatial latents, with first-four chunk parity exactly zero and payload SHA verified. Its 226-frame decoding and quality evaluation were still running at this snapshot. No model is selected and no self-generated-history/DMD promotion is authorized by these CPU or latent results.
