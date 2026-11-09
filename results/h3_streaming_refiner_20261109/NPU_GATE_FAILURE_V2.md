# Full-model NPU gate failed before training

The second device admission succeeded and held all three locks. It prepared two real full67 paired inputs using the exact accepted upsampler, encoded known silence for226frames, loaded the complete66.28GB mixed-precision refiner and20,586,496 trainable parameters, then tested the same causal mask against incremental KV.

**Optimizer updates: 0.** Cache parity produced maxabs0.2439553 and relative RMS0.01743643, above the frozen0.03 /0.005 gates. The process exited with failure before any optimizer update. Gates were not relaxed. This invalidates any claim that full-model incremental parity is already solved; the earlier CPU passes keep their narrower scopes.

A full-model per-module trace was prepared to distinguish temporal/conditioning/cache errors from sequence-shape-dependent floating-point execution. The diagnostic admission was refused by another active job's lock; no process was interrupted. Root cause is still unconfirmed. The two prepared full67 input caches remain reusable; no repeated old experiment or surrogate geometry is required.

The independent heartbeat continues from the saved current state. First collect the diagnostic, then correct a new immutable source version and rerun required gates. Only a gate-passing model may begin the200-update pilot. The frozen evaluation protocol is in the experiment directory. This is an implementation/negative-diagnostic milestone, not trained-weight or video-quality delivery.
