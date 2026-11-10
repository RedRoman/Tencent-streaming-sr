# Existing decoder metrics: startup and refresh groups

This post-hoc analysis uses the completed no-refiner decoder ablation only: the same frozen teacher draft decoded by the strict causal candidate and official overlap decoder, both compared with native HR. It performs no new NPU forwards, training, RGB decoding or compensation. The source result SHA is `079d0052455865ce5375295188c93221c3fa333c576491e4c9e80610af4fc592`.

The observed fidelity regression is not confined to the first frame or initial decoder window. Excluding frame 0 leaves L1 error 4.86% higher and LPIPS 8.66% higher. Excluding the first seven latents (22 frames) leaves L1 4.61% higher and LPIPS 8.43% higher. These exclusions are descriptive diagnostics; they do not replace the original full226 acceptance test.

| Existing output subset | Frames | L1 ratio | LPIPS ratio | MSE ratio |
|---|---:|---:|---:|---:|
| All frames | 226 | 1.056389 | 1.086851 | 1.104119 |
| Exclude frame 0 | 225 | 1.048607 | 1.086641 | 1.095122 |
| Exclude first 7 latents | 204 | 1.046063 | 1.084327 | 1.090939 |
| Refresh pushes after first window | 48 | 1.024097 | 1.035451 | 1.036877 |
| Ordinary pushes after first window | 156 | 1.052768 | 1.098564 | 1.106953 |

Ratios are strict/official arithmetic mean per-frame errors; values above 1 mean worse fidelity. The frame-to-latent mapping and refresh grouping come from actual recorded cumulative output counts and past-slot refresh counts, not timing estimates. The full226 L1, MSE and LPIPS aggregates reproduce the completed receipt within 1e-12; the two post-initial-window groups partition all 204 remaining frames without overlap.

Refresh and ordinary groups both regress on average. The ordinary group has larger ratios in this clip, so the data do not support describing the regression as present only during refresh pushes. This is not a causal comparison: content, available history, latent phase and official overlap behavior differ between groups. No intervention separates missing future context, slot/window position, blending or refresh effects. Temporally dependent frames are not independent experiments; no significance or generalization claim is made.

The original 1.02 full226 fidelity limit remains unchanged and failed. No refiner is present in either decoder arm, so these observations are not new candidate-refiner quality results. Continuous visual review remains incomplete; metrics and static frames cannot establish flicker, drift or absence of hallucination by viewing.

CPU analysis v1 encountered a receipt-schema KeyError because the convenience ratio object contains only L1 and LPIPS. That failure and script are retained. V2 checks MSE against the recorded region means, retaining the same 1e-12 consistency tolerance. No NPU state or original result changed. See `strict_decoder_existing_metrics_diagnostic_v2.json` and the two versioned analysis scripts.
