# CPU long-index checks: 137 latent positions

The CPU diagnostic passed for 137 latent positions. The complete 33B model has not been tested beyond67 latents, and the strict streaming decoder remains unvalidated. No optimizer update or accelerator forward was performed by this diagnostic. The saved raw result uses `frames` for latent indices; it does not mean137 decoded video frames.

## Measured scope

- The actual causal attention/cache implementation was exercised with five synthetic spatial rows per latent and seven static rows. Offline versus incremental output and future perturbation at latent89 both had maximum absolute error0. The frozen synthetic parity assertion remains less than2e-6; the observed value is exactly0.
- A K-projection hook observed seven static rows followed by exactly five new rows for each of137 latents, totaling692 rows. It observed no prefix reprojection. Duplicate final-latent input was rejected.
- The synthetic cache occupied2176 bytes after the second latent and at the final latent. This is the small synthetic test size, not the full-model cache size.
- Full68x120 position packing was compared with the released lower-level layout builder at13 positions through136, including65/66/67/68 and133/134/135/136, for zero and nine text tokens. Both comparisons were exactly0.
- Full24x1x68x120 noise tensors at seven absolute latent positions had identical SHA256 under a different request order and distinct hashes across positions.

The full-model67-latent run is separate evidence with a6,930,022,400-byte cache. A full-spatial, full-model sequence beyond67 and strict VAE future/boundary/cache tests are still required. This result does not establish visual quality, real-time operation, or a complete strict streaming video pipeline.

## Preserved first test failure

The first CPU check stopped at the exact position comparison. Its zero-text reference was built by constructing a one-text-token layout and subtracting1 from time coordinates. At latent1, `(1 + 5/3) - 1` differs from `5/3` by2.220446049250313e-16 in float64. The independent coordinate trace identifies this difference; spatial coordinates and the nine-text-token comparison were unchanged.

The corrected diagnostic calls the released `MiniMaxH3PrepareLayoutStep.build_packed_sequence` directly with zero text tokens. The runtime, native temporal grid, model workers and exact position assertion were not changed. The original test source, sanitized failure log, coordinate trace and original/public hashes are retained. This was a reference-construction error in the diagnostic; it is not evidence of a model failure.

Run in the pinned project Python environment with accelerator autoload disabled and visible accelerator lists empty:

```sh
python src/check_long_stream_cpu.py --stage PATH_TO_PINNED_EVALUATION_V4 --output NEW_RESULT_JSON
```

See `long_cpu_v2.json`, `long_cpu_position_trace_v1.json`, `long_cpu_v1_failure.log`, and the SHA maps. The script checks both runtime source hashes before execution and refuses to overwrite an existing result.
