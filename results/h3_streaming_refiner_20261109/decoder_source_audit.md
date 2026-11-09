# Pinned decoder source audit — not an execution result

SolarWM commit `0193da07e44def7abd661c5f623ea352cec2722b` was resolved on 2026-11-09. Two distinct implementations must not be conflated:

- `solar/backends/minimax_h3/stream_decode.py` (blob `6cd6c10086f8e8e22ba43d3d649ae696ee7c3379`) yields bounded-memory official temporal clips. It still reads overlapping latent windows; it is not strict per-latent zero-lookahead decoding.
- `deploy/solar_wm/shared/vae/streaming.py` (blob `fc3b402f21c84439b523b3c081a9c4783e9f3003`) implements per-latent KV with original 7-latent windows, shared-latent cache refresh, and optional fitted pixel compensation. Emitted frames are not revised. Its supplied calibration tables and compilation assumptions require separate validation for this project's geometry, precision and Ascend backend.

The latter emits 1 frame at latent indices divisible by5, otherwise4: 67 latents yields 14*1+53*4=226 frames. This arithmetic agrees with the target, but establishes neither numerical/visual parity nor correct 1080p tiling/crop. Default compensation uses offline-fitted coefficients; do not silently use them as if calibrated on this project. Official-window parity tests in tools/h3_stream_decode_parity.py target the first implementation, not proof that strict per-latent causal replacement equals the original bidirectional VAE.

Next: adapt only after actual refiner cache parity and TF pilot; verify frames, future perturbations, crop, boundary motion, long-stream cache bound and quality. No decoder has been installed, trained or claimed ready by this source inspection.
