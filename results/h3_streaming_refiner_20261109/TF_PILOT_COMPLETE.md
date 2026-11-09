# Real 200-update Teacher Forcing pilot completed

The run completed all 200 ordered optimizer updates and exited with code 0; its child was reaped. Charge and Caminandes each supplied 100 updates spanning all 67 absolute latent indices. Every recorded gradient norm was finite and nonzero. The frozen accepted upsampler, original sigma schedule and native-HR target stayed unchanged.

All five adapters (steps 0, 50, 100, 150, 200) passed whole-file SHA, schema, parameter-count and finite-FP32 checks. Each contains 401 tensors / 20,586,496 parameters / 82,478,252 bytes. All 401 tensors differ from initialization at each trained checkpoint, including the learned clean-draft projection. Optimizer/RNG state was not loaded for this audit and is not published.

Training elapsed time was 4,106.32 seconds (68.44 minutes); peak allocated NPU memory was 38,889,340,416 bytes. These are training measurements, not streaming throughput or quality results. The complete audit and sanitized event stream are retained alongside earlier failures and partial snapshots.

Full67 inference, held-out full226 native-HR metrics, continuous visual review, strict decoder validation and final model selection remain pending. No trained inference-weight Release or visual improvement is claimed. Evaluation v4 is prepared with a short latest-first five-checkpoint preflight before the unchanged full evaluation order.
