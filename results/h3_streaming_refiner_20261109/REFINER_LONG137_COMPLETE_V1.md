# Fixed stage1000 real33B long-stream result

The pinned stage1000 candidate (cumulative3500 optimizer updates) passed the declared real50-layer33B incremental-KV engineering test and naturally exited0 with its child reaped. Both original processes are absent and all three device locks are released. No training updates were added.

| Check | Observed result |
|---|---|
| Full-spatial stream |137 latent positions,68x120;67 authorized draft latents looped|
| Existing first67 output reproduction |Maximum absolute difference0|
| Incremental K projections |Each of50 layers processes only2040 current rows per projection;548 main forwards|
| KV storage |First latent4,005,478,400bytes; thereafter bounded6,930,022,400bytes|
| Disk KV restore at next66 |Reload adapter/head and real50-layer KV;3-latent chunk replay difference0|
| Disk KV restore at next134 |Reload adapter/head and real50-layer KV;3-latent chunk replay difference0|
| Prefix recomputation |None|
| Optimizer updates |0|

Median generation time was57.42seconds per latent. Main generation took7,910.28seconds, with24 additional replay forwards. Video playback24fps does not describe inference speed. Cache snapshots remain private and are neither downloaded nor published; public evidence contains their hashes and metadata only.

The first67 equality compares the new incremental run against the existing incremental output, not a full137 offline reference. Restoration occurs in the same worker with adapter/head reloaded and disk KV read; it does not demonstrate a new process cold-starting the entire base model. The137 positions repeat67 source latents, so this is not a real distinct long-video quality evaluation. Stage2000 long-stream behavior was not tested.

## Delivery and remaining acceptance limits

Both stage1000 andstage2000 have all three full67/226 development clips, with18 original-resolution local videos and18 lightweight public playback versions. [Video index](VIDEO_DELIVERY_PUBLIC_INDEX_V1.md) and [full226 quality results](ANCHORED_SIX_VIDEO_SETS_FULL226_V1.md) retain the negative results. Neither checkpoint is accepted or automatically ranked best.

The independent strict pretrained decoder passed its declared technical tests but failed the full226 fidelity threshold: same-input no-refiner L1ratio1.056388590 andLPIPSratio1.086851222 relative to the official overlap decoder. Existing comparison videos still use the official overlap decoder. See [decoder evidence](STRICT_DECODER_PRETRAINED_RESULT_V1.md). A refiner engineering pass cannot override this decoder quality failure.

Continuous visual review of content hallucination, fine detail, flicker and drift remains incomplete. Static frame inspection and full-file technical audits are explicitly separate. The complete end-to-end strictly streaming pipeline has not passed acceptance. Frozen sources, checkpoints, failed gates and historical evidence are preserved; no reruns or new training are started.
