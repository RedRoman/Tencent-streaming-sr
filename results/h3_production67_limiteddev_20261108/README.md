# Limited-dev full226 evaluation: native-HR guard failed
完整 **226 帧 / 24 fps** 三列主视频：[ElephantsDream_dev_13070_13295](https://github.com/RedRoman/Tencent-streaming-sr/releases/download/h3-limiteddev-226f-20261108/ElephantsDream_dev_13070_13295__comparison_3column.h264.mp4) · [WingIt_dev_300_525](https://github.com/RedRoman/Tencent-streaming-sr/releases/download/h3-limiteddev-226f-20261108/WingIt_dev_300_525__comparison_3column.h264.mp4) · [WingIt_dev_3586_3811](https://github.com/RedRoman/Tencent-streaming-sr/releases/download/h3-limiteddev-226f-20261108/WingIt_dev_3586_3811__comparison_3column.h264.mp4)。全部21条视频、原生1080单路与同位置局部对比见 [视频目录](VIDEO_LINKS.md)。链接采用最终Release tag；本页保存发布前核验快照，公开链接由发布步骤确认。**完整 native-HR L1 守门 FAILED，连续主观审片未完成，不能宣称整体画质改善。**

The teacher-arm checkpoint selected at step 2000 is a **research candidate that failed the complete native-HR L1 guard**. Both arms completed 2000 optimizer updates. The frozen sparse-dev rule selected control step 0 and teacher step 2000; the full226 evaluation did not select another checkpoint or relax a threshold. Keep the media label **PROVISIONAL / full native-HR guard FAILED**. This result does not establish better overall picture quality or an accepted best model.

Across the fixed three dev clips, native-HR L1 increased **+2.761089%**, exceeding the allowed +2%; PSNR changed **-0.172649 dB**. LPIPS decreased **4.508318%**, while raw latent eWarp increased **+15.568453%**. Teacher agreement improved, but agreement with the original H3 upsampler is a separate objective from fidelity to native HR.

## Scope and protocol

This is the authorized limited-data experiment: 2 training clips and 3 dev windows from 2 films. Two dev windows come from Wing It and are correlated. Results are arithmetic equal-clip means, including PSNR; they are not pooled-pixel scores or evidence of broad generalization. Formal 16-dev validation is not passed and test remains locked.

Both arms started from the same original v5 step1800 weights with fresh AdamW, fixed seed and paired schedule; the supervision target differs. Every student inference uses complete LR latent `[1,24,67,44,80]` (1280×704) and explicit target `[1,24,67,68,120]` (1920×1088). The frozen decoder output is cropped at rows `4:1084` to 1920×1080, with 226 frames at 24 fps (9.4166667 seconds). FP32, convolution/matmul HF32 disabled, highest matmul precision and strict deterministic mode are recorded. Upsampler causality is relative to fixed latent inputs; the VAE/decoder remains noncausal. The public synthetic degradation does not reproduce an unspecified private production pipeline.

## Complete-frame measurements

Control step0 and the initial v5 have identical recorded full226 metrics. The teacher reference is the original upstream bidirectional upsampler, not the distilled student; decoded-HR reconstruction and verified native-HR RGB are reported separately. Lower is better except PSNR. Global score is L1 + 0.2×LPIPS. The common motion ROI is the detached decoded-HR adjacent RGB change mask at channel-mean absolute change ≥0.05.

| Metric / reference | v5 / control step 0 | Teacher step 2000 | Change |
|---|---:|---:|---:|
| Original H3 teacher agreement: L1 ↓ | 0.013008021 | 0.009784407 | -24.781745% |
| Original H3 teacher agreement: LPIPS ↓ | 0.052354691 | 0.035436042 | -32.315442% |
| Original H3 teacher agreement: PSNR ↑ | 32.449567413 | 34.591387571 | +2.141820 dB |
| Original H3 teacher agreement: global score ↓ | 0.023478959 | 0.016871615 | -28.141556% |
| Original H3 teacher agreement: motion ROI L1 ↓ | 0.037807005 | 0.027425253 | -27.459862% |
| Original H3 teacher agreement: temporal delta L1 ↓ | 0.009303573 | 0.007584664 | -18.475791% |
| Decoded HR reconstruction: L1 ↓ | 0.017042369 | 0.016933593 | -0.638269% |
| Decoded HR reconstruction: LPIPS ↓ | 0.093464939 | 0.090261168 | -3.427778% |
| Decoded HR reconstruction: PSNR ↑ | 29.540344446 | 29.460676374 | -0.079668 dB |
| Decoded HR reconstruction: global score ↓ | 0.035735357 | 0.034985826 | -2.097447% |
| Decoded HR reconstruction: motion ROI L1 ↓ | 0.052693259 | 0.048614305 | -7.740942% |
| Decoded HR reconstruction: temporal delta L1 ↓ | 0.013170122 | 0.012952381 | -1.653292% |
| Native HR reference: L1 ↓ | 0.018659727 | 0.019174938 | +2.761089% |
| Native HR reference: LPIPS ↓ | 0.110457254 | 0.105477490 | -4.508318% |
| Native HR reference: PSNR ↑ | 29.306074569 | 29.133425438 | -0.172649 dB |
| Native HR reference: global score ↓ | 0.040751177 | 0.040270436 | -1.179699% |
| Native HR reference: motion ROI L1 ↓ | 0.053272023 | 0.049573096 | -6.943471% |
| Native HR reference: temporal delta L1 ↓ | 0.013505867 | 0.013346375 | -1.180914% |
| Raw latent eWarp L1 ↓ | 0.227205664 | 0.262578070 | +15.568453% |

The six teacher-agreement conditions passed. The decoded-HR global, L1 and LPIPS guards passed in the equal-clip aggregate. Native-HR global and LPIPS guards passed; native-HR L1 failed. Frozen guard multipliers remain global 1.01, L1 1.02 and LPIPS 1.02, with strict teacher improvement >1e-8. Passing a tolerated regression guard is not the same as improving every metric.

## Per-clip native-HR changes

| Dev window | L1 | PSNR Δ dB | LPIPS | ROI L1 | Delta L1 | All recorded conditions |
|---|---:|---:|---:|---:|---:|---|
| WingIt_dev_3586_3811 | -0.653728% | +0.050603 | -5.419281% | -3.194540% | -1.085132% | PASS |
| ElephantsDream_dev_13070_13295 | +8.303555% | -0.557453 | -0.726834% | +6.200727% | +3.817114% | FAIL |
| WingIt_dev_300_525 | +1.952796% | -0.011097 | -7.702842% | -21.123600% | -11.167753% | PASS |

Elephants Dream is the main observed regression: native-HR L1, PSNR, global, ROI and delta deteriorate despite improved teacher agreement. Its decoded-HR L1 and native-HR global/L1 guards fail. Both Wing windows pass their recorded guard conditions, but Wing300 native-HR L1 and raw latent eWarp still worsen. No clip or checkpoint is removed after seeing these results.

## Frame regions and temporal boundaries

| Native-HR region | Frames | L1 | PSNR Δ dB | LPIPS | ROI L1 | Delta L1 |
|---|---:|---:|---:|---:|---:|---:|
| full226 | 226 | +2.761089% | -0.172649 | -4.508318% | -6.943471% | -1.180914% |
| prefix208 | 208 | +2.539403% | -0.146951 | -4.601457% | -7.257807% | -1.533996% |
| tail18 | 18 | +5.479337% | -0.525646 | -3.466000% | +3.301921% | +3.086471% |
| blend | 60 | +1.752205% | -0.169680 | -5.034407% | -6.274921% | -1.796120% |
| nonblend | 166 | +3.137375% | -0.175539 | -4.323552% | -6.148548% | -1.001081% |

Full226 uses 225 adjacent RGB pairs and 66 latent flow pairs. Prefix208 and tail18 are reported separately; 166 nonblend frames and 60 blend frames remain distinct. Region-internal pair counts are 153 nonblend and 48 blend, with 72 blend-touching transitions also reported. Motion masks and exact counts accompany [the detailed CSV](full226_metrics.csv), [outcome JSON](full226_outcome.json) and [sanitized original summary](evidence/full226/pixels/summary.json).

## Evidence and verification

Training evidence binds all 4000 completed update records and 42 checkpoint records, with final logs and independent review; this publication contains compact binding/terminal evidence rather than the full training caches. Full evaluation reused 9 real full67 predictions and performed 15 decoder forwards, with no optimizer updates or checkpoint reselection. The outer pixels job exited 0 and reaped its child at 2026-11-08 07:32:13 UTC.

An independent audit checked 6102 stored per-frame scalar records, performed 981 scalar/consistency checks (675 region scalars + 225 equal-clip aggregate scalars + 81 partition checks) (maximum floating-point roundoff 7.11e-15), and verified aggregate/group/regression/condition logic. The raw per-pair contributions were not saved: ROI/delta have evidence-chain and aggregate checks, not an independently reproduced per-pair sum. [Audit evidence](evidence/full226/independent_pixels_audit_v1.json) retains this boundary.

Original summary SHA-256: `968b9e0342290f6c30b8f05117e780fc435404a5aa8b57608123015ffa05ff38`. Original pixels outer SHA-256: `1bf6d7939623ddfbd4ee9478db53f6995fbd0ff6abab29bd30d8d733e8ca7009`. Deployment paths are redacted in public copies, so original and public SHA values differ; [the mapping](original_public_mapping.json) records both. Hash strings embedded inside historical contracts still identify their original bytes.

## Videos and inference weights

All 21 H264 files are complete locally and passed the recorded full226 / 24fps / 9.4166667-second / PTS / decode / SHA checks (212,464,627 bytes total): 3 main comparisons, 3 fixed native-scale motion/detail crops and 15 native1920×1080 single routes. [Asset identities and final-tag download targets](media_asset_manifest.json), [checksums](media_checksums.sha256) and [technical receipts](evidence/media/full21_H264_delivery_v1.json) are provided. These files are external Release assets, not Git payloads. All three media transport packages and all 84 native-scale contact sheets are now complete locally; [contact identities and coverage](contact_inventory.json) and [transport closeout](evidence/media/media_owner_closeout_v1.json) record this separately from visual inspection. **The 21 complete videos and the original model-only weight have been uploaded to the same Release and verified by exact API size and SHA-256 digest. This is a frozen prepublication snapshot: Release 438270407 was still draft, Git push and final-tag URL verification remained pending.** [Video links](VIDEO_LINKS.md), [the Release target](https://github.com/RedRoman/Tencent-streaming-sr/releases/tag/h3-limiteddev-226f-20261108) and [weight download target](https://github.com/RedRoman/Tencent-streaming-sr/releases/download/h3-limiteddev-226f-20261108/h3_limiteddev_teacher_step2000_fp32.pt) use the final tag and become active after publication. Continuous subjective visual review remains incomplete. The weight was streamed from the complete remote source; the separate local SFTP copy remains incomplete at its recorded observation and was not used for upload. The original SCP remote-close failure remains preserved. The later uploader sent the full correct stream but its v1 verifier rejected GitHub's temporary draft URL; the GET-only v2 verifier accepted the existing asset with exact size/digest, without reupload. [Publication status](publication_status.json) and [verification evidence](evidence/publication/weight_asset_verified_v2.json) keep these events separate. Technical decode is not a subjective quality pass. All 226 PTS values were scanned in each H264 file, but the retained H264 audits contain counts/errors/extrema rather than raw per-frame PTS rows; the original MPEG4 audits retain all 4,746 PTS rows. Nominal duration is 226/24 = 9.4166667 seconds. Seven Wing300 containers report 9.4169921875 seconds, within the already-frozen ±0.001-second allowance; their last frame PTS is 9.375 seconds and cadence is 1/24 second. The manifest records actual container duration separately.

The main comparisons use 960×540 display panels per column (half native width and height), with a 112-pixel label header: 2880×652 total. Fixed detail videos use three 512×512 crops at native scale plus the same header (1536×624); they do not track a moving object. Native single-route files remain 1920×1080. The main video uses the required three columns: sparse-selected causal research candidate (round, step2000 and SHA, PROVISIONAL/FAILED label), original Open H3 bidirectional upsampler, and verified native-HR GT reference. All use the same frame indices and display scale; display downscale is labeled. The original Wing3586 teacher cache has its generation determinism policy unbound, while the other displayed teacher caches record strict determinism; all displayed teacher decoding is FP32/HF32-off strict mode. Historical cache policy is not relabeled by the later strict evaluation. Native 1920×1080 single routes and matching-position motion/detail crops are separate assets. GT is neither decoded-HR reconstruction nor teacher output.

The [inference package](../../weights/h3_limiteddev_20261108/README.md) is prepared for the unchanged model-only candidate, SHA-256 `7b9f43c17462fb2d0554645bc7889dc7abdebc052ee9f18f6780624de912e0b5`. It contains no optimizer/RNG state. Its CPU loader was actually validated on all 322 FP32 tensors; this is a packaging check, not an additional scientific forward or quality pass.

## Recorded static visual scope

Four actual displayed native-scale contact pages were inspected across the three dev clips. The initial Wing3586 local frame113 / source frame3699 cat face/hand crop at (1072,360,512,512) showed similar candidate/original outputs: both softened thin edges and showed colored fringes near goggles/ears and altered small mouth/eye details compared with native GT. Three later observations covered Elephants Dream local frame0 at (1024,284,512,512), local frame225 at (0,284,512,512), and Wing300 local frame142 / source frame442 at (704,284,512,512). The Elephants Dream crops showed softened pipe edges and smeared or uneven highlights in both output columns; the candidate also showed localized ringing/colored edge traces at frame0. The Wing300 crop showed soft window/panel outlines and colored fringes/halo in both outputs, with wall texture/brightness differing from GT.

These are four static observations, not inspection of all 84 pages or continuous playback. They do not establish temporal flicker, motion consistency, a whole-clip winner, a causal-model-specific mechanism, or that LPIPS caused the regression. [The initial record](evidence/media/root_static_visual_observation_v1.json) and [the three additional observations](evidence/media/root_additional_static_observation_v2.json) identify exact image SHAs, frame identities and crop positions. Continuous subjective visual review remains incomplete.

## Follow-up boundary

The [historical hypothesis review](evidence/followup/optimization_lpips01_hypothesis_review_v1.json) proposed one training-only change: teacher LPIPS coefficient 0.2→0.1, keeping other objectives, paired inputs/schedule, evaluation LPIPS coefficient 0.2 and guard thresholds unchanged. The new `limiteddev_lpips01_20261108_v1` run actually started: [the start verification](evidence/followup/root_started_verified_v1.json) confirmed 4 optimizer updates toward 2000 at 2026-11-08 12:30:03 UTC, and [the first-forward crosscheck](evidence/followup/first_actual_forward_crosscheck_v1.json) binds its initial comparison. This is a timestamped start observation, not a live counter or a completed result. The completed control/teacher baselines are reused.

The follow-up remains an unvalidated posthoc test on the same limited dev data. Training loss does not establish better picture quality; neither the numeric tradeoff nor four static pages proves that LPIPS caused the native-HR regression. Continuous visual review remains incomplete. No new full226 result, accepted best model, formal16dev/test success or artifact resolution is claimed.

Archived receipts are frozen prepublication evidence: historical `pending`, `paused`, `failed` or `not_uploaded` fields retain their original observation time. [Publication status](publication_status.json), the timestamped transfer snapshot and later verified upload records state the current delivery scope without rewriting earlier evidence.
