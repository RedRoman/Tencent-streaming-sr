# H3 limited-dev teacher step2000 research candidate

**Sparse-selected research candidate; complete native-HR L1 guard FAILED.** This package describes the unchanged model-only checkpoint from the 2026-11-07 limited-dev run. The selected candidate is not an accepted best model and is not recommended as a demonstrated picture-quality improvement. Native-HR L1 worsened +2.761089%; LPIPS improved, while PSNR and raw latent eWarp worsened. See [complete 226-frame results](../../results/h3_production67_limiteddev_20261108/README.md).

Expected asset: `h3_limiteddev_teacher_step2000_fp32.pt`, 1,381,220,283 bytes, SHA-256 `7b9f43c17462fb2d0554645bc7889dc7abdebc052ee9f18f6780624de912e0b5`. **The complete unchanged model-only weight is uploaded and API-verified in draft Release 438270407, asset 724063298, with exact size and SHA-256.** [Weight download target](https://github.com/RedRoman/Tencent-streaming-sr/releases/download/h3-limiteddev-226f-20261108/h3_limiteddev_teacher_step2000_fp32.pt) · [configuration target](https://github.com/RedRoman/Tencent-streaming-sr/releases/download/h3-limiteddev-226f-20261108/h3_limiteddev_teacher_step2000_config.json) · [checksums target](https://github.com/RedRoman/Tencent-streaming-sr/releases/download/h3-limiteddev-226f-20261108/weights_SHA256SUMS.txt) · [Release target](https://github.com/RedRoman/Tencent-streaming-sr/releases/tag/h3-limiteddev-226f-20261108). These final-tag links require publication; this frozen snapshot does not claim the draft is public or final URLs have already been verified. [The v2 receipt](../../results/h3_production67_limiteddev_20261108/evidence/publication/weight_asset_verified_v2.json) verifies the existing asset with GET-only checks. The complete remote source supplied the upload; the separate local SFTP copy remains incomplete at its preserved observation. The v1 upload sent all bytes with the correct stream SHA but rejected GitHub's draft untagged URL. Its original terminal evidence is retained; no weight reupload was needed. No large weight enters Git. The exact checkpoint keys are `schema`, `arm`, `step`, `protocol_sha256`, `initializer_checkpoint_sha256`, `metrics`, `model`; only 322 FP32 model tensors (345,280,216 parameters), numerical metrics and lineage identifiers are included. No optimizer or RNG state is published and no weight re-export is needed.

## Actual CPU load verification

The unchanged `load_h3_limiteddev_inference.py` and sidecar were actually exercised with PyTorch 2.7.1+cpu from 05:38:24 to 05:38:48 UTC on 2026-11-08. All 322 loaded tensor byte sequences matched the selected checkpoint exactly. The process exited 0, was reaped, imported no `torch_npu`, and executed zero model forwards, decoder calls or optimizer updates. [CPU receipt](cpu_load_evidence/load_receipt.json) and [independent review](review/independent_review.json) are included with original/public SHA mappings.

The frozen candidate sidecar intentionally retains its earlier `pending` full226-quality status; it is the exact configuration bound to the actual CPU receipt, SHA `ad1c03b0e59af4bfe30c57d20628647762575a44a0f4874a574bb3f5f36be706`. The later failed full226 quality conclusion is stated here and in `delivery_quality_status.json`. Do not interpret the historical sidecar's pending field as a quality pass.

## Geometry and use

LR `[1,24,67,44,80]` maps to explicit target `[1,24,67,68,120]`. Frozen downstream decode/crop gives 226 frames, 1920×1080, 24 fps; the loader itself does not include the decoder. The historical construction scale2 is not the target geometry. No old v5 weights, teacher weights or caches are needed to construct and load this candidate.

Provide the exact five sources in [source references](source_references.json), the existing published tensor-name/shape schema, PyTorch 2.7.1 and `einops`. Keep the upstream pair together and retain their source notices. Source references are explicit dependency pointers, not an automatic download or runtime installer.

```python
import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"  # before importing Torch
from load_h3_limiteddev_inference import load_cpu, forward67

student, receipt = load_cpu(
    config_path,
    config_sha256="ad1c03b0e59af4bfe30c57d20628647762575a44a0f4874a574bb3f5f36be706",
    checkpoint=selected_model_only_checkpoint,
    tensor_schema=published_v5_tensor_schema,
    source_paths={
        "causal_upscaler.py": causal_source,
        "h3_production67_geometry_v1.py": geometry_source,
        "h3_conv3d_explicit_target_v1.py": explicit_adapter_source,
        "upscaler.py": original_upstream_loader,
        "upscaler_node.py": original_upstream_node,
    },
)
# This loads on CPU and performs no forward. The caller must separately
# admit its device and set the recorded strict FP32 runtime before:
# output = forward67(student, full67_native_latent)
```

For NPU inference the caller owns device admission, exclusivity locks and backend initialization. Disable convolution/matmul HF32, request highest float32 precision, enable deterministic algorithms with `warn_only=False`, and use evaluation mode with gradients off. `forward67` validates the full67 shape and runtime mode and delegates to the original explicit-target adapter without a resize/cast fallback. No new NPU forward is claimed for the packaging entry point.

The tensor schema supplies names and shapes only; its historical v5 tensor-value hashes are not compared with the new checkpoint. The loader verifies full checkpoint SHA before `weights_only=True` loading, strict header/tensor schema, FP32 finiteness and bitwise tensor identity after wrapping. Earlier 19 metadata boundary checks did not load weights; the later actual CPU receipt supplies that evidence separately.

Only the student latent mapping is causal. The external VAE/decoder remains noncausal and is not included. Model/upstream notices are retained in `LICENSE`, `LICENSE.upstream-node-MIT` and `license_provenance.json`; external dependencies have their own terms.
