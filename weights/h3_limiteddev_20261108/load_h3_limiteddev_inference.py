"""Load the selected model-only H3 checkpoint using the reviewed explicit target.

Import is standard-library only. CPU loading performs no forward, decoder,
download, optimizer construction or device selection. Scientific sources are
provided separately and checked before import; no v5 checkpoint is required.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys
import types

WEIGHT_SHA = "7b9f43c17462fb2d0554645bc7889dc7abdebc052ee9f18f6780624de912e0b5"
WEIGHT_BYTES = 1381220283
TRAIN_PROTOCOL = "cbf2bdc35d854fd427169dda1cede22aea118a5cbaa5e0903a54d3ca2cf011e2"
INITIALIZER_SHA = "c7aaaa396d877a84c3fed2a5536189b7ed5e61226a82ef1a9e6be1e831f7f914"
TENSOR_SCHEMA_SHA = "071b3fd8b71dfde906b1f75f250d79c8d8df32f24b0c0c194b83387bf1705a75"
SOURCE_SHAS = {
    "causal_upscaler.py": "8032ebd29e558d2cad7f31cb91618022892270a5d4027052f251475b04f98677",
    "h3_production67_geometry_v1.py": "147765ef79549bc31fd2b9e5026d5983be2328ee12ee04743f9202543b4af383",
    "h3_conv3d_explicit_target_v1.py": "8c5bcbd1702a252266c68cbdf131e40fa7b56366deaeef17c53a9b84dbd5b3c6",
    "upscaler.py": "e54959b609c5732711fd8e87b5e6cbe4e80977379d988df175afa1c7745f4521",
    "upscaler_node.py": "f5945b500ddb38afee23c377034bf15394ba64748511a7e9608953040c4be437",
}
ADAPTER = dict(version=1, normalization="D(U(N(native_normalized_H3_latent)))",
               initialization="aligned_past", temporal_padding="left_only_zero",
               groupnorm="per_frame", spatial_scale=2.0, temporal_chunking=False,
               dtype="float32", saved_weights_already_initialized=True,
               reload_must_not_reinitialize_weights=True)
ARCHITECTURE = dict(in_channels=24, in_blocks=12, out_blocks=12, channels=512,
                    dropout=0.1, attn=False, temporal_every=2, temporal_kernel=5)
GEOMETRY = dict(input_bcthw=[1, 24, 67, 44, 80], target_bcthw=[1, 24, 67, 68, 120],
                raw_rgb_thwc=[226, 1088, 1920, 3], visible_crop_y0_y1_x0_x1=[4, 1084, 0, 1920],
                fps=[24, 1], historical_scale_is_not_current_geometry=True)
MODE = dict(dtype="float32", conv_hf32=False, matmul_hf32=False,
            float32_matmul_precision="highest", deterministic_algorithms=True, warn_only=False)
METRICS = dict(hr_pixel_l1=0.016354279903074104, hr_lpips=0.08203903088967006,
               teacher_pixel_l1=0.009580027932922045, teacher_lpips=0.03265833109617233,
               teacher_motion_l1=0.027070777490735054, teacher_delta_l1=0.006564391776919365,
               hr_global_score=0.03276208608100811, teacher_global_score=0.016111694152156512,
               selection_score=0.03095996125290791)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def read_json(path):
    path = Path(path)
    require(path.stat().st_size <= 1 << 20, "Unexpected JSON size")
    return json.loads(path.read_text(), object_pairs_hook=_json_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def fingerprint(path):
    path = Path(path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), "Require a regular file, not a symlink or partial substitute")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    after = path.lstat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "File changed during hashing")
    return {"bytes": after.st_size, "sha256": digest.hexdigest()}


def validate_config(config):
    require(config["schema"] == "h3-limiteddev-selected-inference-sidecar-v1", "Unknown sidecar schema")
    require(config["weights"] == dict(sha256=WEIGHT_SHA, bytes=WEIGHT_BYTES,
                                      schema="h3-production67-limiteddev-selected-weights-v1",
                                      arm="teacher", step=2000), "Wrong selected weight identity")
    require(type(config["weights"]["step"]) is int, "Step must be an integer")
    for name, expected in (("effective_geometry", GEOMETRY), ("runtime_mode", MODE),
                           ("historical_adapter", ADAPTER), ("architecture", ARCHITECTURE),
                           ("source_sha256", SOURCE_SHAS), ("selected_sparse_metrics", METRICS)):
        require(canonical(config[name]) == canonical(expected), "Sidecar contract differs: " + name)
    require(config["protocol_sha256"] == TRAIN_PROTOCOL and
            config["initializer_checkpoint_sha256"] == INITIALIZER_SHA and
            config["tensor_schema_sha256"] == TENSOR_SCHEMA_SHA, "Wrong lineage/schema pins")
    require(config["tensor_count"] == 322 and config["parameter_count"] == 345280216 and
            config["reinitialization_forbidden"] is True and config["inference_only"] is True,
            "Wrong model-only inference contract")


def validate_header(saved):
    require(type(saved) is dict and set(saved) ==
            {"schema", "arm", "step", "protocol_sha256", "initializer_checkpoint_sha256", "metrics", "model"},
            "Exact model-only payload required; extra optimizer/RNG/state is forbidden")
    require(saved["schema"] == "h3-production67-limiteddev-selected-weights-v1" and
            saved["arm"] == "teacher" and type(saved["step"]) is int and saved["step"] == 2000 and
            saved["protocol_sha256"] == TRAIN_PROTOCOL and
            saved["initializer_checkpoint_sha256"] == INITIALIZER_SHA, "Selected lineage differs")
    require(canonical(saved["metrics"]) == canonical(METRICS), "Selected sparse metrics differ")


def load_schema(path):
    require(fingerprint(path)["sha256"] == TENSOR_SCHEMA_SHA, "Original tensor schema SHA differs")
    document = read_json(path)
    require(document["tensor_count"] == 322 and document["parameter_count"] == 345280216,
            "Wrong tensor schema totals")
    schema = {name: row["shape"] for name, row in document["checks"].items()}
    require(len(schema) == 322 and sum(math.prod(s) for s in schema.values()) == 345280216,
            "Incomplete tensor schema")
    # The historical tensor VALUE hashes are deliberately not used for the new model.
    return schema


def verify_sources(paths):
    require(set(paths) == set(SOURCE_SHAS), "Supply exactly the five recorded scientific sources")
    for name, expected in SOURCE_SHAS.items():
        require(fingerprint(paths[name])["sha256"] == expected, "Scientific source differs: " + name)
    require(Path(paths["upscaler.py"]).resolve().parent == Path(paths["upscaler_node.py"]).resolve().parent,
            "Upstream source pair must share its original directory")


def _module(name, path):
    if name in sys.modules:
        existing = sys.modules[name]
        require(existing is not None and Path(existing.__file__).resolve() == Path(path).resolve(),
                "Conflicting pre-imported module: " + name)
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def _check_tensors(state, schema, torch):
    require(isinstance(state, dict) and set(state) == set(schema), "Exact 322 tensor names required")
    for name, shape in schema.items():
        value = state[name]
        require(isinstance(value, torch.Tensor) and value.layout == torch.strided and
                value.device.type == "cpu" and value.dtype == torch.float32 and
                list(value.shape) == shape and bool(torch.isfinite(value).all()),
                "Wrong tensor shape/FP32/CPU/finite value: " + name)


def load_cpu(config_path, *, config_sha256, checkpoint, tensor_schema, source_paths):
    """Return (reviewed explicit-target student, CPU-only load receipt).

    No old v5 checkpoint, teacher weight, decoder or device runtime is loaded.
    The caller provides paths; this function never fetches external assets.
    """
    require(fingerprint(config_path)["sha256"] == config_sha256, "Sidecar SHA differs")
    config = read_json(config_path)
    validate_config(config)
    verify_sources(source_paths)
    schema = load_schema(tensor_schema)
    expected = {"bytes": WEIGHT_BYTES, "sha256": WEIGHT_SHA}
    require(fingerprint(checkpoint) == expected, "Complete selected checkpoint SHA/size differs")
    require(os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0", "Disable device backend autoload before loading")
    import torch
    require(torch.get_default_dtype() == torch.float32, "Construct the original architecture in FP32")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    validate_header(saved)
    state = saved["model"]
    _check_tensors(state, schema, torch)
    _module("causal_upscaler", source_paths["causal_upscaler.py"])
    geometry = _module("h3_production67_geometry_v1", source_paths["h3_production67_geometry_v1.py"])
    wrapper = _module("h3_conv3d_explicit_target_v1", source_paths["h3_conv3d_explicit_target_v1.py"])
    upstream = _module("h3_limiteddev_release_upstream", source_paths["upscaler.py"])
    # Reuse the original standalone node loader, restoring its ComfyUI shims.
    names = ("folder_paths", "comfy.model_management", "comfy_api.latest")
    absent = object()
    previous = {name: sys.modules.get(name, absent) for name in names}
    try:
        node = upstream.load(checkpoint)
    finally:
        for name, value in previous.items():
            if value is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    cfg = node._detect_arch({name: types.SimpleNamespace(shape=tuple(shape)) for name, shape in schema.items()})
    require(cfg == ARCHITECTURE, "Upstream architecture differs")
    with torch.device("cpu"):
        model = node.LatentResizer3D(**cfg)
    require(sum(p.numel() for p in model.parameters()) == 345280216, "Wrong parameter count")
    model.load_state_dict(state, strict=True, assign=False)
    # Already-aligned provenance only: do not invoke aligned-past initialization.
    model._aligned_past_initialized = True
    student = wrapper.ExplicitTargetCausalStudent(node, model, ADAPTER,
                geometry=geometry.production67(), source_checkpoint_sha256=WEIGHT_SHA)
    student.eval().requires_grad_(False)
    actual = student.original_state_dict()
    _check_tensors(actual, schema, torch)
    # Byte views additionally distinguish signed zero and preserve exact FP32 bits.
    require(all(torch.equal(actual[name].contiguous().view(torch.uint8),
                            state[name].contiguous().view(torch.uint8)) for name in schema),
            "Strict load/causal wrapper changed tensor bits")
    require(fingerprint(checkpoint) == expected, "Checkpoint changed during CPU load")
    verify_sources(source_paths)
    require(fingerprint(config_path)["sha256"] == config_sha256, "Sidecar changed during CPU load")
    return student, dict(status="cpu_load_and_322_tensor_bitwise_checks_passed", weights=expected,
        config_sha256=config_sha256, actual_tensor_count=322, actual_parameter_count=345280216,
        explicit_target=[67, 68, 120], aligned_initialization_called=False,
        optimizer_or_rng_loaded=False, actual_forward_count=0, actual_decoder_count=0,
        actual_device_runtime_loaded_by_loader=False, source_sha256=SOURCE_SHAS)


def forward67(student, native):
    """Run the existing full67 reviewed path; caller owns device admission/locks."""
    import torch
    require(torch.are_deterministic_algorithms_enabled() and
            not torch.is_deterministic_algorithms_warn_only_enabled() and
            torch.get_float32_matmul_precision() == "highest", "Require strict deterministic/highest mode")
    if native.device.type == "npu":
        require(torch.npu.conv.allow_hf32 is False and torch.npu.matmul.allow_hf32 is False,
                "NPU HF32 must be disabled explicitly by the caller")
    require(native.device.type in ("cpu", "npu"), "Only reviewed CPU/NPU routing is supported")
    require(not student.training and all(not p.requires_grad for p in student.parameters()), "Inference-only model required")
    with torch.inference_mode():
        return student.forward_full(native, target_size=(67, 68, 120))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tensor-schema", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: (args.upstream_dir if name in ("upscaler.py", "upscaler_node.py") else args.runtime_dir) / name
             for name in SOURCE_SHAS}
    _, receipt = load_cpu(args.config, config_sha256=args.config_sha256, checkpoint=args.checkpoint,
                           tensor_schema=args.tensor_schema, source_paths=paths)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
