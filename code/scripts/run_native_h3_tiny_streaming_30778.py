"""Normal installed Diffusers H3 tiny-model CPU integration; no pretrained weights.

Run under an external hard deadline, e.g. timeout -k 5s 120s .venv/bin/python -B
scripts/run_native_h3_tiny_streaming_30778.py. This script never initializes an NPU.
The stock model forward is run before wrapping, then the explicit LR streaming
adapter runs the real native submodules. No AST extraction or package stubs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback
import warnings

DEFAULT_PROJECT = Path("/home/ma-user/modelarts/user-job-dir/sol_h3")
DEFAULT_SOURCE = DEFAULT_PROJECT/"software/engineering-current-3aaeba66608c5bee"


def file_identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def run(args, report):
    # Import registration is allowed; autoload and all compute remain explicitly CPU.
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    import torch_npu  # official backend registration only; no set_device/init
    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock
    report["packages"] = {name: importlib.metadata.version(name) for name in
                          ("torch", "torch-npu", "diffusers", "transformers", "safetensors")}
    report["torch_runtime_version"] = torch.__version__
    npu_checks = []

    def check_no_npu(label):
        state = bool(torch.npu.is_initialized())
        npu_checks.append({"stage": label, "initialized": state})
        if state:
            raise RuntimeError(f"NPU unexpectedly initialized at {label}; stop CPU test")

    check_no_npu("after_official_torch_npu_and_native_class_import")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")
    # Seed only the CPU default generator; do not queue accelerator seed callbacks.
    torch.default_generator.manual_seed(20261008)
    if torch.is_autocast_enabled("cpu"):
        raise RuntimeError("CPU autocast must be disabled for the native precision test")
    source = args.source_root.resolve()
    required = ("experiments/runtime/src/__init__.py", "experiments/dit_condition/sampler.py",
                "experiments/streaming_lr/src/models/softmax_attention/streaming_lr.py")
    for relative in required:
        if not (source/relative).is_file():
            raise FileNotFoundError(f"staged engineering source missing: {relative}")
    if "src" in sys.modules:
        raise RuntimeError("src already imported before selecting the ordinary runtime package route")
    sys.path.insert(0, str(source/"experiments/runtime"))
    sys.path.insert(0, str(source))
    from src.models.hybrid_transform import apply_hybrid_attention_transform
    from src.models.hybrid_attention import HybridAttention, enable_causal_hybrid_attention
    from src.models.linear_attention.branch import enable_forward_causal_linear
    from src.models.softmax_attention.streaming_lr import enable_streaming_lr_attention, prepare_lr_latent_condition
    from experiments.dit_condition import sampler
    report["module_sources"] = {
        "installed_native_transformer": file_identity(inspect.getfile(MiniMaxH3Transformer3DModel)),
        "ordinary_runtime_HybridAttention": file_identity(inspect.getfile(HybridAttention)),
        "streaming_sampler": file_identity(sampler.__file__),
    }
    config = {"num_attention_heads": 2, "attention_head_dim": 16, "hidden_size": 32,
              "num_layers": 2, "num_refiner_layers": 1, "ffn_dim": 64,
              "in_channels": 24, "audio_in_channels": 32, "patch_size": (1, 2, 2),
              "text_dim": 16, "freq_dim": 32, "time_embed_hidden_dim": 32,
              "time_embed_dim": 16, "rope_freq_dim": 2, "rope_theta": 10000.,
              "norm_eps": 1e-5, "qk_norm_eps": 1e-5, "final_norm_eps": 1e-5}
    report["tiny_native_config"] = config
    report["native_modality_tags"] = {"video": 0, "text": 1, "audio": 2}
    observations = {}
    phase = ["native_forward_before_transform"]
    handles = []

    def record(name):
        def hook(module, inputs):
            tensor = inputs[0]
            value = {"dtype": str(tensor.dtype), "device": str(tensor.device), "shape": list(tensor.shape)}
            key = (name, phase[0])
            bucket = observations.setdefault(key, {"calls": 0, "inputs": []})
            bucket["calls"] += 1
            if value not in bucket["inputs"]:
                bucket["inputs"].append(value)
        return hook

    with torch.device("cpu"):
        model = MiniMaxH3Transformer3DModel(**config).to(device="cpu", dtype=torch.bfloat16).eval()
        # Match the real native _keep_in_fp32_modules, including audio heads and RoPE.
        for name in ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out", "rope"):
            getattr(model, name).to(device="cpu", dtype=torch.float32)
        if type(model) is not MiniMaxH3Transformer3DModel or any(type(b) is not MiniMaxH3TransformerBlock for b in model.transformer_blocks):
            raise RuntimeError("actual installed native classes are required")
        if len(model.token_refiner.refiner_blocks) != 1:
            raise RuntimeError("tiny model must retain an actual native token-refiner block")
        for name in ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out", "context_embedder", "token_refiner"):
            handles.append(getattr(model, name).register_forward_pre_hook(record(name)))
        for i, block in enumerate(model.transformer_blocks):
            for name, module in (("native_block", block), ("norm1", block.norm1), ("norm2", block.norm2),
                                 ("adaln_proj", block.adaln_proj), ("ff", block.ff), ("qkv", block.attn.to_q)):
                handles.append(module.register_forward_pre_hook(record(f"block{i}.{name}")))
        prompt = torch.randn((1, 3, config["text_dim"]), device="cpu", dtype=torch.bfloat16)
        hr_noise = torch.randn((1, 24, 2, 4, 4), device="cpu", dtype=torch.bfloat16)
        lr_latents = torch.randn((1, 24, 2, 2, 2), device="cpu", dtype=torch.float32)
        text_rows_count, frames, tokens_per_frame = prompt.shape[1], 2, 4
        source_clocks = (0, 4)
        positions = torch.zeros((text_rows_count+frames*tokens_per_frame, 3), device="cpu", dtype=torch.float32)
        for frame in range(frames):
            for h in range(2):
                for w in range(2):
                    positions[text_rows_count+frame*tokens_per_frame+h*2+w] = torch.tensor((frame, h, w), device="cpu", dtype=torch.float32)
        lr_positions = torch.tensor(((0, 0, 0), (1, 0, 0)), device="cpu", dtype=torch.float32)
        # Prove the real framework model and block .forward paths work before wrapping.
        with warnings.catch_warnings(record=True) as native_warnings, torch.no_grad():
            warnings.simplefilter("always")
            native_output = model(hidden_states=sampler.patchify(hr_noise),
                audio_hidden_states=torch.zeros((1, 0, config["audio_in_channels"]), device="cpu", dtype=torch.float32),
                encoder_hidden_states=prompt,
                timestep=torch.tensor((1., .5), device="cpu", dtype=torch.float32),
                timestep_indices=torch.cat((torch.zeros(text_rows_count, device="cpu", dtype=torch.long),
                                             torch.ones(frames*tokens_per_frame, device="cpu", dtype=torch.long))),
                token_tags=torch.cat((torch.ones(text_rows_count, device="cpu", dtype=torch.long),
                                      torch.zeros(frames*tokens_per_frame, device="cpu", dtype=torch.long))),
                position_ids=positions,
                video_indices=torch.arange(text_rows_count, len(positions), device="cpu", dtype=torch.long),
                audio_indices=torch.empty((0,), device="cpu", dtype=torch.long),
                text_indices=torch.arange(text_rows_count, device="cpu", dtype=torch.long),
                return_dict=True)
        if list(native_output.sample.shape) != [1, 8, 96] or native_output.sample.dtype != torch.float32 or not torch.isfinite(native_output.sample).all():
            raise RuntimeError("native unwrapped model forward shape/dtype/finite check failed")
        report["unwrapped_native_forward"] = {"passed": True, "video_shape": list(native_output.sample.shape),
            "video_dtype": str(native_output.sample.dtype), "audio_shape": list(native_output.audio_sample.shape),
            "warnings": [str(w.message) for w in native_warnings],
            "source": "actual installed MiniMaxH3Transformer3DModel.forward and MiniMaxH3TransformerBlock.forward; full attention, before causal/LR wrapping"}
        check_no_npu("after_native_unwrapped_forward")
        transform = {"anchor_frames": "none", "enable_softmax_gate": True,
                     "softmax_attention": {"radius": 1, "chunk": 0},
                     "linear_attention": {"delta_rule": "vdn_solve", "linear_head_dim": 16,
                        "bridge": "alpha", "a_fp32": True, "enable_text_state": False,
                        "short_conv": {"targets": ["k", "v"]}}}
        hybrids = apply_hybrid_attention_transform(model, transform)
        for hybrid in hybrids:
            hybrid.to(device="cpu", dtype=torch.bfloat16)
            # Official transform defaults to its training backend. The causal
            # streaming prototype deliberately accepts the eager ref path only.
            hybrid.softmax_impl = "ref"
        enable_forward_causal_linear(model, 1)
        enable_causal_hybrid_attention(model, 1, globals_policy="text_only_exogenous")
        enable_streaming_lr_attention(model, condition_history_frames=1)
        phase[0] = "native_text_refiner_for_streaming"
        with torch.no_grad():
            text_rows = model.token_refiner(model.context_embedder(prompt.to(model.context_embedder.weight.dtype)))
            full_rope = model.rope(positions)
            lr_rope = model.rope(lr_positions)
        if text_rows.dtype != torch.bfloat16 or any(p.device.type != "cpu" for p in model.parameters()):
            raise RuntimeError("native refined text and every model parameter must remain CPU/BF16 where intended")

        def rollout(parts, *, raw_noise=hr_noise, lowres=lr_latents, condition_enabled=True, label="streaming"):
            phase[0] = label
            state, begin, outputs = None, 0, []
            for count in parts:
                end = begin+count
                ids = list(range(begin, end))
                selected = torch.cat((torch.arange(text_rows_count, device="cpu"),
                    torch.arange(text_rows_count+begin*tokens_per_frame, text_rows_count+end*tokens_per_frame, device="cpu")))
                condition = None
                if condition_enabled:
                    condition = prepare_lr_latent_condition(lowres[:, :, ids], model.proj_in,
                        frame_ids=ids, dependency_end=[source_clocks[i] for i in ids],
                        available_at=[source_clocks[i] for i in ids],
                        source_id="tiny_random_LR_fixture_not_real_H3_producer", producer_causal_verified=True,
                        rotary=tuple(values[ids] for values in lr_rope))
                output, state = sampler.sample_chunk(model, raw_noise[:, :, begin:end], text_rows,
                    condition=condition, source_cutoffs=[source_clocks[i] for i in ids],
                    release_watermarks=[source_clocks[i] for i in ids],
                    stream_id="installed_native_tiny_framework_cpu_test", num_steps=2, shift=12.,
                    video_tag=0, text_tags=torch.ones(text_rows_count, device="cpu", dtype=torch.long),
                    frame_start=begin, state=state, rotary_emb=tuple(values[selected] for values in full_rope))
                if output.device.type != "cpu" or not torch.isfinite(output).all():
                    raise RuntimeError("sampler produced invalid or non-CPU output")
                outputs.append(output)
                begin = end
            if begin != frames:
                raise RuntimeError("test partition must cover exactly two latent frames")
            return torch.cat(outputs, dim=2), state

        with warnings.catch_warnings(record=True) as stream_warnings, torch.no_grad():
            warnings.simplefilter("always")
            if torch.is_autocast_enabled("cpu"):
                raise RuntimeError("unexpected CPU autocast before streaming tests")
            whole, whole_state = rollout([2], label="whole_two_frame_chunk")
            streamed, stream_state = rollout([1, 1], label="two_single_frame_chunks")
            altered_lr, altered_hr = lr_latents.clone(), hr_noise.clone()
            altered_lr[:, :, 1:] += 10
            altered_hr[:, :, 1:] += 10
            future, _ = rollout([1, 1], raw_noise=altered_hr, lowres=altered_lr, label="future_LR_and_HR_perturbed")
            lr_only_future, _ = rollout([1, 1], lowres=altered_lr, label="future_LR_only_perturbed")
            changed_past_lr = lr_latents.clone()
            changed_past_lr[:, :, :1] += 3
            changed_past, _ = rollout([1, 1], lowres=changed_past_lr, label="past_LR_condition_perturbed")
            unconditional, _ = rollout([1, 1], condition_enabled=False, label="condition_none_control")
        partition_error = (whole-streamed).abs().max().item()
        future_prefix_error = (future[:, :, :1]-streamed[:, :, :1]).abs().max().item()
        future_lr_prefix_error = (lr_only_future[:, :, :1]-streamed[:, :, :1]).abs().max().item()
        condition_effect = (unconditional-streamed).abs().max().item()
        past_lr_effect = (changed_past-streamed).abs().max().item()
        checks = {"chunk_partition_equivalence": bool(torch.allclose(whole, streamed, atol=args.atol, rtol=args.rtol)),
                  "future_target_and_LR_prefix_invariant": future_prefix_error == 0,
                  "future_LR_only_prefix_invariant": future_lr_prefix_error == 0,
                  "nonempty_LR_condition_changes_output": condition_effect > 1e-5,
                  "past_LR_latent_changes_output": past_lr_effect > 1e-5,
                  "output_is_FP32_normalized_latent": streamed.dtype == torch.float32,
                  "autocast_disabled": not torch.is_autocast_enabled("cpu")}
        expected = {"proj_in": torch.float32, "time_embedder": torch.float32, "proj_out": torch.float32,
                    "context_embedder": torch.bfloat16, "token_refiner": torch.bfloat16}
        for i in range(2):
            for name in ("norm1", "norm2", "ff", "qkv"):
                expected[f"block{i}.{name}"] = torch.bfloat16
            expected[f"block{i}.adaln_proj"] = torch.float32
        for name, dtype in expected.items():
            seen = [value for (key, stage), bucket in observations.items() if key == name for value in bucket["inputs"]]
            checks["dtype_inputs_"+name] = bool(seen) and all(value["dtype"] == str(dtype) and value["device"] == "cpu" for value in seen)
        cache_records = []
        for step_index, step in enumerate(stream_state.step_caches):
            checks[f"cache_step{step_index}_temb_FP32"] = step.temb.dtype == torch.float32
            for layer, cache in enumerate(step.blocks):
                record_value = {"step": step_index, "layer": layer, "next_frame": cache.next_frame,
                    "target_key_shape": list(cache.video_key.shape), "LR_key_shape": list(cache.lr_key.shape),
                    "target_key_dtype": str(cache.video_key.dtype), "LR_key_dtype": str(cache.lr_key.dtype),
                    "LR_frame_ids_per_token": list(cache.lr_ids), "last_cutoff": cache.last_cutoff,
                    "last_release_watermark": cache.last_watermark, "context": list(cache.context)}
                cache_records.append(record_value)
                checks[f"cache_step{step_index}_layer{layer}_BF16"] = cache.video_key.dtype == cache.lr_key.dtype == torch.bfloat16
                checks[f"cache_step{step_index}_layer{layer}_history_bound"] = cache.video_key.shape[0] == 1 and sorted(set(cache.lr_ids)) == [1]
        report["streaming_results"] = {"checks": checks, "passed": all(checks.values()),
            "max_partition_error": partition_error, "future_prefix_error": future_prefix_error,
            "future_LR_only_prefix_error": future_lr_prefix_error, "condition_effect_max": condition_effect,
            "past_LR_effect_max": past_lr_effect, "tolerance": {"atol": args.atol, "rtol": args.rtol},
            "partitions": [[2], [1, 1]], "num_steps": 2, "output_shape": list(streamed.shape),
            "LR_shape": list(lr_latents.shape), "LR_clean_native_time": 1.,
            "warnings": [str(w.message) for w in stream_warnings], "cache_records": cache_records}
        report["forward_observations"] = [{"module": name, "phase": stage, **value}
                                           for (name, stage), value in sorted(observations.items())]
        report["parameter_dtypes"] = {name: str(getattr(model, name).weight.dtype) for name in
                                      ("proj_in", "audio_proj_in", "proj_out", "audio_proj_out", "context_embedder")}
        report["parameter_dtypes"].update(time_embedder=str(next(model.time_embedder.parameters()).dtype),
            block0_adaln=str(model.transformer_blocks[0].adaln_proj.linear.weight.dtype),
            norm_out_linear=str(model.norm_out.linear.weight.dtype), rope_inv_freq=str(model.rope.inv_freq.dtype))
        report["tiny_parameter_bytes"] = sum(p.numel()*p.element_size() for p in model.parameters())
        report["framework_paths_executed"] = {
            "stock_model_forward": "actual installed class .forward before wrapping, including both native block .forward calls",
            "streaming_DiT_driver": "existing experiments.dit_condition.sampler.sample_chunk -> adapter.run_blocks; explicit block orchestration",
            "streaming_native_submodules": "installed native proj_in/context_embedder/token_refiner/time_proj/time_embedder/AdaLN/norm/FF/norm_out/proj_out .forward",
            "streaming_attention": "actual ordinary-package HybridAttention with causal KV5tap, forward scan, guarded causal Softmax and immutable LR KV cache",
            "streaming_native_block_forward_called": False,
            "AST_extraction_or_package_stubs": False,
            "production_H3_LR_producer_certified": False}
        check_no_npu("after_all_native_CPU_streaming_tests")
        report["NPU_initialization_checks"] = npu_checks
        report["passed"] = all(checks.values())
    for handle in handles:
        handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    args = parser.parse_args()
    output = args.output or args.project_root/"results/native_h3_tiny_streaming_cpu_20261008.json"
    if output.exists():
        parser.error("output already exists; use a fresh result path")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "actual installed Diffusers H3 tiny random-weight model, CPU integration only",
        "device": "cpu", "source_root": str(args.source_root),
        "checkpoint_loaded": False, "NPU_compute": False, "visual_demo_created": False,
        "weights": "tiny random initialized weights; not official pretrained checkpoint",
        "limits": ["Framework/interface/cache evidence only; not pretrained VSR quality or full-depth capacity.",
                   "LR latent stimuli and provenance/release clocks are fixtures; not generated by a certified causal H3 producer.",
                   "No pixel decoder or output media is produced."]}
    try:
        run(args, report)
    except Exception as exc:
        report.update(passed=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        if "torch" in sys.modules and hasattr(sys.modules["torch"], "npu"):
            report["NPU_initialized_on_error"] = bool(sys.modules["torch"].npu.is_initialized())
    report["wall_seconds"] = time.monotonic()-started
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report.get(key) for key in ("passed", "packages", "tiny_parameter_bytes",
        "unwrapped_native_forward", "streaming_results", "NPU_initialization_checks", "error", "wall_seconds")}, indent=2))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
