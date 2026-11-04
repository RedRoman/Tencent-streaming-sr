"""Full-weight H3 refiner causality checks and two frozen full-clip rollouts.

Run with torchrun (normally two authorized NPUs). No optimization occurs.
Causality is conditional on a fixed upscaled latent and fixed full text/silence
audio; this does not test the upstream VAE/upscaler or causal audio encoding.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401; registers the NPU backend

from model_runtime import Runtime
from temporal import self_forcing_rollout


SEED = 20261104
NOISE_SEED = SEED + 991
FRAMES = 13
BLOCK_FRAMES = 5
SIGMAS = (0.909375, 0.3)


def difference(left, right):
    left, right = left.detach().cpu(), right.detach().cpu()
    if left.shape != right.shape:
        raise ValueError(f"Comparison shape mismatch: {left.shape} != {right.shape}")
    if not (torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise FloatingPointError("Nonfinite comparison tensor")
    delta = left.double() - right.double()
    return {"max_abs": float(delta.abs().max()),
            "rms": float(delta.square().mean().sqrt()),
            "bit_exact": bool(torch.equal(left, right))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get(
        "EXP_ROOT", ".")))
    parser.add_argument("--output", type=Path, help="Fresh directory; default ROOT/refiner_causality_probe")
    parser.add_argument("--generator-checkpoint", type=Path,
                        help="Optional saved adapters.pt; otherwise bypass G/F adapters (frozen real role)")
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--strict-frame", action="store_true",
                        help="Also test one-latent-frame causality and run 13 single-frame blocks at sigma=0.3")
    args = parser.parse_args()
    if args.adapter_rank < 1 or not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("adapter-rank must be positive and tolerance finite/nonnegative")
    root = args.root
    output = args.output or root / "refiner_causality_probe"
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")
    dist.init_process_group("hccl")
    started = time.perf_counter()
    try:
        # Check existence on rank 0, then communicate before any rank creates it.
        exists = torch.tensor([int(output.exists()) if rank == 0 else 0], device=device)
        dist.broadcast(exists, src=0)
        if exists.item():
            raise FileExistsError(f"Use a fresh output directory: {output}")
        if rank == 0:
            output.mkdir(parents=True)
        dist.barrier()
        torch.manual_seed(SEED)
        torch.npu.manual_seed_all(SEED)
        runtime = Runtime(root / "merged_transformer", root / "assets", rank, world,
                          adapter_rank=args.adapter_rank, block_frames=BLOCK_FRAMES)
        runtime.model.eval()
        role = "real"
        if args.generator_checkpoint:
            checkpoint = torch.load(args.generator_checkpoint, map_location="cpu", weights_only=False)
            if checkpoint.get("args", {}).get("adapter_rank", args.adapter_rank) != args.adapter_rank:
                raise ValueError("Checkpoint adapter rank differs from --adapter-rank")
            runtime.load_adapter_state(checkpoint["adapters"])
            del checkpoint
            role = "generator"
        runtime.set_role(role)
        manifest = root / "data/holdout.jsonl"
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if len(rows) != 1:
            raise ValueError("This minimal probe requires exactly one held-out sample")
        sample_path = Path(rows[0]["path"])
        sample_path = sample_path if sample_path.is_absolute() else manifest.parent / sample_path
        sample = torch.load(sample_path, map_location="cpu", weights_only=False)
        source = sample["upscaled"].to(device).float()
        if tuple(source.shape) != (24, FRAMES, 24, 42) or not torch.isfinite(source).all():
            raise ValueError("Expected finite held-out upscaled latent [24,13,24,42]")
        if tuple(sample["audio_latents"].shape) != (2, 32, 72):
            raise ValueError("Expected unchanged full 43-frame encoded-silence audio [2,32,72]")
        noise = torch.randn(source.shape, generator=torch.Generator().manual_seed(NOISE_SEED)).to(device)
        records, outputs = [], {}

        def save(label, value, record):
            if rank == 0:
                torch.save({"latent": value.detach().float().cpu(),
                            "metadata": {**sample["metadata"], "decoded_frames": 43, "fps": 24.},
                            "experiment": record}, output / f"{label}.pt")

        def forward(label, state, *, causal, sigma=SIGMAS[0]):
            before = time.perf_counter()
            value = state + sigma * runtime.forward(
                state, sample, sigma, causal=causal, block_index=0)
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Nonfinite result: {label}")
            torch.npu.synchronize()
            records.append({"label": label, "frames": state.shape[1], "causal": causal,
                            "block_frames": runtime.block_frames, "sigma": sigma,
                            "seconds": time.perf_counter() - before})
            return value

        with torch.no_grad():
            state = (1 - SIGMAS[0]) * source + SIGMAS[0] * noise
            poisoned = state.clone()
            future = torch.randn(state[:, BLOCK_FRAMES:].shape,
                                 generator=torch.Generator().manual_seed(NOISE_SEED + 1)).to(device)
            poisoned[:, BLOCK_FRAMES:] = 3 * future
            assert torch.equal(state[:, :BLOCK_FRAMES], poisoned[:, :BLOCK_FRAMES])
            future_input_change = difference(state[:, BLOCK_FRAMES:], poisoned[:, BLOCK_FRAMES:])
            assert future_input_change["max_abs"] > 0
            causal_full = forward("causal13", state, causal=True)
            causal_poison = forward("causal13_future_perturbed", poisoned, causal=True)
            dense_full = forward("dense13", state, causal=False)
            dense_poison = forward("dense13_future_perturbed", poisoned, causal=False)
            causal_prefix = forward("causal5", state[:, :BLOCK_FRAMES].contiguous(), causal=True)
            checks = {
                "future_perturbation_causal_first5": difference(
                    causal_full[:, :BLOCK_FRAMES], causal_poison[:, :BLOCK_FRAMES]),
                "future_perturbation_dense_first5": difference(
                    dense_full[:, :BLOCK_FRAMES], dense_poison[:, :BLOCK_FRAMES]),
                "prefix5_vs_full13_causal_first5": difference(causal_prefix, causal_full[:, :BLOCK_FRAMES]),
            }
            checks["future_perturbation_causal_first5"]["passed"] = (
                checks["future_perturbation_causal_first5"]["max_abs"] <= args.tolerance)
            checks["future_perturbation_dense_first5"]["passed"] = (
                checks["future_perturbation_dense_first5"]["max_abs"] > args.tolerance)
            checks["prefix5_vs_full13_causal_first5"]["passed"] = (
                checks["prefix5_vs_full13_causal_first5"]["max_abs"] <= args.tolerance)
            if rank == 0:
                print(json.dumps({"event": "causality_checks", "checks": checks}), flush=True)
            save("dense_one_sigma0909375", dense_full, {"role": role, "sigma": SIGMAS[0], "causal": False})

            rollouts = {}
            for sigma in SIGMAS:
                label = "causal_rollout_sigma0909375" if sigma == SIGMAS[0] else "causal_rollout_sigma03"
                before = time.perf_counter()
                result = self_forcing_rollout(
                    source, sigma, BLOCK_FRAMES,
                    lambda prefix, **kw: runtime.forward(prefix, sample, causal=True, **kw),
                    noise=noise, grad_blocks="none")
                if not torch.isfinite(result.latent).all():
                    raise FloatingPointError(f"Nonfinite rollout: {label}")
                torch.npu.synchronize()
                record = {"role": role, "sigma": sigma, "seconds": time.perf_counter() - before,
                          "audit": result.audit, "latent_path": f"{label}.pt"}
                save(label, result.latent, record)
                rollouts[label] = record
                outputs[label] = result.latent.cpu()
                if sigma == SIGMAS[0]:
                    replay = difference(result.latent[:, :BLOCK_FRAMES], causal_prefix)
                    replay["passed"] = replay["max_abs"] <= args.tolerance
                    checks["rollout_first_block_replay"] = replay
                if rank == 0:
                    print(json.dumps({"event": "rollout_completed", **record}), flush=True)

            if args.strict_frame:
                runtime.block_frames = 1
                sigma = SIGMAS[1]
                strict_state = (1 - sigma) * source + sigma * noise
                strict_poison = strict_state.clone()
                strict_future = torch.randn(strict_state[:, 1:].shape,
                    generator=torch.Generator().manual_seed(NOISE_SEED + 2)).to(device)
                strict_poison[:, 1:] = 3 * strict_future
                assert torch.equal(strict_state[:, :1], strict_poison[:, :1])
                strict_full = forward("strict_frame13", strict_state, causal=True, sigma=sigma)
                strict_changed = forward("strict_frame13_future_perturbed", strict_poison,
                                         causal=True, sigma=sigma)
                value = difference(strict_full[:, :1], strict_changed[:, :1])
                value.update({"passed": value["max_abs"] <= args.tolerance,
                              "sigma": sigma, "cut_latent_frame": 1,
                              "perturbation_seed": NOISE_SEED + 2,
                              "future_input_change": difference(strict_state[:, 1:], strict_poison[:, 1:])})
                checks["future_perturbation_strict_first1"] = value
                before = time.perf_counter()
                result = self_forcing_rollout(
                    source, sigma, 1,
                    lambda prefix, **kw: runtime.forward(prefix, sample, causal=True, **kw),
                    noise=noise, grad_blocks="none")
                if not torch.isfinite(result.latent).all():
                    raise FloatingPointError("Nonfinite strict-frame rollout")
                torch.npu.synchronize()
                label = "strict_frame_rollout_sigma03"
                record = {"role": role, "sigma": sigma, "block_frames": 1,
                          "seconds": time.perf_counter() - before, "audit": result.audit,
                          "latent_path": f"{label}.pt"}
                save(label, result.latent, record)
                rollouts[label] = record
                value = difference(result.latent[:, :1], strict_full[:, :1])
                value["passed"] = value["max_abs"] <= args.tolerance
                checks["strict_first_frame_rollout_replay"] = value
                runtime.block_frames = BLOCK_FRAMES
                if rank == 0:
                    print(json.dumps({"event": "rollout_completed", **record}), flush=True)

            # Layout values, rather than packed row offsets, must agree across prefixes.
            layout_checks = {}
            full_key = (sample["metadata"]["id"], FRAMES, 24, 42, 72)
            full_layout = runtime.layouts[full_key]
            for frames in (5, 10):
                layout = runtime.layouts[(sample["metadata"]["id"], frames, 24, 42, 72)]
                for modality in ("text", "audio", "video"):
                    ids = getattr(layout, f"{modality}_indices")
                    full_ids = getattr(full_layout, f"{modality}_indices")[:ids.numel()]
                    value = difference(layout.position_ids[ids], full_layout.position_ids[full_ids])
                    value["passed"] = value["bit_exact"]
                    layout_checks[f"prefix{frames}_{modality}"] = value

            anchors = {}
            if role == "real":
                for name, value in (("dense_one_untrained", dense_full),
                                    ("causal_one_untrained", outputs["causal_rollout_sigma0909375"])):
                    path = root / "outputs" / f"{name}.pt"
                    if path.exists():
                        expected = torch.load(path, map_location="cpu", weights_only=False)["latent"]
                        anchors[name] = difference(value, expected)
                        anchors[name]["passed"] = anchors[name]["max_abs"] <= args.tolerance
            save("upscaled", source, {"role": "fixed_input"})
            save("reference", sample["reference"], {"role": "comparison_only_not_model_input"})

        passed = all(value["passed"] for group in (checks, layout_checks, anchors) for value in group.values())
        report = {"completed": True, "passed": passed, "sample_id": sample["metadata"]["id"],
                  "role": role, "generator_checkpoint": str(args.generator_checkpoint) if args.generator_checkpoint else None,
                  "world_size": world, "physical_devices": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
                  "noise_seed": NOISE_SEED, "tolerance": args.tolerance,
                  "block_frames": BLOCK_FRAMES, "latent_frames": FRAMES,
                  "strict_frame_requested": args.strict_frame,
                  "fixed_audio_shape": list(sample["audio_latents"].shape),
                  "future_perturbation": {"start_latent_frame": BLOCK_FRAMES,
                      "replacement": "3 * independently seeded standard normal; only frames 5: change",
                      "seed": NOISE_SEED + 1, "input_change": future_input_change},
                  "checks": checks, "position_prefix_checks": layout_checks,
                  "existing_output_anchors": anchors, "forward_records": records, "rollouts": rollouts,
                  "seconds": time.perf_counter() - started,
                  "peak_allocated_gib": torch.npu.max_memory_allocated() / 2**30,
                  "scope": "Conditional block causality of the refiner relative to fixed upscaled input and fixed full text/audio",
                  "limitations": [
                      "The perturbation test checks the first block boundary on one clip and one seed, not every input/boundary.",
                      "Video attention is bidirectional within each five-frame latent block; this is not framewise causality.",
                      "Optional strict-frame mode means one latent frame per block, not pixel-frame or end-to-end causality.",
                      "Full 43-frame encoded-silence audio and text stay fixed; future-audio leakage is not tested.",
                      "The upstream VAE and latent upscaler remain noncausal; no end-to-end streaming claim.",
                      "Rollouts recompute prefixes and detach generated history; no KV cache or optimization is used.",
                      "Lower sigma changes input fidelity, noise amplitude, model time and Euler step simultaneously.",
                      "Latent invariance is a causality diagnostic; decoded full-clip review is still required for quality.",
                  ]}
        if rank == 0:
            (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"event": "completed", "passed": passed, "summary": str(output / "summary.json")}), flush=True)
        dist.barrier()
        if not passed:
            raise AssertionError("At least one causality, layout, replay or anchor check failed; inspect summary.json")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
