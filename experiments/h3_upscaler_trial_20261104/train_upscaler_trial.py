"""Small supervised fine-tune of the released LBH/Open-H3 latent upscaler.

This is our experimental MSE + temporal-difference objective, not the unpublished
author training recipe. It preserves Open-H3's D(U(N(z))) convention and full-clip
noncausal architecture. No VAE, diffusion refiner, text, or DMD is trained here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from data_utils import empty_cache, select_device, synchronize, tensor_load, video_tensor
from prepare_data import author_upscale, load_author_upscaler


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    return value


def read_pairs(manifest, split):
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty {split} manifest")
    pairs = []
    for row in rows:
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str) or Path(sample_id).name != sample_id or sample_id in (".", ".."):
            raise ValueError("Sample IDs must be safe directory names")
        path = Path(row["path"])
        if not path.is_absolute():
            path = manifest.parent / path
        sample = tensor_load(path)
        draft = video_tensor(sample["draft"], name="draft").float()
        reference = video_tensor(sample["reference"], name="reference").float()
        if reference.shape[:2] != draft.shape[:2] or reference.shape[-2:] != tuple(2 * n for n in draft.shape[-2:]):
            raise ValueError(f"{sample_id}: expected same-time, exact 2x spatial pair")
        if draft.shape[1] < 2 or any(n % 2 for n in reference.shape[-2:]):
            raise ValueError("Need at least two latent frames and 32-pixel-aligned targets")
        pairs.append({"sample_id": sample_id, "split": split, "path": str(path.resolve()),
                      "sha256": sha256(path), "draft": draft, "reference": reference,
                      "metadata": sample.get("metadata", {})})
    return pairs


def wrapper_forward(node, model, draft, target_shape):
    """Differentiable equivalent of the released node, including both N and D.

    `draft` is already a native normalized H3 latent; Open-H3 intentionally feeds
    it unchanged into the author node, whose own N/D are retained here.
    """
    if draft.ndim != 5 or len(target_shape) != 3:
        raise ValueError("Expected batched 5D latent and target (T,H,W)")
    mean, std = node._make_norm_tensors(draft.device, draft.dtype)
    scale = (target_shape[-1] / draft.shape[-1] + target_shape[-2] / draft.shape[-2]) / 2.0
    return model((draft - mean) / std, scale=scale, target_size=tuple(target_shape),
                 enable_chunking=False) * std + mean


def enable_activation_checkpointing(model):
    """Checkpoint blocks without altering modules, parameters, or state-dict keys."""
    for block in [*model.in_blocks, *model.out_blocks]:
        original = block.forward

        def forward(*values, _original=original):
            if torch.is_grad_enabled():
                return checkpoint(_original, *values, use_reentrant=False, preserve_rng_state=False)
            return _original(*values)

        block.forward = forward


def losses(prediction, reference, temporal_weight):
    mse = (prediction - reference).square().mean()
    temporal_mse = prediction[:, :, 1:] - prediction[:, :, :-1]
    temporal_mse = (temporal_mse - (reference[:, :, 1:] - reference[:, :, :-1])).square().mean()
    return mse + temporal_weight * temporal_mse, mse, temporal_mse


def metrics(prediction, reference, temporal_weight):
    prediction, reference = prediction.detach().float().cpu(), reference.detach().float().cpu()
    if prediction.shape != reference.shape or not torch.isfinite(prediction).all():
        raise ValueError("Invalid output shape or nonfinite prediction")
    objective, mse, temporal_mse = losses(prediction[None], reference[None], temporal_weight)
    return {"objective": float(objective), "latent_mse": float(mse),
            "latent_rmse": math.sqrt(float(mse)), "latent_mae": float((prediction - reference).abs().mean()),
            "temporal_difference_mse": float(temporal_mse),
            "signed_error_mean": float((prediction - reference).mean()),
            "prediction_std": float(prediction.std()), "reference_std": float(reference.std())}


def parameter_sample(model):
    # Inspect a deterministic spread in every parameter, including all biases.
    # This avoids a second 1.38 GB copy just to establish a nonzero update.
    values = []
    for parameter in model.parameters():
        flat = parameter.detach().flatten()
        indices = torch.linspace(0, flat.numel() - 1, min(64, flat.numel()), device=flat.device).long()
        values.append(flat[indices].float().cpu())
    return torch.cat(values)


def save_latent(output, pair, label, latent):
    path = output / "latents" / pair["sample_id"] / f"{label}.pt"
    atomic_torch_save(path, {"latent": latent.detach().float().cpu().contiguous(),
                            "metadata": {**pair["metadata"], "sample_id": pair["sample_id"],
                                         "split": pair["split"], "upscaler_trial_variant": label,
                                         "pair_sha256": pair["sha256"]}})
    return str(path.relative_to(output))


def run_evaluation(node, model, pairs, output, label, temporal_weight, device):
    results = {}
    with torch.no_grad():
        for pair in pairs:
            started = time.perf_counter()
            draft = pair["draft"][None].to(device)
            prediction = wrapper_forward(node, model, draft, pair["reference"].shape[-3:])[0].cpu()
            synchronize(device)
            value = metrics(prediction, pair["reference"], temporal_weight)
            value["seconds"] = time.perf_counter() - started
            value["latent_path"] = save_latent(output, pair, label, prediction)
            value["split"] = pair["split"]
            results[pair["sample_id"]] = value
            print(json.dumps({"evaluation": label, "sample_id": pair["sample_id"], **value}), flush=True)
            del prediction, draft
    return results


def save_checkpoint(output, model, optimizer, step, signature, config):
    directory = output / "checkpoint"
    atomic_torch_save(directory / "model.pt", {"model_state_dict": to_cpu(model.state_dict()), "step": step})
    atomic_torch_save(directory / "trainer.pt", {"optimizer_state_dict": to_cpu(optimizer.state_dict()),
                                                "step": step, "data_signature": signature, "config": config,
                                                "torch_rng_state": torch.get_rng_state()})
    atomic_json(directory / "checkpoint.json", {"step": step, "model": "model.pt", "trainer": "trainer.pt",
                                               "data_signature": signature})
    return directory


def memory_stats(device):
    backend = getattr(torch, device.type, None)
    if backend is None or not hasattr(backend, "max_memory_allocated"):
        return None
    return {"peak_allocated_gib": backend.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": backend.max_memory_reserved(device) / 2**30}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(os.environ.get("EXP_ROOT", Path(__file__).resolve().parents[1]))
    parser.add_argument("--train", type=Path, default=root / "upscaler_trial/pairs/train.jsonl")
    parser.add_argument("--holdout", type=Path, default=root / "upscaler_trial/pairs/holdout.jsonl")
    parser.add_argument("--upstream", type=Path, default=Path(os.environ.get(
        "H3_UPSTREAM_DIR", root / "upstream-sana/models/Open-H3-Regenerate-2K")))
    parser.add_argument("--upscaler", type=Path, default=root / "assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--steps", type=int, default=20, help="Total updates, including any resumed updates")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--temporal-weight", type=float, default=.1)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Optional periodic checkpoint; final checkpoint always saved")
    parser.add_argument("--resume", type=Path, help="Checkpoint directory; use a new --output directory")
    parser.add_argument("--parity-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if (args.steps < 1 or args.lr <= 0 or args.temporal_weight < 0 or args.weight_decay < 0
            or args.cpu_threads < 1 or args.checkpoint_every < 0 or args.parity_tolerance < 0):
        parser.error("Invalid training configuration")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new output directory; existing trial files will not be overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    train = read_pairs(args.train, "train")
    holdout = read_pairs(args.holdout, "holdout")
    pairs = train + holdout
    ids = [pair["sample_id"] for pair in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("Sample IDs must be unique and train/holdout disjoint")
    signature = {"train": [{"sample_id": p["sample_id"], "sha256": p["sha256"]} for p in train],
                 "holdout": [{"sample_id": p["sample_id"], "sha256": p["sha256"]} for p in holdout],
                 "upscaler_sha256": sha256(args.upscaler)}
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    summary = {"status": "initializing", "config": config, "data_signature": signature,
               "compute_dtype": "float32", "checkpoint_storage_dtype": "float32",
               "released_input_weight_dtype": "bfloat16 cast to float32 by the author loader",
               "parity_scope": "Same-device FP32 author node; not a comparison with BF16 computation",
               "normalization": "Open-H3 D(U(N(z))); z is native normalized H3 latent",
               "objective": "Our supervised normalized-latent MSE + 0.1 temporal-difference MSE (weight configurable); not author's unpublished recipe",
               "training_scope": "All released upscaler parameters; eval mode keeps dropout off; no VAE or refiner",
               "causal": False, "causal_note": "Released symmetric temporal convolutions and clip-wide GroupNorm are preserved",
               "holdout_policy": "Never optimized or used for checkpoint selection; final fixed-step checkpoint only",
               "limitations": "Tiny paired-data feasibility experiment, not evidence of generalization or perceptual superiority",
               "variants": {}}
    summary_path = args.output / "summary.json"
    atomic_json(summary_path, summary)
    device = select_device(args.device)
    backend = getattr(torch, device.type, None)
    if backend is not None and hasattr(backend, "reset_peak_memory_stats"):
        backend.reset_peak_memory_stats(device)
    started = time.perf_counter()
    node = load_author_upscaler(args.upstream, args.upscaler, device)
    model = node.load_model("upscaler", device, "fp32")
    model.eval()
    summary["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())

    pair = train[0]
    native = author_upscale(node, pair["draft"], height=pair["reference"].shape[-2],
                            width=pair["reference"].shape[-1], precision="fp32")
    with torch.no_grad():
        differentiable = wrapper_forward(node, model, pair["draft"][None].to(device),
                                         pair["reference"].shape[-3:])[0].cpu()
    synchronize(device)
    difference = (native - differentiable).abs()
    summary["author_node_parity"] = {"sample_id": pair["sample_id"], "max_abs": float(difference.max()),
                                    "mean_abs": float(difference.mean()), "tolerance": args.parity_tolerance}
    print(json.dumps({"author_node_parity": summary["author_node_parity"]}), flush=True)
    if not torch.isfinite(difference).all() or float(difference.max()) > args.parity_tolerance:
        raise RuntimeError("Differentiable wrapper failed released-node forward parity")
    del native, differentiable, difference

    summary["variants"]["pretrained"] = run_evaluation(node, model, pairs, args.output, "pretrained", args.temporal_weight, device)
    summary["variants"]["trilinear"] = {}
    for pair in pairs:
        prediction = F.interpolate(pair["draft"][None], size=pair["reference"].shape[-3:],
                                   mode="trilinear", align_corners=False)[0]
        value = metrics(prediction, pair["reference"], args.temporal_weight)
        value["latent_path"] = save_latent(args.output, pair, "trilinear", prediction)
        value["split"] = pair["split"]
        summary["variants"]["trilinear"][pair["sample_id"]] = value
        save_latent(args.output, pair, "reference", pair["reference"])
        save_latent(args.output, pair, "draft", pair["draft"])
    del prediction
    if args.activation_checkpointing:
        enable_activation_checkpointing(model)
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, foreach=False)
    first_step = 0
    if args.resume:
        marker = json.loads((args.resume / "checkpoint.json").read_text())
        restored_model = tensor_load(args.resume / "model.pt")
        restored_trainer = tensor_load(args.resume / "trainer.pt")
        if not (marker["step"] == restored_model["step"] == restored_trainer["step"]):
            raise ValueError("Incomplete checkpoint: model/optimizer/marker step mismatch")
        if restored_trainer["data_signature"] != signature or marker["data_signature"] != signature:
            raise ValueError("Resume requires identical paired data and initial released weights")
        for key in ("lr", "temporal_weight", "weight_decay", "seed", "activation_checkpointing"):
            if restored_trainer["config"][key] != config[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        model.load_state_dict(restored_model["model_state_dict"], strict=True)
        optimizer.load_state_dict(restored_trainer["optimizer_state_dict"])
        torch.set_rng_state(restored_trainer["torch_rng_state"])
        first_step = restored_trainer["step"]
        del restored_model, restored_trainer
        if first_step >= args.steps:
            raise ValueError("--steps must exceed completed checkpoint updates")
    initial_parameters = parameter_sample(model)
    summary["status"] = "training"
    summary["resumed_updates"] = first_step
    atomic_json(summary_path, summary)
    training_started = time.perf_counter()
    with (args.output / "steps.jsonl").open("w") as log:
        for step in range(first_step, args.steps):
            step_started = time.perf_counter()
            pair = train[step % len(train)]
            draft, reference = pair["draft"][None].to(device), pair["reference"][None].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = wrapper_forward(node, model, draft, reference.shape[-3:])
            loss, mse, temporal_mse = losses(prediction, reference, args.temporal_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step + 1}")
            loss.backward()
            # Infinite max_norm reports the norm without clipping. error_if_nonfinite
            # rejects NaN/Inf gradients before optimizer mutation.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"),
                                                       error_if_nonfinite=True, foreach=False)
            if not float(grad_norm) > 0:
                raise RuntimeError("No nonzero parameter gradients")
            with_grad = sum(parameter.grad is not None for parameter in model.parameters())
            if with_grad != sum(1 for _ in model.parameters()):
                raise RuntimeError("Some upscaler parameters are disconnected from the training objective")
            optimizer.step()
            synchronize(device)
            record = {"step": step + 1, "sample_id": pair["sample_id"], "loss": float(loss.detach()),
                      "latent_mse": float(mse.detach()), "temporal_difference_mse": float(temporal_mse.detach()),
                      "grad_norm": float(grad_norm), "parameters_with_grad": with_grad,
                      "seconds": time.perf_counter() - step_started, "lr": optimizer.param_groups[0]["lr"]}
            if step == first_step:
                first_change = (parameter_sample(model) - initial_parameters).abs()
                record["first_update_sampled_max_abs"] = float(first_change.max())
                record["first_update_sampled_changed_values"] = int((first_change > 0).sum())
                if not record["first_update_sampled_changed_values"]:
                    raise RuntimeError("First update did not alter sampled parameter values")
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            del draft, reference, prediction, loss, mse, temporal_mse, grad_norm
            if args.checkpoint_every and (step + 1) % args.checkpoint_every == 0 and step + 1 < args.steps:
                save_checkpoint(args.output, model, optimizer, step + 1, signature, config)
    summary["training_seconds"] = time.perf_counter() - training_started
    optimizer.zero_grad(set_to_none=True)
    change = (parameter_sample(model) - initial_parameters).abs()
    summary["parameter_update"] = {"sampling": "Up to 64 evenly spaced values from every parameter",
                                   "values": len(change), "changed_values": int((change > 0).sum()),
                                   "max_abs": float(change.max()), "mean_abs": float(change.mean())}
    summary["variants"]["finetuned"] = run_evaluation(node, model, pairs, args.output, "finetuned", args.temporal_weight, device)
    checkpoint_dir = save_checkpoint(args.output, model, optimizer, args.steps, signature, config)
    summary["checkpoint"] = {"directory": str(checkpoint_dir), "completed_updates": args.steps,
                             "model_sha256": sha256(checkpoint_dir / "model.pt"),
                             "optimizer_saved": True}
    # Reload into a newly constructed author model, after releasing the trained
    # model and optimizer from the accelerator. This verifies the saved artifact.
    model.to("cpu")
    del optimizer, model
    node.MODEL_CACHE.clear()
    empty_cache(device)
    reloaded_node = load_author_upscaler(args.upstream, args.upscaler, device)
    reloaded_model = reloaded_node.load_model("upscaler", device, "fp32")
    saved = tensor_load(checkpoint_dir / "model.pt")
    reloaded_model.load_state_dict(saved["model_state_dict"], strict=True)
    del saved
    pair = holdout[0]
    with torch.no_grad():
        actual = wrapper_forward(reloaded_node, reloaded_model, pair["draft"][None].to(device),
                                 pair["reference"].shape[-3:])[0].cpu()
    synchronize(device)
    expected = tensor_load(args.output / summary["variants"]["finetuned"][pair["sample_id"]]["latent_path"])["latent"]
    reload_error = (actual - expected).abs()
    summary["checkpoint_reload"] = {"sample_id": pair["sample_id"], "max_abs": float(reload_error.max()),
                                    "mean_abs": float(reload_error.mean()), "tolerance": args.parity_tolerance}
    if not torch.isfinite(reload_error).all() or float(reload_error.max()) > args.parity_tolerance:
        raise RuntimeError("Checkpoint reload output mismatch")
    summary["memory"] = memory_stats(device)
    summary["completed_updates"] = args.steps
    summary["total_seconds"] = time.perf_counter() - started
    summary["status"] = "completed"
    atomic_json(summary_path, summary)
    print(json.dumps({"status": "completed", "summary": str(summary_path),
                      "checkpoint_reload": summary["checkpoint_reload"], "memory": summary["memory"]}), flush=True)


if __name__ == "__main__":
    main()
