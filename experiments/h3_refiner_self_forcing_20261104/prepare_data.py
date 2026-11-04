"""Prepare explicit 2x source-video drafts for the H3 Self Forcing/DMD smoke run.

Default: source video -> 336x192 RGB -> released FP32 VAE -> official learned
upscaler -> 672x384 normalized latent. A synthetic cached-latent draft is only
used when explicitly requested or enabled with --allow-synthetic-fallback.
All stored tensors are CPU tensors. No transformer is loaded here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data_utils import (empty_cache, encode_pixels, encode_silence, load_video_vae,
                        read_video, released_decoded_frames, select_device,
                        synchronize, tensor_load, video_tensor, decode_latent, write_mp4)


DEFAULT_MANIFEST = os.environ.get("H3_SOURCE_MANIFEST", "data/source_manifest.jsonl")
DEFAULT_BASE = os.environ.get("H3_BASE_MODEL", "models/h3-base")


def resolve_path(row, key, root):
    p = Path(row[key])
    return p if p.is_absolute() else root / p


def indices(value):
    result = [int(x) for x in value.split(",") if x.strip()]
    if not result or len(result) != len(set(result)) or min(result) < 0:
        raise argparse.ArgumentTypeError("Expected unique nonnegative comma-separated indices")
    return result


def load_author_upscaler(source: Path, checkpoint: Path, device: torch.device):
    spec = importlib.util.spec_from_file_location("h3_author_upscaler_loader", source / "open_h3/upscaler.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    node = module.load(checkpoint)
    # Only route devices; architecture, normalization, resize/rounding and weights
    # remain the unchanged author node. Do not use its CUDA-only wrapper.
    node._resolve_device = lambda backend: device
    original_label = node._backend_label
    node._backend_label = lambda dev: "Ascend NPU" if dev.type == "npu" else original_label(dev)
    return node


def author_upscale(node, draft, *, height, width, precision):
    mode = {"mode": node.UpscaleMode.TARGET_DIMENSIONS,
            "width": width * 16, "height": height * 16}
    with torch.inference_mode():
        result, = node.MinimaxH3LatentUpscaler3D.execute(
            {"samples": draft[None].float()}, "upscaler", mode, align=32,
            enable_temporal_chunking=False, force_unload=False, device="cpu", precision=precision)
    # Author node copies output to CPU asynchronously; complete that transfer
    # before validating/saving the host tensor.
    synchronize(node._resolve_device("cpu"))
    out = video_tensor(result["samples"][0].float(), name="upscaled")
    if tuple(out.shape) != (24, draft.shape[1], height, width):
        raise ValueError(f"Author upscaler returned unexpected geometry {out.shape}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path(DEFAULT_MANIFEST))
    p.add_argument("--base-model", type=Path, default=Path(DEFAULT_BASE))
    p.add_argument("--upstream", type=Path, default=Path(os.environ.get("H3_UPSTREAM_DIR", "upstream")))
    p.add_argument("--upscaler", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-indices", type=indices, default=[0, 3])
    p.add_argument("--holdout-indices", type=indices, default=[2])
    p.add_argument("--latent-frames", type=int, default=13)
    p.add_argument("--source-frames", type=int, default=64)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=672)
    p.add_argument("--fps", type=float, default=24.)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--cpu-threads", type=int, default=24)
    p.add_argument("--upscaler-device", default=None)
    p.add_argument("--upscaler-precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--draft-mode", choices=["source", "synthetic"], default="source")
    p.add_argument("--audio-mode", choices=["silence", "cached"], default="silence")
    p.add_argument("--allow-synthetic-fallback", action="store_true")
    p.add_argument("--draft-cache", type=Path, help="Persist/reuse source VAE encodes and official encoded silence")
    p.add_argument("--drafts-only", action="store_true", help="Prepare draft cache without requiring upscaler weights")
    p.add_argument("--validate-drafts", action="store_true", help="Decode source drafts to MP4 and check temporal geometry")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--inspect-only", action="store_true", help="Check selected CPU caches; no NPU/model initialization")
    args = p.parse_args()
    if args.cpu_threads < 1:
        p.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    if (args.drafts_only or args.validate_drafts) and args.draft_cache is None:
        p.error("--drafts-only/--validate-drafts requires --draft-cache")
    if not args.drafts_only and not args.inspect_only and args.upscaler is None:
        p.error("Full preparation requires --upscaler")
    if args.height % 32 or args.width % 32:
        p.error("Target dimensions must be divisible by 32 (draft VAE grid and target DiT patches)")
    if args.fps <= 0 or args.source_frames < 1:
        p.error("FPS/source-frame count must be positive")
    if set(args.train_indices) & set(args.holdout_indices):
        p.error("Train and holdout indices must be disjoint")
    decoded_frames = released_decoded_frames(args.latent_frames)
    if 5 * math.ceil(args.source_frames / 17) - 3 < args.latent_frames:
        p.error("--source-frames encodes to fewer latents than requested")
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    selected = [(i, split) for split, group in (("train", args.train_indices), ("holdout", args.holdout_indices)) for i in group]
    ids = []
    items = []
    for idx, split in selected:
        row = rows[idx]
        ids.append(row["id"])
        reference = video_tensor(tensor_load(resolve_path(row, "latent_path", args.manifest.parent)), name="reference")
        expected_hw = (args.height // 16, args.width // 16)
        if reference.shape[2:] != expected_hw or reference.shape[1] < args.latent_frames:
            raise ValueError(f"{row['id']}: cached reference geometry {reference.shape} does not match request")
        text = tensor_load(resolve_path(row, "text_path", args.manifest.parent))
        audio = tensor_load(resolve_path(row, "audio_path", args.manifest.parent))
        if text["prompt_embeds"].ndim != 2 or text["prompt_embeds"].shape[1] != 5120:
            raise ValueError("Expected prompt_embeds [N,5120]")
        if text["text_token_tags"].shape != text["prompt_embeds"].shape[:1] or not (text["text_token_tags"] == 1).all():
            raise ValueError("Text tags must be N copies of released modality tag 1")
        if audio.ndim != 3 or audio.shape[:2] != (2, 32):
            raise ValueError("Expected cached audio [2,32,A]")
        source = resolve_path(row, "source_video_path", args.manifest.parent)
        summary = {"index": idx, "id": row["id"], "split": split,
                   "source_exists": source.is_file(), "reference": list(reference.shape),
                   "text": list(text["prompt_embeds"].shape), "source_audio": list(audio.shape),
                   "target_latent_frames": args.latent_frames, "decoded_frames": decoded_frames}
        print(json.dumps(summary), flush=True)
        items.append((idx, split, row, source, reference, text, audio))
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated sample ID across splits")
    if args.inspect_only:
        return
    if not args.drafts_only and not args.upscaler.is_file():
        raise FileNotFoundError(args.upscaler)
    args.output.mkdir(parents=True, exist_ok=True)
    sample_dir = args.output / "samples"
    sample_dir.mkdir(exist_ok=True)
    if args.draft_cache:
        args.draft_cache.mkdir(parents=True, exist_ok=True)
    for sample_id in ids:
        if not args.drafts_only and (sample_dir / f"{sample_id}.pt").exists() and not args.overwrite:
            raise FileExistsError(f"{sample_id}.pt already exists; pass --overwrite intentionally")
    device = select_device(args.device)
    up_device = select_device(args.upscaler_device or args.device)
    silence, silence_meta = None, None
    if args.audio_mode == "silence":
        silence_key = {"base_model": str(args.base_model.resolve()), "frames": decoded_frames, "fps": args.fps}
        silence_path = args.draft_cache / "silence.pt" if args.draft_cache else None
        if silence_path and silence_path.exists():
            cached = tensor_load(silence_path)
            if cached["cache_key"] != silence_key:
                raise ValueError("Silence cache parameters differ; use a separate --draft-cache directory")
            silence, silence_meta = cached["audio"], cached["metadata"]
            print(json.dumps({"reused_silence": str(silence_path), "shape": list(silence.shape)}), flush=True)
        else:
            silence, silence_meta = encode_silence(args.base_model, frames=decoded_frames, fps=args.fps, device=device)
            if silence_path:
                torch.save({"audio": silence, "metadata": silence_meta, "cache_key": silence_key}, silence_path)
            print(json.dumps({"encoded_silence": silence_meta, "shape": list(silence.shape)}), flush=True)
    vae = None
    prepared = []
    for idx, split, row, source, full_reference, text, source_audio in items:
        start = time.perf_counter()
        reference = full_reference[:, :args.latent_frames].contiguous()
        mode, fallback_reason, encoded_shape = args.draft_mode, None, None
        draft_path = args.draft_cache / f"{row['id']}.draft.pt" if args.draft_cache else None
        draft_key = {"source_video": str(source), "source_video_sha256": row.get("source_video_sha256"),
                     "base_model": str(args.base_model.resolve()), "source_frames": args.source_frames,
                     "latent_frames": args.latent_frames, "width": args.width // 2, "height": args.height // 2,
                     "fps": args.fps, "seed": args.seed, "resize": "area", "mode": mode}
        if mode == "source":
            try:
                if draft_path and draft_path.exists():
                    cached = tensor_load(draft_path)
                    if cached["cache_key"] != draft_key:
                        raise ValueError("Draft cache parameters differ; use a separate --draft-cache directory")
                    draft = video_tensor(cached["draft"])
                    encoded_shape = cached["encoded_shape"]
                    print(json.dumps({"reused_draft": str(draft_path)}), flush=True)
                else:
                    if vae is None:
                        vae = load_video_vae(args.base_model, device)
                    pixels = read_video(source, width=args.width // 2, height=args.height // 2,
                                        frames=args.source_frames, fps=args.fps)
                    encoded = encode_pixels(vae, pixels, device, args.seed)
                    encoded_shape = list(encoded.shape)
                    if encoded.shape[1] < args.latent_frames:
                        raise ValueError(f"Only {encoded.shape[1]} encoded latent frames")
                    draft = encoded[:, :args.latent_frames].contiguous()
                    del pixels, encoded
                    if draft_path:
                        temp_path = draft_path.with_suffix(".pt.tmp")
                        torch.save({"draft": draft, "encoded_shape": encoded_shape, "cache_key": draft_key}, temp_path)
                        temp_path.replace(draft_path)
            except Exception as exc:
                if not args.allow_synthetic_fallback:
                    raise
                fallback_reason = f"{type(exc).__name__}: {exc}"
                mode = "synthetic"
                print(json.dumps({"WARNING": "EXPLICIT SYNTHETIC FALLBACK", "id": row["id"],
                                  "reason": fallback_reason}), flush=True)
                del vae
                vae = None
                empty_cache(device)
        if mode == "synthetic":
            draft = F.interpolate(reference.float().permute(1, 0, 2, 3), size=(args.height // 32, args.width // 32),
                                  mode="area").permute(1, 0, 2, 3).contiguous()
        if tuple(draft.shape) != (24, args.latent_frames, args.height // 32, args.width // 32):
            raise ValueError(f"Unexpected draft shape {draft.shape}")
        # A preview encoder failure must never change the training draft mode.
        if args.validate_drafts:
            if vae is None:
                vae = load_video_vae(args.base_model, device)
            preview = decode_latent(vae, draft, device)
            write_mp4(preview, args.draft_cache / f"{row['id']}.draft.mp4", fps=args.fps)
            print(json.dumps({"validated_draft_decode": row["id"], "pixel_shape": list(preview.shape)}), flush=True)
            del preview
        if silence is not None:
            audio, audio_meta = silence, silence_meta
        else:
            n_audio = math.ceil(source_audio.shape[-1] * decoded_frames / row["frames"])
            audio = source_audio[..., :n_audio].contiguous()
            audio_meta = {"kind": "cached_source_audio_prefix", "latent_frames": n_audio,
                          "source_latent_frames": source_audio.shape[-1]}
        metadata = {"id": row["id"], "split": split, "source_manifest_index": idx,
                    "source_manifest": str(args.manifest), "source_video_path": str(source),
                    "source_latent_path": str(resolve_path(row, "latent_path", args.manifest.parent)),
                    "source_video_sha256": row.get("source_video_sha256"),
                    "source_latent_sha256": row.get("latent_sha256"), "source_split": row.get("split"),
                    "split_scope": "disjoint within this 3-sample smoke experiment; source manifest is old train data",
                    "prompt": text.get("prompt", row.get("prompt", "")), "fps": args.fps,
                    "decoded_frames": decoded_frames, "latent_frames": args.latent_frames,
                    "draft_mode": mode, "synthetic_fallback_reason": fallback_reason,
                    "source_frames_encoded": args.source_frames if mode == "source" else None,
                    "encoded_latent_shape_before_crop": encoded_shape,
                    "draft_pixels": [args.width // 2, args.height // 2], "target_pixels": [args.width, args.height],
                    "pixel_preprocessing": "fps resample; cover-area resize; center crop; final-frame hold; ImageNet normalization",
                    "latent_normalization": "(raw-vae.latents_mean)/vae.latents_std",
                    "reference_provenance": row.get("preprocessing"), "posterior_seed": args.seed,
                    "audio": audio_meta, "draft_encode_seconds": time.perf_counter() - start}
        prepared.append({"draft": draft.float().cpu(), "reference": reference.cpu(),
                         "prompt_embeds": text["prompt_embeds"].cpu().contiguous(),
                         "text_token_tags": text["text_token_tags"].cpu().long().contiguous(),
                         "audio_latents": audio.cpu().contiguous(),
                         "source_audio_latents": source_audio.cpu().contiguous(), "metadata": metadata})
        print(json.dumps({"prepared_draft": row["id"], "mode": mode, "shape": list(draft.shape)}), flush=True)
    del vae
    empty_cache(device)
    if args.drafts_only:
        (args.draft_cache / "drafts.json").write_text(json.dumps([x["metadata"] for x in prepared], indent=2))
        print(json.dumps({"drafts_only_complete": str(args.draft_cache), "samples": len(prepared)}), flush=True)
        return
    node = load_author_upscaler(args.upstream, args.upscaler, up_device)
    manifests = {"train": [], "holdout": []}
    for item in prepared:
        start = time.perf_counter()
        item["upscaled"] = author_upscale(node, item["draft"], height=args.height // 16,
                                            width=args.width // 16, precision=args.upscaler_precision)
        synchronize(up_device)
        meta = item["metadata"]
        meta["upscaler"] = {"checkpoint": str(args.upscaler), "source": str(args.upstream / "open_h3/upscaler_node.py"),
                            "precision": args.upscaler_precision, "device": str(up_device), "scale": 2.0,
                            "normalization": "author node D(U(N(normalized_native_latent))); preserved exactly",
                            "seconds": time.perf_counter() - start}
        for name, value in item.items():
            if torch.is_tensor(value) and (value.device.type != "cpu" or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid output tensor {name}")
        path = sample_dir / f"{meta['id']}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(item, tmp)
        tmp.replace(path)
        manifests[meta["split"]].append({"id": meta["id"], "path": str(path.resolve()), "split": meta["split"],
                                         "draft_mode": meta["draft_mode"]})
        print(json.dumps({"saved": str(path), "audio_shape": list(item["audio_latents"].shape),
                          "upscaled_shape": list(item["upscaled"].shape)}), flush=True)
    for split, lines in manifests.items():
        (args.output / f"{split}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    (args.output / "preparation.json").write_text(json.dumps({"arguments": {k: str(v) if isinstance(v, Path) else v
                                                       for k, v in vars(args).items()}, "samples": manifests}, indent=2))


if __name__ == "__main__":
    main()
