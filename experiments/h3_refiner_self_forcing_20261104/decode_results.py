"""Decode normalized H3 latent results with the released FP32 tiled VAE.

Example: --sample data/samples/ID.pt --latent student=results.pt::student
Each input produces an MP4, plus a labelled first/middle/last contact sheet and
a side-by-side MP4. Draft pixels are enlarged only for comparison displays.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from data_utils import (decode_latent, empty_cache, load_video_vae, released_decoded_frames,
                        select_device, synchronize, tensor_load, video_tensor, write_mp4)


def load_input(spec):
    if "=" not in spec:
        raise ValueError("--latent expects label=path.pt or label=path.pt::key")
    label, location = spec.split("=", 1)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError("Labels may contain letters, digits, dot, underscore and hyphen")
    path, sep, key = location.partition("::")
    value = tensor_load(path)
    if sep:
        for component in key.split("."):
            value = value[component]
    elif isinstance(value, dict):
        candidates = [k for k in ("latent", "refined", "student", "latents", "video_latents", "prediction") if k in value]
        if len(candidates) != 1:
            raise ValueError(f"{path}: specify ::key; candidates={list(value)}")
        value = value[candidates[0]]
    return label, video_tensor(value, name=label), {"path": path, "key": key or None}


def label_image(frame, label, size):
    image = Image.fromarray(frame.numpy()).resize(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size[0], size[1] + 28), "#111111")
    canvas.paste(image, (0, 28))
    ImageDraw.Draw(canvas).text((8, 7), label, fill="white")
    return canvas


def comparisons(outputs, target_dir, *, fps, crf, comparison_video):
    lengths = {len(x) for x in outputs.values()}
    if len(lengths) != 1:
        raise ValueError(f"Comparison videos must have equal length, got {lengths}")
    count = next(iter(lengths))
    width = max(x.shape[2] for x in outputs.values())
    height = max(x.shape[1] for x in outputs.values())
    columns = len(outputs)
    rows = [0, count // 2, count - 1]
    sheet = Image.new("RGB", (width * columns, (height + 28) * len(rows)))
    for row, index in enumerate(rows):
        for col, (label, frames) in enumerate(outputs.items()):
            cell = label_image(frames[index], f"{label} | frame {index}", (width, height))
            sheet.paste(cell, (col * width, row * (height + 28)))
    sheet.save(target_dir / "comparison.png")
    if comparison_video:
        frames = []
        for index in range(count):
            canvas = Image.new("RGB", (width * columns, height + 28))
            for col, (label, value) in enumerate(outputs.items()):
                canvas.paste(label_image(value[index], label, (width, height)), (col * width, 0))
            frames.append(torch.from_numpy(np.asarray(canvas).copy()))
        write_mp4(torch.stack(frames), target_dir / "comparison.mp4", fps=fps, crf=crf)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default=os.environ.get("H3_BASE_MODEL", "models/h3-base"))
    p.add_argument("--sample", type=Path)
    p.add_argument("--fields", default="draft,upscaled,reference")
    p.add_argument("--latent", action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--cpu-threads", type=int, default=24)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--crf", type=int, default=16)
    p.add_argument("--no-comparison-video", action="store_true")
    args = p.parse_args()
    if args.cpu_threads < 1:
        p.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    inputs, provenance, sample_meta = {}, {}, {}
    if args.sample:
        sample = tensor_load(args.sample)
        sample_meta = sample.get("metadata", {})
        for field in filter(None, args.fields.split(",")):
            inputs[field] = video_tensor(sample[field], name=field)
            provenance[field] = {"path": str(args.sample), "key": field}
    for spec in args.latent:
        label, value, origin = load_input(spec)
        if label in inputs:
            p.error(f"Duplicate label {label}")
        inputs[label], provenance[label] = value, origin
    if not inputs:
        p.error("Supply --sample and/or --latent")
    fps = args.fps or sample_meta.get("fps", 24.)
    counts = {released_decoded_frames(x.shape[1]) for x in inputs.values()}
    if len(counts) != 1 and args.frames is None:
        p.error(f"Latent inputs decode to different lengths {counts}; pass --frames intentionally")
    keep = args.frames if args.frames is not None else sample_meta.get("decoded_frames")
    args.output.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    vae = load_video_vae(args.base_model, device)
    outputs, records = {}, {}
    for label, value in inputs.items():
        started = time.perf_counter()
        frames = decode_latent(vae, value, device, keep_frames=keep)
        synchronize(device)
        elapsed = time.perf_counter() - started
        outputs[label] = frames
        codec = write_mp4(frames, args.output / f"{label}.mp4", fps=fps, crf=args.crf)
        records[label] = {**provenance[label], "latent_shape": list(value.shape),
                          "pixel_shape": list(frames.shape), "decode_seconds": elapsed, "video_codec": codec}
        print(json.dumps({"decoded": label, **records[label]}), flush=True)
    del vae
    empty_cache(device)
    comparisons(outputs, args.output, fps=fps, crf=args.crf, comparison_video=not args.no_comparison_video)
    (args.output / "decode.json").write_text(json.dumps({"base_model": args.base_model, "dtype": "float32",
        "tiling": True, "fps": fps, "latent_denormalization": "normalized*std+mean",
        "pixel_denormalization": "ImageNet std/mean; clamp[0,1]", "inputs": records,
        "source_metadata": sample_meta}, indent=2))


if __name__ == "__main__":
    main()
