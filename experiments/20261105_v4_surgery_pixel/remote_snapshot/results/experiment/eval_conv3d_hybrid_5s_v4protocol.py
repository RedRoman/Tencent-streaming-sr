#!/usr/bin/env python
"""Compare conv3d-hybrid best.pt against v4 on the SAME v4 5s protocol (11 clips).

Loads the hybrid student via its own loader, runs native_forward (adapter target
2x spatial, same T), teacher, causality cuts on holdout, decode + pixel metric.
Reference latents are reused from runs/v4_eval_5s.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch

import sys as _sys
_sys.path.insert(0, "/home/ma-user/workspace/zzjrua/vdn-minimax-h3/diffusers/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import decode_latent, empty_cache, load_video_vae, select_device, synchronize, tensor_load, write_mp4
from evaluate_causal_upscaler_pixels import metric
from prepare_data import load_author_upscaler
from train_upscaler_trial import atomic_json, metrics, read_pairs, save_latent, wrapper_forward
from h3_conv3d_hybrid import load_student_checkpoint

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, default=Path("upscaler_5s_v4exp/pairs"))
    p.add_argument("--upstream", type=Path, default=Path("upstream"))
    p.add_argument("--upscaler", type=Path, default=Path(
        "assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--base-model", default="/temp/h3_training_assets_20260921/VDN-MiniMax-H3/h3-base")
    p.add_argument("--reuse-latents", type=Path, default=Path("runs/v4_eval_5s"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="npu:6")
    p.add_argument("--tolerance", type=float, default=1e-4)
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.output.exists():
        raise FileExistsError(args.output)
    pairs = read_pairs(args.pairs / "train.jsonl", "train") + read_pairs(args.pairs / "holdout.jsonl", "holdout")
    device = select_device(args.device)
    args.output.mkdir(parents=True)
    summary = {"status": "running", "checkpoint": str(args.checkpoint), "variants": {},
               "causality": {}, "pixel_metrics": {}}

    student = load_student_checkpoint(args.checkpoint, args.upstream, args.upscaler, device)
    student.eval().requires_grad_(False)

    def sfn(native):
        return student(native)

    node = load_author_upscaler(args.upstream, args.upscaler, device)
    teacher = node.load_model("upscaler", device, "fp32").eval().requires_grad_(False)
    started = time.perf_counter()
    for pair in pairs:
        ident = pair["sample_id"]
        for label, key in (("reference", "reference"), ("draft", "draft")):
            (args.output / "latents" / ident).mkdir(parents=True, exist_ok=True)
            torch.save({"latent": pair[key].clone()}, args.output / "latents" / ident / f"{label}.pt")
    with torch.no_grad():
        for pair in pairs:
            native = pair["draft"][None].to(device)
            target = tuple(pair["reference"].shape[-3:])
            then = time.perf_counter()
            tout = wrapper_forward(node, teacher, native, target).cpu()
            synchronize(device)
            rec = metrics(tout[0], pair["reference"], 0.1)
            rec.update(split=pair["split"], seconds=round(time.perf_counter() - then, 2))
            summary["variants"].setdefault("teacher", {})[pair["sample_id"]] = rec
            then = time.perf_counter()
            sfull = sfn(native).cpu()
            synchronize(device)
            rec = metrics(sfull[0], pair["reference"], 0.1)
            rec["vs_teacher"] = metrics(sfull[0], tout[0], 0.1)
            rec.update(split=pair["split"], seconds=round(time.perf_counter() - then, 2))
            summary["variants"].setdefault("conv3d_hybrid", {})[pair["sample_id"]] = rec
            save_latent(args.output, pair, "conv3d_hybrid", sfull[0])
            if pair["split"] == "holdout":
                checks = {}
                for cut in (1, 13, 25):
                    changed = native.clone()
                    changed[:, :, cut:] = 3 * torch.randn(changed[:, :, cut:].shape,
                        generator=torch.Generator().manual_seed(20261104 + cut)).to(device)
                    actual = sfn(changed).cpu()
                    checks[f"future_cut{cut}_max_abs"] = float((actual[:, :, :cut] - sfull[:, :, :cut]).abs().max())
                    prefix = sfn(native[:, :, :cut].contiguous()).cpu()
                    checks[f"prefix_cut{cut}_max_abs"] = float((prefix - sfull[:, :, :cut]).abs().max())
                student.reset_stream()
                seqs = [student.stream_step(native[:, :, i:i+1]).cpu() for i in range(native.shape[2])]
                checks["streaming_max_abs"] = float((torch.cat(seqs, 2) - sfull).abs().max())
                checks["passed"] = all(v < args.tolerance for k, v in checks.items() if isinstance(v, float))
                summary["causality"][pair["sample_id"]] = checks
                print(json.dumps({"causality": pair["sample_id"], **checks}), flush=True)
            print(json.dumps({"conv3d_hybrid": pair["sample_id"], "latent_mse": rec["latent_mse"]}), flush=True)
            del native, tout, sfull
    del node, teacher
    empty_cache(device)
    summary["status"] = "decoding"
    atomic_json(args.output / "summary.json", summary)
    vae = load_video_vae(args.base_model, device)
    for pair in pairs:
        ident = pair["sample_id"]
        directory = args.output / "decoded" / ident
        directory.mkdir(parents=True)
        frames = {}
        for label in ("conv3d_hybrid", "reference"):
            value = tensor_load(args.output / "latents" / ident / f"{label}.pt")["latent"]
            frames[label] = decode_latent(vae, value, device, keep_frames=120)
            if frames[label].shape != (120, 384, 672, 3):
                raise ValueError("bad decode geometry")
            write_mp4(frames[label], directory / f"{label}.mp4", fps=24.)
        row = {"conv3d_hybrid": {"vs_decoded_reference": metric(frames["conv3d_hybrid"], frames["reference"])}}
        summary["pixel_metrics"][ident] = row
        atomic_json(args.output / "summary.json", summary)
        print(json.dumps({"pixel_metrics": ident, **row}), flush=True)
    summary["seconds"] = round(time.perf_counter() - started, 1)
    summary["status"] = "completed"
    summary["causality_passed"] = all(v["passed"] for v in summary["causality"].values())
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps({"completed": True, **summary["causality"]}), flush=True)
    empty_cache(device)

if __name__ == "__main__":
    main()
