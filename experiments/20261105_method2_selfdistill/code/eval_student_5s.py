"""5s evaluation of the self-distilled causal student on all upscaler_5s clips.

Latent metrics vs teacher (bidirectional Open-H3 upscaler) output and vs the
encoded reference; prefix causality checks; streaming consistency; decode to
120-frame 24fps mp4; pixel PSNR vs decoded reference. Read-only for existing runs.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch

import sys as _sys
_sys.path.insert(0, "/home/ma-user/workspace/zzjrua/vdn-minimax-h3/diffusers/src")  # vendored diffusers with AutoencoderKLMiniMaxH3

from data_utils import decode_latent, empty_cache, load_video_vae, select_device, synchronize, tensor_load, write_mp4
from evaluate_causal_upscaler_pixels import metric
from prepare_data import load_author_upscaler
from train_upscaler_trial import atomic_json, metrics, read_pairs, save_latent, wrapper_forward


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, default=Path("upscaler_5s/pairs"))
    p.add_argument("--upstream", type=Path, default=Path("../upstream"))
    p.add_argument("--upscaler", type=Path, default=Path("../assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"))
    p.add_argument("--student", type=Path, required=True)
    p.add_argument("--base-model", default="/temp/h3_training_assets_20260921/VDN-MiniMax-H3/h3-base")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="npu:6")
    p.add_argument("--tolerance", type=float, default=1e-4)
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.output.exists(): raise FileExistsError(args.output)
    import sys; sys.path.insert(0, "..")
    from experiment.train_student_distill import CausalStudentUpscaler
    pairs = read_pairs(args.pairs / "train.jsonl", "train") + read_pairs(args.pairs / "holdout.jsonl", "holdout")
    device = select_device(args.device)
    args.output.mkdir(parents=True)
    summary = {"status": "running", "student": str(args.student), "variants": {}, "causality": {}, "pixel_metrics": {}}
    # load student
    ck = torch.load(args.student, map_location="cpu", weights_only=False)
    student = CausalStudentUpscaler(24, 256, 6)
    student.load_state_dict(ck["model"]); student.eval().requires_grad_(False)
    student = student.to(device)
    started = time.perf_counter()
    for pair in pairs:
        save_latent(args.output, pair, "reference", pair["reference"])
        save_latent(args.output, pair, "draft", pair["draft"])
    # teacher outputs
    node = load_author_upscaler(args.upstream, args.upscaler, device)
    teacher = node.load_model("upscaler", device, "fp32").eval().requires_grad_(False)
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
            save_latent(args.output, pair, "teacher", tout[0])
            # student full forward
            then = time.perf_counter()
            sfull = student(native).cpu()
            synchronize(device)
            rec = metrics(sfull[0], pair["reference"], 0.1)
            rec["vs_teacher"] = metrics(sfull[0], tout[0], 0.1)
            rec.update(split=pair["split"], seconds=round(time.perf_counter() - then, 2))
            summary["variants"].setdefault("student", {})[pair["sample_id"]] = rec
            save_latent(args.output, pair, "student", sfull[0])
            # causality checks on holdout clips
            if pair["split"] == "holdout":
                checks = {}
                for cut in (1, 13, 25):
                    changed = native.clone()
                    changed[:, :, cut:] = 3 * torch.randn(changed[:, :, cut:].shape,
                        generator=torch.Generator().manual_seed(20261104 + cut)).to(device)
                    actual = student(changed).cpu()
                    d = float((actual[:, :, :cut] - sfull[:, :, :cut]).abs().max())
                    checks[f"future_cut{cut}_max_abs"] = d
                    prefix = student(native[:, :, :cut].contiguous()).cpu()
                    checks[f"prefix_cut{cut}_max_abs"] = float((prefix - sfull[:, :, :cut]).abs().max())
                caches, seqs = None, []
                for i in range(native.shape[2]):
                    o, caches = student.forward_sequential(native[:, :, i:i + 1], caches)
                    seqs.append(o)
                checks["streaming_max_abs"] = float((torch.cat(seqs, 2).cpu() - sfull).abs().max())
                checks["passed"] = all(
                    v < args.tolerance for k, v in checks.items() if isinstance(v, float))
                summary["causality"][pair["sample_id"]] = checks
                print(json.dumps({"causality": pair["sample_id"], **checks}), flush=True)
            print(json.dumps({"student": pair["sample_id"], "latent_mse": rec["latent_mse"]}), flush=True)
            del native, tout, sfull
    del node, teacher; empty_cache(device)
    summary["status"] = "decoding"; atomic_json(args.output / "summary.json", summary)
    vae = load_video_vae(args.base_model, device)
    for pair in pairs:
        ident = pair["sample_id"]; directory = args.output / "decoded" / ident; directory.mkdir(parents=True)
        frames = {}
        for label in ("teacher", "student", "reference"):
            value = tensor_load(args.output / "latents" / ident / f"{label}.pt")["latent"]
            frames[label] = decode_latent(vae, value, device, keep_frames=120)
            if frames[label].shape != (120, 384, 672, 3): raise ValueError("bad decode geometry")
            write_mp4(frames[label], directory / f"{label}.mp4", fps=24.)
            print(json.dumps({"decoded": ident, "variant": label}), flush=True)
        row = {}
        for label in ("teacher", "student"):
            row[label] = {"vs_decoded_reference": metric(frames[label], frames["reference"])}
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
