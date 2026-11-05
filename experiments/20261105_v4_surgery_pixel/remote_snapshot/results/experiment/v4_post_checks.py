#!/usr/bin/env python
"""Post-training gates + summary for a v4 surgery checkpoint.

Re-runs future_cut / prefix_cut / streaming causality checks (prefix_cut target
fixed: the prefix forward upscales only the T=cut prefix), final_vs_teacher on
all clips, and writes train_summary.json next to the checkpoint.
"""
import argparse, json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiment.data_utils import select_device, tensor_load, video_tensor, empty_cache
from experiment.train_upscaler_trial import read_pairs, wrapper_forward, metrics, atomic_json
from experiment.prepare_data import load_author_upscaler
from experiment.causal_upscaler import CausalUpscalerAdapter

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--pairs", type=Path, default=Path("upscaler_5s_v4exp/pairs"))
    p.add_argument("--upstream", type=Path, default=Path(__file__).resolve().parent.parent / "upstream")
    p.add_argument("--upscaler", type=Path, default=Path(
        "assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"))
    p.add_argument("--device", default="npu:0")
    args = p.parse_args()
    device = select_device(args.device)
    ckpt = args.checkpoint or (args.run_dir / "checkpoint" / "v4.pt")
    saved = tensor_load(ckpt)
    node = load_author_upscaler(args.upstream, args.upscaler, device)
    model = node.load_model('upscaler', device, 'fp32')
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model._aligned_past_initialized = True
    del saved
    adapter = CausalUpscalerAdapter(node, model)
    model.eval().requires_grad_(False)

    tnode = load_author_upscaler(args.upstream, args.upscaler, device)
    teacher = tnode.load_model("upscaler", device, "fp32")
    for p_ in teacher.parameters():
        p_.requires_grad_(False)

    def load(prs):
        return [{"id": pr["sample_id"],
                 "draft": video_tensor(tensor_load(pr["path"])["draft"]).float(),
                 "reference": video_tensor(tensor_load(pr["path"])["reference"]).float()}
                for pr in prs]
    samples = load(read_pairs(args.pairs / "train.jsonl", "train"))
    hold = load(read_pairs(args.pairs / "holdout.jsonl", "holdout"))

    s = hold[0]
    draft = s["draft"][None].to(device)
    full = adapter.native_forward(draft, tuple(s["reference"].shape[-3:]))
    checks = {}
    with torch.no_grad():
        for cut in (1, 13, 25):
            changed = draft.clone()
            changed[:, :, cut:] = 3 * torch.randn(changed[:, :, cut:].shape,
                generator=torch.Generator().manual_seed(20261104 + cut)).to(device)
            actual = adapter.native_forward(changed, tuple(s["reference"].shape[-3:]))
            checks[f"future_cut{cut}_max_abs"] = float((actual[:, :, :cut] - full[:, :, :cut]).abs().max())
            # fixed: prefix target uses the prefix's own T
            prefix = adapter.native_forward(draft[:, :, :cut].contiguous(),
                                            (cut, s["reference"].shape[-2], s["reference"].shape[-1]))
            checks[f"prefix_cut{cut}_max_abs"] = float((prefix - full[:, :, :cut]).abs().max())
        adapter.reset_stream()
        seqs = [adapter.stream_step(draft[:, :, i:i+1], (draft.shape[3]*2, draft.shape[4]*2))
                for i in range(draft.shape[2])]
        checks["streaming_max_abs"] = float((torch.cat(seqs, dim=2) - full).abs().max())
    checks["passed"] = all(v < 1e-2 for k, v in checks.items() if isinstance(v, float))
    print(json.dumps({"causality": checks}), flush=True)

    final = {}
    with torch.no_grad():
        for s2 in samples + hold:
            d2 = s2["draft"][None].to(device)
            te2 = wrapper_forward(tnode, teacher, d2, tuple(s2["reference"].shape[-3:]))
            m = metrics(adapter.native_forward(d2, tuple(s2["reference"].shape[-3:]))[0],
                        te2[0], 0.1)
            final[s2["id"]] = {k: round(float(v), 5) for k, v in m.items()}
            print(json.dumps({"final_vs_teacher": s2["id"], **final[s2["id"]]}), flush=True)
    atomic_json(args.run_dir / "train_summary.json", {
        "checkpoint": str(ckpt), "causality": checks, "final_vs_teacher": final,
        "passed": checks["passed"]})
    empty_cache(device)

if __name__ == "__main__":
    main()
