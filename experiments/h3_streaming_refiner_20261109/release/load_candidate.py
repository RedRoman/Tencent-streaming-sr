"""Load the two published inference-only candidates without private resume files.

This is a CPU artifact loader, not a full-model inference acceptance test.
It leaves the frozen runtime and evaluation stages unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

PINNED = {
    1000: "f16119620e6bcbce0c6da7ce00560dc1733e9d0b3c0d6d4107c38974d43c50a6",
    2000: "5e95b0826db51003968597b8ffbc6159f42d0c784ad8c0a10843734fae3dc485",
}
UPSCALER = "7b9f43c17462fb2d0554645bc7889dc7abdebc052ee9f18f6780624de912e0b5"
TRAINING_MANIFEST = "1ac8ae9c82a052386cfff984fa1adbad07c4740383770e7ec31afd2144c08a0a"
KEYS = {"format", "base_step", "stage_step", "step", "adapter", "residual_head",
        "residual_bound", "upsampler_sha256", "manifest_sha256"}


def load_candidate(path, *, stage):
    """Verify original bytes and read tensors on CPU. Never choose latest."""
    path = Path(path)
    if stage not in PINNED:
        raise ValueError("Choose the explicit new-stage checkpoint 1000 or 2000")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if path.stat().st_size != 82508131 or actual != PINNED[stage]:
        raise ValueError("Pinned candidate size/SHA mismatch; no fallback checkpoint")
    weights = torch.load(path, map_location="cpu", weights_only=True)
    if set(weights) != KEYS:
        raise ValueError("Unexpected checkpoint fields")
    expected = {"format": "anchored-streaming-refiner-v1", "base_step": 2500,
                "stage_step": stage, "step": 2500 + stage, "residual_bound": 0.25,
                "upsampler_sha256": UPSCALER, "manifest_sha256": TRAINING_MANIFEST}
    if any(weights[k] != v for k, v in expected.items()):
        raise ValueError("Checkpoint provenance mismatch")
    if len(weights["adapter"]) != 401 or len(weights["residual_head"]) != 4:
        raise ValueError("Incomplete adapter or residual head")
    tensors = list(weights["adapter"].values()) + list(weights["residual_head"].values())
    if not all(isinstance(t, torch.Tensor) and t.device.type == "cpu" and
               t.dtype == torch.float32 and torch.isfinite(t).all() for t in tensors):
        raise ValueError("Invalid inference tensors")
    if sum(t.numel() for t in tensors) != 20593528:
        raise ValueError("Unexpected parameter count")
    return weights


def load_head(weights, *, device="cpu"):
    """Install all four head tensors using the unchanged experiment class."""
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from anchored_refiner_v1 import ResidualHead
    head = ResidualHead(device=device, bound=weights["residual_bound"])
    head.load_state_dict(weights["residual_head"], strict=True)
    return head.eval().requires_grad_(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--stage", type=int, required=True, choices=sorted(PINNED))
    args = parser.parse_args()
    weights = load_candidate(args.checkpoint, stage=args.stage)
    head = load_head(weights)
    if not all(torch.equal(v.cpu(), weights["residual_head"][k])
               for k, v in head.state_dict().items()):
        raise ValueError("Residual-head load changed tensor values")
    print(json.dumps({"status": "CPU_checkpoint_and_head_load_passed",
                      "stage_step": args.stage, "cumulative_updates": 2500 + args.stage,
                      "sha256": PINNED[args.stage], "tensors": 405,
                      "parameters": 20593528, "accelerator_forwards": 0,
                      "model_accepted": False}, indent=2))


if __name__ == "__main__":
    main()
