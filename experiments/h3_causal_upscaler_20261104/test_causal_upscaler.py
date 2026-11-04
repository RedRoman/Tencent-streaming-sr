"""CPU semantic checks on the actual author architecture at reduced width/depth.

No checkpoint or accelerator is loaded. This checks dependency and cache
semantics; it does not assess pretrained full-model quality or NPU performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from causal_upscaler import CausalUpscalerAdapter, initialize_aligned_past_, temporal_structure
from prepare_data import load_author_upscaler


def check_close(a, b, label, tolerance=2e-5):
    error = float((a - b).abs().max())
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()) or error > tolerance:
        raise AssertionError(f"{label}: max absolute error {error}, tolerance {tolerance}")
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    node = load_author_upscaler(args.upstream, Path("unused-no-checkpoint-loaded"), torch.device("cpu"))
    # Inspect the full released architecture without allocating its 345M weights.
    with torch.device("meta"):
        full_architecture = node.LatentResizer3D()
    full_structure = temporal_structure(full_architecture)
    assert sum(value.numel() for value in full_architecture.parameters()) == 345280216
    assert full_structure["temporal_convolution_counts_by_kernel"] == {"3": 50, "5": 12}
    assert full_structure["longest_convolution_receptive_field"] == 149
    del full_architecture
    model = node.LatentResizer3D(in_channels=24, channels=32, in_blocks=2, out_blocks=2,
                                dropout=.1, temporal_every=2, temporal_kernel=5, attn=False).eval()
    # The author's constructor zero-initializes residual tails. Activate them so
    # dependency tests exercise every temporal convolution instead of an identity.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "out_layers.2.weight" in name or "pwconv.weight" in name:
                parameter.normal_(0, .02)
    original_state = {name: value.clone() for name, value in model.state_dict().items()}
    native = torch.randn(1, 24, 9, 3, 4)
    altered = native.clone()
    altered[:, :, 4:] = torch.randn_like(altered[:, :, 4:]) * 5 + 3
    results = {"device": "cpu", "architecture": "actual author classes, channels32, in2/out2",
               "full_architecture_structure": full_structure,
               "full_model_quality_verified": False, "variants": {}}
    with torch.no_grad():
        mean, std = node._make_norm_tensors(native.device, native.dtype)
        original_output = model((native-mean)/std, scale=2., target_size=(9, 6, 8), enable_chunking=False)*std+mean
        original_changed = model((altered-mean)/std, scale=2., target_size=(9, 6, 8), enable_chunking=False)*std+mean
    noncausal_delta = float((original_output[:, :, :4] - original_changed[:, :, :4]).abs().max())
    if noncausal_delta <= 1e-5:
        raise AssertionError("Noncausal negative control failed to detect future dependence")
    results["noncausal_negative_control_future_delta"] = noncausal_delta
    for initialization in ("unchanged", "aligned_past"):
        model.load_state_dict(original_state, strict=True)
        if initialization == "aligned_past":
            initialization_report = initialize_aligned_past_(model)
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv3d) and module.kernel_size[0] > 1:
                    middle = module.kernel_size[0] // 2
                    assert torch.count_nonzero(module.weight[:, :, :middle]) == 0
                    assert torch.equal(module.weight[:, :, middle:], original_state[name + ".weight"][:, :, :middle+1])
        else:
            initialization_report = {"kind": "unchanged"}
        state_before = {name: value.clone() for name, value in model.state_dict().items()}
        adapter = CausalUpscalerAdapter(node, model)
        assert state_before.keys() == model.state_dict().keys()
        assert all(torch.equal(value, model.state_dict()[name]) for name, value in state_before.items())
        with torch.no_grad():
            full = adapter.native_forward(native, (9, 6, 8))
            changed = adapter.native_forward(altered, (9, 6, 8))
            prefix = adapter.native_forward(native[:, :, :4], (4, 6, 8))
        result = {"initialization": initialization_report, "structure": adapter.structure,
                  "future_perturbation_max_abs": check_close(full[:, :, :4], changed[:, :, :4], "future invariance"),
                  "prefix_max_abs": check_close(full[:, :, :4], prefix, "prefix invariance"), "chunks": []}
        for chunks in ([1]*9, [2, 1, 4, 2], [5, 4], [9]):
            adapter.reset_stream()
            outputs, position = [], 0
            for frames in chunks:
                outputs.append(adapter.stream_step(native[:, :, position:position+frames], (6, 8)))
                position += frames
            result["chunks"].append({"partition": chunks,
                                     "max_abs": check_close(torch.cat(outputs, dim=2), full, "stream/full equivalence"),
                                     "state": adapter.stream_state()})
        # A new clip after reset must produce the same output regardless of the
        # previous stream, and stream inference must leave all weight values intact.
        adapter.reset_stream()
        check_close(adapter.stream_step(native, (6, 8)), full, "reset isolation")
        assert all(torch.equal(value, model.state_dict()[name]) for name, value in state_before.items())
        differentiable = native.clone().requires_grad_(True)
        output = adapter.native_forward(differentiable, (9, 6, 8))
        output[:, :, 3].square().mean().backward()
        result["future_input_gradient_max_abs"] = float(differentiable.grad[:, :, 4:].abs().max())
        result["current_input_gradient_max_abs"] = float(differentiable.grad[:, :, 3].abs().max())
        assert result["future_input_gradient_max_abs"] == 0
        assert result["current_input_gradient_max_abs"] > 0
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        adapter.remove()
        if initialization == "unchanged":
            with torch.no_grad():
                restored = model((native-mean)/std, scale=2., target_size=(9, 6, 8), enable_chunking=False)*std+mean
            result["remove_restores_original_max_abs"] = check_close(restored, original_output, "restore original")
        results["variants"][initialization] = result
        model.zero_grad(set_to_none=True)
    results["status"] = "passed"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
