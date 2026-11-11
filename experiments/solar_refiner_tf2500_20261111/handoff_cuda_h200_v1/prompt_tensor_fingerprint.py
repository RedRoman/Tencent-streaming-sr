"""CPU-only tensor fingerprints for a trusted restoration_prompt.pt cache.

The .pt container hash is informational when comparing reserialized files.
Exact tensor comparison includes names, shapes, dtypes and logical C-order bytes.
No model is loaded and the input cache is never modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def fingerprint(path):
    import torch

    if sys.byteorder != "little":
        raise RuntimeError("v1 requires a little-endian CPU; do not silently hash host-endian bytes")
    torch.set_num_threads(1)
    path = Path(path)
    before = sha(path.read_bytes())
    data = torch.load(path, map_location="cpu", weights_only=True)
    expected_keys = {"prompt", "prompt_embeds", "text_token_tags", "encoder_layer",
                     "add_special_tokens", "presentation"}
    if not isinstance(data, dict) or set(data) != expected_keys:
        raise ValueError("Expected the six original prompt-cache keys; report schema changes explicitly")
    metadata = {k: data[k] for k in ("prompt", "encoder_layer", "add_special_tokens", "presentation")}
    if not isinstance(metadata["prompt"], str):
        raise TypeError("prompt must be text")
    if type(metadata["encoder_layer"]) is not int or type(metadata["add_special_tokens"]) is not bool:
        raise TypeError("encoder_layer/add_special_tokens must retain their original types")
    if not isinstance(metadata["presentation"], str):
        raise TypeError("presentation must be text")
    rows = []
    for name in ("prompt_embeds", "text_token_tags"):
        tensor = data[name]
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise TypeError(f"{name} must be a dense strided tensor")
        tensor = tensor.detach().to("cpu").contiguous()
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} contains non-finite values")
        header = {"name": name, "dtype": str(tensor.dtype).removeprefix("torch."),
                  "shape": list(tensor.shape), "byte_order": "little", "order": "C"}
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        row = {**header, "numel": tensor.numel(), "bytes": len(raw),
               "raw_sha256": sha(raw), "bound_sha256": sha(canonical(header) + b"\x00" + raw)}
        if tensor.is_floating_point():
            # Secondary value fingerprint: BF16 -> FP32 is exact, without decimal formatting.
            f32 = tensor.to(torch.float32).contiguous().reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
            row["float32_value_sha256"] = sha(f32)
        rows.append(row)
    if sha(path.read_bytes()) != before:
        raise RuntimeError("Input file changed while reading")
    binding = {"metadata": metadata,
               "tensors": [{"name": r["name"], "bound_sha256": r["bound_sha256"]} for r in rows]}
    return {"schema": "solar-restoration-prompt-tensor-fingerprint-v1",
            "hash_definition": {
                "raw": "SHA256 of CPU contiguous logical C-order tensor payload bytes; little-endian; no .pt header, storage padding, stride or device",
                "bound": "SHA256(canonical_json({name,dtype,shape,byte_order,order}) UTF-8 + one NUL byte + raw payload)",
                "canonical_json": "json.dumps(sort_keys=True,separators=(',',':'),ensure_ascii=False)",
                "bundle": "SHA256(canonical_json({metadata,tensors:[{name,bound_sha256},...]})); order prompt_embeds then text_token_tags",
                "float32_value": "SHA256 of the floating tensor cast losslessly from BF16 to FP32, CPU contiguous C-order little-endian bytes; auxiliary only",
            },
            "source_file_sha256": before, "source_file_bytes": path.stat().st_size,
            "metadata": metadata, "prompt_utf8_sha256": sha(metadata["prompt"].encode("utf-8")),
            "tensors": rows, "bundle_sha256": sha(canonical(binding))}


def compare(actual, reference):
    if reference.get("schema") != actual["schema"] or reference.get("hash_definition") != actual["hash_definition"]:
        raise ValueError("Fingerprint schema/hash-definition mismatch")
    comparisons = []
    expected_rows = {r["name"]: r for r in reference["tensors"]}
    for row in actual["tensors"]:
        expected = expected_rows[row["name"]]
        comparisons.append({"name": row["name"],
                            "exact_tensor_match": row["bound_sha256"] == expected["bound_sha256"],
                            "float32_value_match": (row.get("float32_value_sha256") == expected.get("float32_value_sha256"))
                                if "float32_value_sha256" in expected else None})
    exact = (actual["bundle_sha256"] == reference["bundle_sha256"] and
             actual["metadata"] == reference["metadata"] and
             all(r["exact_tensor_match"] for r in comparisons))
    return {"exact_condition_match": exact,
            "container_sha256_match": actual["source_file_sha256"] == reference["source_file_sha256"],
            "tensors": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt_file", type=Path)
    parser.add_argument("--reference", type=Path, help="Compare exact tensor values and metadata; .pt file hash need not match")
    args = parser.parse_args()
    result = fingerprint(args.prompt_file)
    if args.reference:
        result["comparison"] = compare(result, json.loads(args.reference.read_text()))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.reference and not result["comparison"]["exact_condition_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
