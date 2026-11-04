"""Small detached CPU traces for comparing H3 forward paths.

No model construction, device initialization or file writes occur here. Hooks
observe module inputs/outputs without modifying them. The caller controls which
forward is traced and is responsible for removing the returned handles.
"""
from __future__ import annotations

import torch


def _first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def attach_trace(model, static_rows: int, capture_dict: dict):
    """Attach tracing hooks and return their removable PyTorch handles.

    Usage::

        captured = {}
        handles = attach_trace(runtime.model, static_rows, captured)
        try:
            value = runtime.forward(...)
        finally:
            for handle in handles:
                handle.remove()

    ``static_rows`` is the number of packed text+audio rows, i.e. the index of
    the first video row. Packed tensors sample six shared rows: the first two
    static rows, last two static rows, and first two video rows. Video-only and
    text-only tensors sample their first six rows. Time tensors sample their
    first six timestep rows (normally there are only two). All feature axes are
    retained. Only batch item zero is retained, with its batch axis preserved.

    Each entry contains ``shape`` (the original full shape), ``dtype``,
    ``device``, ``row_axis``, ``row_indices``, and a detached CPU ``sample``.
    A key is ``<module path>.input`` or ``<module path>.output``. If the same
    module runs repeatedly, subsequent keys have ``#2``, ``#3``, etc.; traces
    are never silently overwritten. Use a fresh dict or clear it between calls.

    This collects actual values, not a diagnosis: small sampled rows cannot
    prove equality of every hidden element or explain a divergence by itself.
    """
    if not isinstance(static_rows, int) or static_rows < 2:
        raise ValueError("static_rows must be the positive packed text+audio row count (>=2)")
    if not isinstance(capture_dict, dict):
        raise TypeError("capture_dict must be a dict")
    handles = []
    packed_rows = list(dict.fromkeys((0, 1, static_rows - 2, static_rows - 1,
                                    static_rows, static_rows + 1)))

    def capture(key, value, row_kind):
        tensor = _first_tensor(value)
        if tensor is None:
            raise TypeError(f"{key}: no tensor found in module value")
        row_axis = 0 if row_kind == "time" else 1
        if tensor.ndim <= row_axis:
            raise ValueError(f"{key}: expected a sequence tensor, got {tuple(tensor.shape)}")
        length = tensor.shape[row_axis]
        indices = packed_rows if row_kind == "packed" else list(range(min(6, length)))
        if not indices or max(indices) >= length:
            raise ValueError(f"{key}: selected rows {indices} do not fit shape {tuple(tensor.shape)}")
        with torch.no_grad():
            # Select on the original device before transfer: no full activation
            # reaches the CPU, and no autograd graph is retained.
            selected = tensor.detach()
            if row_axis == 1:
                selected = selected[:1]
            index = torch.tensor(indices, dtype=torch.long, device=tensor.device)
            sample = selected.index_select(row_axis, index).cpu().clone()
        name, occurrence = key, 2
        while name in capture_dict:
            name = f"{key}#{occurrence}"
            occurrence += 1
        capture_dict[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                             "device": str(tensor.device), "row_axis": row_axis,
                             "row_indices": indices, "sample": sample}

    def attach(path, row_kind, *, inputs=True, outputs=True):
        module = model.get_submodule(path)
        if inputs:
            def before(_module, args, kwargs, _path=path, _kind=row_kind):
                # Current H3 modules pass their activation positionally; the
                # fallback also covers callers using hidden_states by keyword.
                activation = args[0] if args else kwargs.get("hidden_states")
                if activation is None:
                    activation = kwargs
                capture(f"{_path}.input", activation, _kind)
            handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
        if outputs:
            def after(_module, _args, result, _path=path, _kind=row_kind):
                capture(f"{_path}.output", result, _kind)
            handles.append(module.register_forward_hook(after))

    try:
        for path, kind in (("proj_in", "video"), ("context_embedder", "text"),
                           ("token_refiner", "text"), ("time_embedder", "time")):
            attach(path, kind)
        first = "transformer_blocks.0"
        attach(first, "packed", outputs=False)
        for suffix in ("norm1", "attn.to_q", "attn.to_k", "attn.to_v",
                       "attn.norm_q", "attn.norm_k", "attn.to_out.0",
                       "attn", "norm2", "ff", "adaln_proj"):
            attach(f"{first}.{suffix}", "time" if suffix == "adaln_proj" else "packed")
        for index in range(len(model.transformer_blocks)):
            attach(f"transformer_blocks.{index}", "packed", inputs=False)
        attach("norm_out", "packed")
        attach("proj_out", "packed")
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    return handles
