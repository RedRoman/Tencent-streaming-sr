"""Explicit causal operator surgery for the released LBH/Open-H3 upscaler.

The adapter retains parameter/state-dict names and shapes. Left padding and
per-frame GroupNorm change the model's function, so this is not an equivalent
implementation of the pretrained noncausal network. The optional aligned-past
initialization also changes weights, discarding pretrained future coefficients.

This provides strict latent-frame causality only. The H3 VAE and any subsequent
diffusion refiner need independent causality treatment for an end-to-end stream.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def temporal_structure(model):
    """Describe the longest sequential convolution path in the released model.

    For this ResBlock stack, skip paths are identity/1x1 and do not enlarge the
    field. GroupNorm's original whole-clip dependency is reported separately.
    """
    kernels = {}
    history = 0
    for module in model.modules():
        if isinstance(module, nn.Conv3d) and module.kernel_size[0] > 1:
            kernel = module.kernel_size[0]
            kernels[str(kernel)] = kernels.get(str(kernel), 0) + 1
            history += (kernel - 1) * module.dilation[0]
    return {"temporal_convolution_counts_by_kernel": kernels,
            "causal_history_latent_frames": history,
            "longest_convolution_receptive_field": history + 1,
            "original_groupnorm_dependency": "whole input clip",
            "causal_groupnorm_dependency": "current frame channels and spatial dimensions"}


@torch.no_grad()
def initialize_aligned_past_(model):
    """Preserve old past/current offsets, drop future taps, zero older taps.

    k=3: [past,current,future] -> [0,past,current].
    k=5: [a,b,current,d,e] -> [0,0,a,b,current].
    This mutates weights; reload original weights to undo it. Apply only once.
    """
    if getattr(model, "_aligned_past_initialized", False):
        raise RuntimeError("Aligned-past initialization already applied to this model")
    converted = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv3d) or module.kernel_size[0] == 1:
            continue
        kernel = module.kernel_size[0]
        if kernel % 2 != 1 or module.stride[0] != 1:
            raise ValueError("Aligned-past initialization requires odd kernels and temporal stride 1")
        center = kernel // 2
        original = module.weight.detach().clone()
        module.weight.zero_()
        module.weight[:, :, center:].copy_(original[:, :, :center + 1])
        converted.append({"module": name, "kernel": kernel, "discarded_future_slices": center})
    model._aligned_past_initialized = True
    return {"kind": "aligned_past", "converted": converted,
            "description": "Old past/current kernel offsets retained; old future taps discarded; older causal taps zeroed"}


class CausalUpscalerAdapter:
    """Install causal forwards without adding or renaming parameters/buffers.

    Call native_forward for differentiable whole-clip execution. stream_step is
    inference-only and maintains each temporal convolution's actual input cache;
    work per chunk does not grow with elapsed stream length. Use reset_stream
    before a new clip or after changing weights. Concurrent/reentrant use of one
    model is unsupported.
    """

    def __init__(self, node, model):
        if getattr(model, "_causal_upscaler_adapter", None) is not None:
            raise RuntimeError("This model already has a causal adapter")
        # The published checkpoint has no attention. Unknown attention modules
        # cannot be silently treated as causal by patching Conv/GroupNorm alone.
        if any("attn" in type(module).__name__.lower() or isinstance(module, nn.MultiheadAttention)
               for module in model.modules()):
            raise ValueError("Attention-containing upscalers need a separate causal attention implementation")
        self.node = node
        self.model = model
        self._original_forwards = []
        self._cache = {}
        self._streaming = False
        self._stream_signature = None
        self._stream_frames = 0
        self._removed = False
        self.structure = temporal_structure(model)
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv3d):
                if module.stride[0] != 1 or module.padding_mode != "zeros":
                    raise ValueError("Only stride-1 zero-padded temporal convolutions are supported")
                if module.kernel_size[0] > 1 and (
                    module.kernel_size[0] % 2 != 1
                    or module.padding[0] != module.dilation[0] * (module.kernel_size[0] // 2)
                ):
                    raise ValueError("Expected the released odd, symmetric temporal kernels")
            if isinstance(module, nn.GroupNorm):
                original = module.forward
                self._original_forwards.append((module, original))

                def norm_forward(value, _module=module):
                    if value.ndim != 5:
                        raise ValueError("Causal GroupNorm expects [B,C,T,H,W]")
                    batch, channels, frames, height, width = value.shape
                    value = value.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
                    value = F.group_norm(value, _module.num_groups, _module.weight, _module.bias, _module.eps)
                    return value.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()

                module.forward = norm_forward
            elif isinstance(module, nn.Conv3d) and module.kernel_size[0] > 1:
                original = module.forward
                self._original_forwards.append((module, original))

                def conv_forward(value, _module=module, _name=name):
                    history = (_module.kernel_size[0] - 1) * _module.dilation[0]
                    if self._streaming:
                        prefix = self._cache.get(_name)
                        if prefix is None:
                            prefix = value.new_zeros((*value.shape[:2], history, *value.shape[-2:]))
                        if (prefix.shape[:2] != value.shape[:2] or prefix.shape[-2:] != value.shape[-2:]
                                or prefix.dtype != value.dtype or prefix.device != value.device):
                            raise ValueError("Stream shape/device/dtype changed; reset_stream first")
                        value = torch.cat((prefix, value), dim=2)
                        # Clone prevents a tiny cache from retaining a large chunk's storage.
                        self._cache[_name] = value[:, :, -history:].detach().clone()
                    else:
                        value = F.pad(value, (0, 0, 0, 0, history, 0))
                    return F.conv3d(value, _module.weight, _module.bias, _module.stride,
                                    (0, *_module.padding[1:]), _module.dilation, _module.groups)

                module.forward = conv_forward
        model._causal_upscaler_adapter = self

    def native_forward(self, native, target_size):
        """Open-H3 normalized z -> D(U(N(z))), with time length unchanged."""
        if self._removed:
            raise RuntimeError("Adapter removed")
        if native.ndim != 5 or len(target_size) != 3 or target_size[0] != native.shape[2]:
            raise ValueError("Expected [B,C,T,H,W] and target (same T,H,W)")
        if native.shape[2] < 1 or any(int(value) != value or value < 1 for value in target_size):
            raise ValueError("Nonempty integer target dimensions required")
        if target_size[-2] < native.shape[-2] or target_size[-1] < native.shape[-1]:
            raise ValueError("This adapter only supports spatial upscaling")
        mean, std = self.node._make_norm_tensors(native.device, native.dtype)
        scale = (target_size[-2] / native.shape[-2] + target_size[-1] / native.shape[-1]) / 2.0
        output = self.model((native - mean) / std, scale=scale, target_size=tuple(target_size),
                            enable_chunking=False)
        return output * std + mean

    @torch.inference_mode()
    def stream_step(self, native, target_hw):
        if self._streaming:
            raise RuntimeError("Nested/concurrent streaming calls are unsupported")
        if self.model.training:
            raise RuntimeError("Streaming requires model.eval() for deterministic dropout-free execution")
        if native.ndim != 5 or len(target_hw) != 2:
            raise ValueError("Expected [B,C,T,H,W] and target (H,W)")
        signature = (native.shape[0], native.shape[1], *native.shape[-2:], *target_hw,
                     str(native.device), native.dtype)
        if self._stream_signature is not None and signature != self._stream_signature:
            raise ValueError("Stream geometry/device/dtype changed; reset_stream first")
        self._stream_signature = signature
        self._streaming = True
        try:
            output = self.native_forward(native, (native.shape[2], *target_hw))
            self._stream_frames += native.shape[2]
            return output
        except Exception:
            # A partial forward may already have advanced a subset of caches.
            self.reset_stream()
            raise
        finally:
            self._streaming = False

    def reset_stream(self):
        self._cache.clear()
        self._stream_signature = None
        self._stream_frames = 0

    def stream_state(self):
        return {"processed_latent_frames": self._stream_frames,
                "cached_convolutions": len(self._cache),
                "cache_bytes": sum(value.numel() * value.element_size() for value in self._cache.values()),
                "cache_temporal_lengths": {name: value.shape[2] for name, value in self._cache.items()}}

    def remove(self):
        """Restore original forwards; this does not undo aligned-past weights."""
        if not self._removed:
            self.reset_stream()
            for module, original in self._original_forwards:
                module.forward = original
            self._original_forwards.clear()
            delattr(self.model, "_causal_upscaler_adapter")
            self._removed = True
