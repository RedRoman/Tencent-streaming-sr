"""Original 345M H3 causal Conv3d student for decoder-aware adaptation.

The released upscaler parameters are retained. Causality comes from the existing
left-only Conv3d/per-frame GroupNorm adapter, not from a new CMN architecture.
Already aligned and fine-tuned weights are loaded verbatim: never shift again.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from causal_upscaler import CausalUpscalerAdapter
from data_utils import tensor_load
from prepare_data import load_author_upscaler
from train_upscaler_trial import enable_activation_checkpointing


MODEL_KIND = 'h3_causal_conv3d'
STATE_DICT_FORMAT = 'original_upscaler'
EXPECTED_PARAMETER_COUNT = 345280216


def validate_adapter_config(config):
    expected = dict(initialization='aligned_past', temporal_padding='left_only_zero',
                    groupnorm='per_frame', spatial_scale=2.0, temporal_chunking=False,
                    dtype='float32', saved_weights_already_initialized=True,
                    reload_must_not_reinitialize_weights=True)
    if not isinstance(config, dict) or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('Expected the already aligned, per-frame-normalized FP32 causal adapter')


@dataclass(frozen=True)
class CausalConv3dState:
    """Sequential stream token; caches remain owned by one non-reentrant adapter.

    ``memories`` exposes actual cached feature tensors for memory accounting.
    This token is not a portable branchable stream snapshot.
    """
    memories: tuple
    next_latent_index: int
    generation: int
    owner: int


class CausalConv3dStudent(nn.Module):
    def __init__(self, node, model, adapter_config, activation_checkpointing=False):
        super().__init__()
        validate_adapter_config(adapter_config)
        self.model = model  # Only the original upscaler is registered as a submodule.
        self.adapter = CausalUpscalerAdapter(node, model)
        self.adapter_config = dict(adapter_config)
        self.model_config = dict(kind=MODEL_KIND, latent_channels=24, spatial_scale=2,
                                 adapter_config=dict(adapter_config))
        self.activation_checkpointing = bool(activation_checkpointing)
        self._stream_generation = 0
        if activation_checkpointing:
            enable_activation_checkpointing(self.model)
        # Original Conv3d training disabled its real 0.1 dropout. Keep that
        # deterministic recipe; eval mode does not disable parameter gradients.
        self.model.eval().requires_grad_(True)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, native):
        if native.ndim != 5 or native.shape[1] != 24:
            raise ValueError('Expected native H3 latent [B,24,T,H,W]')
        return self.adapter.native_forward(native, (native.shape[2],
                                                   2*native.shape[3], 2*native.shape[4]))

    def reset_stream(self):
        self.adapter.reset_stream()
        self._stream_generation += 1

    def stream_state(self):
        return self.adapter.stream_state()

    @torch.no_grad()
    def stream_step(self, native, target_hw=None):
        target_hw = tuple(target_hw or (2*native.shape[-2], 2*native.shape[-1]))
        if target_hw != (2*native.shape[-2], 2*native.shape[-1]):
            raise ValueError('This student is fixed to spatial factor two')
        return self.adapter.stream_step(native, target_hw)

    @torch.no_grad()
    def forward_chunk(self, native, state=None, latent_index=0):
        if native.ndim != 5 or native.shape[1] != 24 or native.shape[2] < 1:
            raise ValueError('Expected a nonempty native H3 latent chunk [B,24,T,H,W]')
        if state is None:
            if latent_index != 0:
                raise ValueError('A new stream must begin at latent index zero')
            self.reset_stream()
        elif (not isinstance(state, CausalConv3dState) or state.owner != id(self)
              or state.generation != self._stream_generation
              or state.next_latent_index != latent_index
              or self.adapter.stream_state()['processed_latent_frames'] != latent_index):
            raise ValueError('Stale, foreign or out-of-order stream state')
        output = self.stream_step(native)
        state = CausalConv3dState(tuple(self.adapter._cache.values()),
                                 latent_index + native.shape[2], self._stream_generation, id(self))
        return output, state

    def forward_step(self, native, state=None, latent_index=0):
        if native.ndim != 4:
            raise ValueError('Expected one latent frame [B,24,H,W]')
        return self.forward_chunk(native.unsqueeze(2), state, latent_index)

    def original_state_dict(self):
        """Checkpoint contract: original parameter names, without model. prefix."""
        return self.model.state_dict()

    def load_original_state_dict(self, state):
        self.reset_stream()
        return self.model.load_state_dict(state, strict=True)

    def pixel_probe_parameter(self):
        name = 'conv_out.weight'
        parameter = dict(self.model.named_parameters()).get(name)
        if parameter is None or not isinstance(self.model.conv_out, nn.Conv3d):
            raise ValueError('Missing original Conv3d output head')
        return 'model.' + name, parameter


def _build_from_saved(saved, upstream, weights, device, activation_checkpointing):
    device = torch.device(device)
    config = saved.get('adapter_config')
    validate_adapter_config(config)
    node = load_author_upscaler(Path(upstream), Path(weights), device)
    model = node.load_model('upscaler', device, 'fp32')
    if sum(p.numel() for p in model.parameters()) != EXPECTED_PARAMETER_COUNT:
        raise ValueError('Expected the original 345280216-parameter H3 Conv3d network')
    # Deliberately do not call initialize_aligned_past_: saved weights have
    # already been transformed and optimized in the earlier causal eWarp run.
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model._aligned_past_initialized = True
    return CausalConv3dStudent(node, model, config, activation_checkpointing)


def build_student(initial_path, upstream, weights, device, activation_checkpointing=False):
    """Warm-start ONLY from the completed old causal eWarp checkpoint."""
    saved = tensor_load(Path(initial_path))
    if saved.get('family') != 'ewarp' or 'model_config' in saved:
        raise ValueError('Expected the original causal eWarp checkpoint, not a CMN or hybrid checkpoint')
    return _build_from_saved(saved, upstream, weights, device, activation_checkpointing)


def load_student_checkpoint(path, upstream, weights, device, activation_checkpointing=False):
    """Load this experiment's explicit original-parameter-name checkpoint format."""
    saved = tensor_load(Path(path))
    expected = dict(kind=MODEL_KIND, latent_channels=24, spatial_scale=2,
                    adapter_config=saved.get('adapter_config'))
    if saved.get('model_config') != expected or saved.get('state_dict_format') != STATE_DICT_FORMAT:
        raise ValueError('Wrong student architecture or state-dict format')
    return _build_from_saved(saved, upstream, weights, device, activation_checkpointing)
