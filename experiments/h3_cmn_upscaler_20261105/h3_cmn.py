"""Small CMN-inspired H3 latent upsampler, independently implemented.

Inspired by Ultra Flash v4 section 2.2, not a full architecture/training-recipe
reproduction. Three stages use Conv2d current/previous-input feature fusion,
three-convolution ReLU residual blocks, and spatial factors (1, 2, 1).
The input/output are native normalized H3 latents; no extra author-wrapper N/D
is applied. A zero-initialized output head adds a learned residual to bilinear
2x interpolation. This module does not implement or train a decoder.

States are explicit, immutable containers. Memory retains its graph unless the
caller invokes detach_state for TBPTT. There is no implicit five-token reset,
normalization across time, dropout, or hidden module-owned inference cache.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CMNState:
    memories: tuple[Tensor, ...]
    next_latent_index: int
    input_shape: tuple[int, int, int, int]
    dtype: torch.dtype
    device: torch.device
    owner: int


def detach_state(state: Optional[CMNState]) -> Optional[CMNState]:
    """Keep stream position and values, cut previous-chunk autograd history."""
    if state is None:
        return None
    if not isinstance(state, CMNState):
        raise TypeError('Expected CMNState or None')
    return CMNState(tuple(value.detach() for value in state.memories),
                    state.next_latent_index, state.input_shape,
                    state.dtype, state.device, state.owner)


def state_nbytes(state: Optional[CMNState]) -> int:
    """Logical cache tensor bytes; excludes graph/allocator overhead."""
    return 0 if state is None else sum(value.numel()*value.element_size() for value in state.memories)


def _frames(x: Tensor) -> Tensor:
    b, c, t, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w)


def _sequence(x: Tensor, batch: int, frames: int) -> Tensor:
    _, c, h, w = x.shape
    return x.reshape(batch, frames, c, h, w).permute(0, 2, 1, 3, 4)


class CausalMemoryBlock(nn.Module):
    """Each block remembers its previous *input*, not recurrent output.

    y_t = ReLU(conv3(ReLU(conv2(ReLU(conv1([x_t, x_{t-1}]))))) + x_t).
    That choice admits exact dependency-equivalent B*T vectorization. A block
    has one-frame memory; depth propagates a longer, strictly past-only context.
    Initial memory is zero: a local choice, differing from the official
    CausalMemBlock default that replicates its first input feature.
    """
    def __init__(self, width: int):
        super().__init__()
        self.conv1 = nn.Conv2d(2*width, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.conv3 = nn.Conv2d(width, width, 3, padding=1)

    def _transform(self, current: Tensor, previous: Tensor) -> Tensor:
        hidden = F.relu(self.conv1(torch.cat((current, previous), dim=1)))
        hidden = F.relu(self.conv2(hidden))
        return F.relu(current + self.conv3(hidden))

    def forward_step(self, x: Tensor, memory: Optional[Tensor] = None):
        previous = torch.zeros_like(x) if memory is None else memory
        if previous.shape != x.shape or previous.dtype != x.dtype or previous.device != x.device:
            raise ValueError('Memory does not match the current feature; reset the stream')
        return self._transform(x, previous), x

    def forward_sequence(self, x: Tensor, memory: Optional[Tensor] = None):
        b, _, t, _, _ = x.shape
        first = torch.zeros_like(x[:, :, 0]) if memory is None else memory
        if first.shape != x[:, :, 0].shape or first.dtype != x.dtype or first.device != x.device:
            raise ValueError('Memory does not match the current feature; reset the stream')
        previous = torch.cat((first.unsqueeze(2), x[:, :, :-1]), dim=2)
        result = self._transform(_frames(x), _frames(previous))
        # Clone keeps only one frame's storage during no-grad streaming; it does
        # not detach history during training/TBPTT.
        return _sequence(result, b, t), x[:, :, -1].clone()


class _Stage(nn.Module):
    def __init__(self, width: int, blocks: int, spatial_factor: int):
        super().__init__()
        self.blocks = nn.ModuleList(CausalMemoryBlock(width) for _ in range(blocks))
        self.spatial = (nn.Sequential(nn.Conv2d(width, width*spatial_factor**2, 3, padding=1),
                                      nn.PixelShuffle(spatial_factor))
                        if spatial_factor != 1 else nn.Identity())


class H3LatentUpsampler(nn.Module):
    """24-channel native H3 latents -> 2x spatial, same temporal resolution.

    forward:       B,24,T,H,W -> B,24,T,2H,2W (fresh stream)
    forward_chunk: B,24,T,H,W + state -> (output, next_state), vectorized B*T
    forward_step:  B,24,H,W + state -> (B,24,1,2H,2W, next_state)

    With state=None, latent_index may set the global start of a deliberately
    cold-started window. With a state, it must equal next_latent_index; omitting
    it continues automatically. Reset by passing None, never by phase modulo5.
    """
    spatial_factors = (1, 2, 1)
    temporal_factors = (1, 1, 1)

    def __init__(self, width: int = 64, blocks: int = 2):
        super().__init__()
        if type(width) is not int or width < 1 or type(blocks) is not int or blocks < 1:
            raise ValueError('width and blocks must be positive integers')
        self.width, self.blocks = width, blocks
        self.input = nn.Conv2d(24, width, 3, padding=1)
        self.stages = nn.ModuleList(_Stage(width, blocks, factor) for factor in self.spatial_factors)
        self.output = nn.Conv2d(width, 24, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.config = {
            'architecture': 'H3 CMN-inspired latent upsampler', 'version': 1,
            'width': width, 'blocks_per_stage': blocks, 'stages': 3,
            'input_channels': 24, 'output_channels': 24,
            'spatial_factors': list(self.spatial_factors), 'temporal_factors': list(self.temporal_factors),
            'memory': 'previous input feature of each residual block',
            'memory_init': 'zero',
            'memory_init_note': 'Local choice; official CausalMemBlock defaults to first-input replication',
            'normalization': 'native normalized H3 latents, no additional N/D',
            'residual_base': 'bilinear spatial2, align_corners=False',
            'output_head_initialization': 'zero', 'phase_resets': False,
            'decoder': False, 'official_recipe_reproduction': False,
        }

    @staticmethod
    def detach_state(state: Optional[CMNState]) -> Optional[CMNState]:
        return detach_state(state)

    def _prepare(self, x: Tensor, state: Optional[CMNState], latent_index: Optional[int]):
        if x.ndim != 4 or x.shape[1] != 24 or any(size <= 0 for size in x.shape):
            raise ValueError('Expected nonempty [B,24,H,W] latent frame')
        if not x.is_floating_point():
            raise ValueError('Expected floating-point native normalized latents')
        if latent_index is not None and (type(latent_index) is not int or latent_index < 0):
            raise ValueError('latent_index must be a nonnegative integer')
        if state is None:
            return (0 if latent_index is None else latent_index), (None,)*(3*self.blocks)
        if not isinstance(state, CMNState) or state.owner != id(self):
            raise ValueError('State belongs to another model; start with state=None')
        if state.input_shape != tuple(x.shape) or state.dtype != x.dtype or state.device != x.device:
            raise ValueError('Batch/geometry/dtype/device changed; start with state=None')
        if len(state.memories) != 3*self.blocks:
            raise ValueError('Wrong number of memory tensors')
        if latent_index is not None and latent_index != state.next_latent_index:
            raise ValueError('Noncontiguous latent_index; carry state or explicitly reset it')
        return state.next_latent_index, state.memories

    def _next(self, x: Tensor, memories, index: int):
        return CMNState(tuple(memories), index, tuple(x.shape), x.dtype, x.device, id(self))

    def forward_step(self, x: Tensor, state: Optional[CMNState] = None,
                     latent_index: Optional[int] = None):
        index, old = self._prepare(x, state, latent_index)
        feature = self.input(x)
        memories, position = [], 0
        for stage in self.stages:
            for block in stage.blocks:
                feature, memory = block.forward_step(feature, old[position])
                memories.append(memory)
                position += 1
            feature = stage.spatial(feature)
        base = F.interpolate(x, scale_factor=2., mode='bilinear', align_corners=False)
        output = base + self.output(feature)
        return output.unsqueeze(2), self._next(x, memories, index+1)

    def forward_chunk(self, x: Tensor, state: Optional[CMNState] = None,
                      latent_index: Optional[int] = None):
        if x.ndim != 5 or x.shape[2] <= 0:
            raise ValueError('Expected nonempty [B,24,T,H,W] latent chunk')
        b, _, t, _, _ = x.shape
        index, old = self._prepare(x[:, :, 0], state, latent_index)
        feature = _sequence(self.input(_frames(x)), b, t)
        memories, position = [], 0
        for stage in self.stages:
            for block in stage.blocks:
                feature, memory = block.forward_sequence(feature, old[position])
                memories.append(memory)
                position += 1
            feature = _sequence(stage.spatial(_frames(feature)), b, t)
        base = F.interpolate(_frames(x), scale_factor=2., mode='bilinear', align_corners=False)
        output = base + self.output(_frames(feature))
        return _sequence(output, b, t), self._next(x[:, :, 0], memories, index+t)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_chunk(x)[0]
