"""Fixed-shape causal refiner execution with inaccessible zero future slots.

This controls sequence-length-dependent execution differences. It is an offline
fixed-window prototype, not a KV cache or an unbounded streaming implementation.
"""
from __future__ import annotations

import torch


class FixedHorizonRefiner:
    def __init__(self, runtime, horizon):
        if horizon < 1:
            raise ValueError('A positive latent-frame horizon is required')
        self.runtime = runtime
        self.horizon = horizon

    def forward(self, value, sample, sigma, *, causal=True, block_index=0, **kwargs):
        if not causal or value.ndim != 4:
            raise ValueError('Expected a causal [C,T,H,W] prefix')
        frames = value.shape[1]
        expected = min((block_index + 1) * self.runtime.block_frames, self.horizon)
        if frames != expected or not 0 < frames <= self.horizon:
            raise ValueError(f'Prefix must end at the current block boundary: {frames} != {expected}')
        padded = value.new_zeros(value.shape[0], self.horizon, *value.shape[2:])
        padded[:, :frames] = value
        # Runtime's attention mask excludes later blocks for both video and
        # static queries. No external future content enters this tensor.
        velocity = self.runtime.forward(padded, sample, sigma, causal=True,
                                        block_index=block_index, **kwargs)
        return velocity[:, :frames]
