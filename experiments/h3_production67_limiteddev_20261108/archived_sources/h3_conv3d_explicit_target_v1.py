"""Candidate explicit-target wrapper for H3 [1,24,T,44,80] -> [1,24,T,68,120].

This changes call routing and state validation only. The supplied model is already
loaded/aligned. It never initializes or shifts weights, edits the frozen adapter,
normalizes a second time, casts precision, crops pixels, or trains by itself.
Historical adapter spatial_scale=2.0 is retained as historical provenance, NOT
as the effective target geometry. This candidate has not passed real345M/NPU,
activation-checkpoint, new-data, or decoder/gradient preflight.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import re
import threading

import torch
from torch import nn

from causal_upscaler import CausalUpscalerAdapter
from h3_production67_geometry_v1 import Geometry, production67

MODEL_KIND = 'h3_causal_conv3d_explicit_target_v1'
STATE_DICT_FORMAT = 'original_upscaler'


def validate_historical_adapter_config(config):
    expected = dict(initialization='aligned_past', temporal_padding='left_only_zero',
                    groupnorm='per_frame', spatial_scale=2.0, temporal_chunking=False,
                    dtype='float32', saved_weights_already_initialized=True,
                    reload_must_not_reinitialize_weights=True)
    if not isinstance(config, dict) or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('Require the historical already-aligned FP32 causal-adapter provenance')
    # bool compares equal to 1; require important booleans to remain actual bools.
    for key in ('temporal_chunking','saved_weights_already_initialized','reload_must_not_reinitialize_weights'):
        if type(config[key]) is not bool:
            raise ValueError('Historical adapter boolean fields must be actual bools')


def _positive_shape_tuple(value, count, name):
    if type(value) is not tuple or len(value) != count or any(type(x) is not int or x < 1 for x in value):
        raise ValueError(f'{name} must be an explicit positive integer tuple of length{count}')
    return value


@dataclass(frozen=True)
class ExplicitTargetState:
    """Opaque sequential token, not a portable cache or a branchable snapshot.

    No mutable cache tensors are exposed. A token is accepted only when it is the
    current token object owned by this wrapper at the current model version.
    """
    next_latent_index: int
    generation: int
    owner: int
    target_hw: tuple
    geometry_sha256: str


class ExplicitTargetCausalStudent(nn.Module):
    def __init__(self, node, model, historical_adapter_config, *, geometry,
                 source_checkpoint_sha256):
        super().__init__()
        if not isinstance(geometry, Geometry) or geometry != production67():
            raise ValueError('Require the reviewed explicit production67 Geometry')
        validate_historical_adapter_config(historical_adapter_config)
        if not isinstance(source_checkpoint_sha256,str) or not re.fullmatch('[0-9a-f]{64}',source_checkpoint_sha256):
            raise ValueError('Require a declared original checkpoint SHA256')
        if not isinstance(model,nn.Module) or getattr(model,'_aligned_past_initialized',None) is not True:
            raise ValueError('Caller must load and mark already-aligned weights before wrapping')
        params=tuple(model.parameters())
        if not params or any(p.dtype != torch.float32 for p in params):
            raise ValueError('This candidate preserves FP32; no automatic cast is supported')
        self._call_lock=threading.Lock()
        self._stream_generation=0
        self._live_state=None
        self._stream_versions=None
        self.geometry=geometry
        self.historical_adapter_config=copy.deepcopy(historical_adapter_config)
        self.source_checkpoint_sha256=source_checkpoint_sha256
        self.model=model
        self.adapter=CausalUpscalerAdapter(node,model)
        self.model_config={
            'kind':MODEL_KIND,'state_dict_format':STATE_DICT_FORMAT,
            'effective_geometry':geometry.description(),
            'geometry_sha256':geometry.sha256,
            'source_provenance':{'historical_adapter_config':copy.deepcopy(historical_adapter_config),
                                 'declared_checkpoint_sha256':source_checkpoint_sha256,
                                 'checkpoint_bytes_verified_by_this_constructor':False},
            'normalization':'unchanged CausalUpscalerAdapter.native_forward; no additional normalize',
            'formal_training_preflight_passed':False}
        self.model.eval().requires_grad_(True)
        self.train(False)

    def train(self, mode=True):
        # Preserve the historical deterministic/dropout-off recipe while still
        # permitting differentiable whole/prefix forwards and parameter grads.
        if type(mode) is not bool:
            raise ValueError('Training mode must be bool')
        super().train(mode)
        self.model.eval()
        return self

    def _acquire(self):
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError('Concurrent/reentrant wrapper calls are unsupported')

    def _check_native(self, native):
        if not isinstance(native,torch.Tensor) or native.ndim != 5:
            raise ValueError('Require native [1,24,T,44,80] tensor')
        if tuple(native.shape[:2]) != (1,24) or tuple(native.shape[-2:]) != (44,80):
            raise ValueError('Input must match explicit LR [1,24,T,44,80] geometry')
        if not 1 <= native.shape[2] <= 67 or native.dtype != torch.float32:
            raise ValueError('Require1..67 FP32 latent tokens; no cast/resize/truncation')
        params=tuple(self.model.parameters())
        if any(p.dtype != torch.float32 or p.device != native.device for p in params):
            raise ValueError('Native device/dtype must equal unchanged FP32 model parameters')
        return int(native.shape[2])

    def _target(self,target_size,frames):
        target_size=_positive_shape_tuple(target_size,3,'target_size')
        if target_size != (frames,68,120):
            raise ValueError('Target must explicitly be (current T,68,120)')
        return target_size

    def _version_signature(self):
        # Reject an optimizer/load/device mutation while a live stream cache
        # reflects earlier weights. _version is internal torch bookkeeping;
        # unsupported mutation through .data is outside this API contract.
        return tuple((id(v),v._version,v.data_ptr(),str(v.device),str(v.dtype))
                     for v in tuple(self.model.parameters())+tuple(self.model.buffers()))

    def _reset_unlocked(self):
        self.adapter.reset_stream()
        self._stream_generation+=1
        self._live_state=None
        self._stream_versions=None

    def reset_stream(self):
        self._acquire()
        try:self._reset_unlocked()
        finally:self._call_lock.release()

    def forward(self,native,*,target_size):
        """Differentiable full67 or prefix1..67 forward with explicit target."""
        self._acquire()
        try:
            frames=self._check_native(native)
            target_size=self._target(target_size,frames)
            # A differentiable/stateless forward does not consume or branch a
            # live inference cache. Require an explicit reset before mixing APIs.
            if self._live_state is not None:
                raise RuntimeError('Reset the live stream before a stateless full/prefix forward')
            output=self.adapter.native_forward(native,target_size)
            if tuple(output.shape)!=(1,24,frames,68,120):
                raise RuntimeError('Model returned a shape outside the explicit target contract')
            return output
        finally:self._call_lock.release()

    def forward_full(self,native,*,target_size):
        if not isinstance(native,torch.Tensor) or native.ndim!=5 or native.shape[2]!=67:
            raise ValueError('forward_full requires all67 latent tokens')
        return self.forward(native,target_size=target_size)

    def forward_prefix(self,native,*,target_size):
        return self.forward(native,target_size=target_size)

    @torch.no_grad()
    def forward_chunk(self,native,state=None,latent_index=0,*,target_hw):
        self._acquire()
        try:
            frames=self._check_native(native)
            target_hw=_positive_shape_tuple(target_hw,2,'target_hw')
            if target_hw!=(68,120):
                raise ValueError('Chunk target must explicitly be (68,120)')
            if type(latent_index) is not int or latent_index<0 or latent_index+frames>67:
                raise ValueError('Chunk exceeds total67 tokens or uses an invalid index')
            if state is None:
                if latent_index!=0:
                    raise ValueError('A new stream must start at index0')
                self._reset_unlocked()
                self._stream_versions=self._version_signature()
            elif (type(state) is not ExplicitTargetState or state is not self._live_state
                  or state.owner!=id(self) or state.generation!=self._stream_generation
                  or state.target_hw!=target_hw or state.geometry_sha256!=self.geometry.sha256
                  or state.next_latent_index!=latent_index
                  or self.adapter.stream_state()['processed_latent_frames']!=latent_index):
                raise ValueError('Stale, foreign, forged or out-of-order stream token')
            if self._stream_versions!=self._version_signature():
                self._reset_unlocked()
                raise ValueError('Model changed while streaming; explicit reset/restart is required')
            try:
                output=self.adapter.stream_step(native,target_hw)
                if tuple(output.shape)!=(1,24,frames,68,120):
                    raise RuntimeError('Streaming model returned wrong explicit target shape')
            except BaseException:
                self._reset_unlocked()
                raise
            new_state=ExplicitTargetState(latent_index+frames,self._stream_generation,id(self),target_hw,self.geometry.sha256)
            self._live_state=new_state
            return output,new_state
        finally:self._call_lock.release()

    def stream_step(self,native,state=None,latent_index=0,*,target_hw):
        """State-token inference API; returns (output,new_token), unlike old v5 API."""
        return self.forward_chunk(native,state,latent_index,target_hw=target_hw)

    def forward_step(self,native,state=None,latent_index=0,*,target_hw):
        if not isinstance(native,torch.Tensor) or native.ndim!=4:
            raise ValueError('Require one native frame [1,24,44,80]')
        return self.forward_chunk(native.unsqueeze(2),state,latent_index,target_hw=target_hw)

    def stream_state(self):
        self._acquire()
        try:
            info=dict(self.adapter.stream_state())
            info.update(generation=self._stream_generation,total_latent_limit=67,
                        effective_target_hw=[68,120],live_token=self._live_state is not None,
                        stream_snapshot_portable=False)
            return info
        finally:self._call_lock.release()

    def state_dict(self,*args,**kwargs):
        """Original upscaler names, without a wrapper model. prefix."""
        return self.model.state_dict(*args,**kwargs)

    def original_state_dict(self):
        return self.state_dict()

    def load_state_dict(self,state_dict,strict=True,assign=False):
        if strict is not True or assign is not False:
            raise ValueError('Only strict original-name loading preserving Parameter identity is supported')
        expected=self.model.state_dict()
        if not isinstance(state_dict,dict) or set(state_dict)!=set(expected):
            raise ValueError('Checkpoint must contain exactly the original upscaler tensor names')
        if any(not isinstance(state_dict[k],torch.Tensor) or state_dict[k].shape!=v.shape
               or state_dict[k].dtype!=v.dtype for k,v in expected.items()):
            raise ValueError('Checkpoint tensor shapes/dtypes must match without silent conversion')
        self._acquire()
        try:
            self._reset_unlocked()
            # No initialize_aligned_past_ call: source tensors are copied verbatim.
            return self.model.load_state_dict(state_dict,strict=True,assign=False)
        finally:self._call_lock.release()

    def load_original_state_dict(self,state):
        return self.load_state_dict(state)

    def pixel_probe_parameter(self):
        parameter=dict(self.model.named_parameters()).get('conv_out.weight')
        if parameter is None or not isinstance(getattr(self.model,'conv_out',None),nn.Conv3d):
            raise ValueError('Missing original Conv3d output head')
        return 'model.conv_out.weight',parameter
