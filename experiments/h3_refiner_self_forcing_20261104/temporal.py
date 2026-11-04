"""H3 block-causal attention and a one-step, prefix-recomputed Self Forcing rollout.

H3 uses data-ward velocity and model time t = 1 - sigma. Clean history therefore
has t=1, unlike the zero clean timestep used by some other video models.
This module imports neither Diffusers nor distributed/CUDA-specific libraries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F


STATIC_BLOCK = -1
PADDING_BLOCK = -2


@dataclass(frozen=True)
class TemporalPlan:
    timesteps: torch.Tensor
    timestep_indices: torch.Tensor
    row_blocks: torch.Tensor
    valid_rows: int
    current_block: int
    rows_per_frame: int
    frame_count: int
    block_frames: int

    def to(self, device: torch.device | str) -> "TemporalPlan":
        return TemporalPlan(
            self.timesteps.to(device), self.timestep_indices.to(device),
            self.row_blocks.to(device), self.valid_rows, self.current_block,
            self.rows_per_frame, self.frame_count, self.block_frames,
        )


def build_temporal_plan(
    video_latent: torch.Tensor,
    layout: Any,
    *,
    block_frames: int,
    current_block: int,
    sigma: float,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    pad_to: int = 1,
) -> TemporalPlan:
    """Build row times and block IDs for a packed ``[text|audio|video]`` prefix.

    ``video_latent`` is [C,F,H,W]. ``layout`` supplies ``video_indices``,
    ``audio_indices``, ``text_indices`` and ``token_tags`` as upstream geometry
    does. Video rows must have upstream's frame-major patch order. The prefix may
    contain future blocks for leakage tests; they receive the noisy time too,
    but the attention processor prevents earlier blocks from reading them.
    Padding is absent from all three index lists and receives no attention.
    """
    if video_latent.ndim != 4:
        raise ValueError("video_latent must have shape [C,F,H,W]")
    _, frames, height, width = video_latent.shape
    pt, ph, pw = patch_size
    if min(frames, height, width, pt, ph, pw, block_frames, pad_to) <= 0:
        raise ValueError("geometry, block_frames and pad_to must be positive")
    if frames % pt or height % ph or width % pw or block_frames % pt:
        raise ValueError("latent and block dimensions must be divisible by patch_size")
    if not 0 <= current_block < (frames + block_frames - 1) // block_frames:
        raise ValueError("current_block is outside the video prefix")
    if not 0.0 <= float(sigma) <= 1.0:
        raise ValueError("sigma must be in [0,1]")

    device = layout.token_tags.device
    video_ids = torch.as_tensor(layout.video_indices, device=device, dtype=torch.long).flatten()
    audio_ids = torch.as_tensor(layout.audio_indices, device=device, dtype=torch.long).flatten()
    text_ids = torch.as_tensor(layout.text_indices, device=device, dtype=torch.long).flatten()
    rows_per_frame = (height // ph) * (width // pw)
    expected_video_rows = (frames // pt) * rows_per_frame
    if video_ids.numel() != expected_video_rows:
        raise ValueError(f"layout has {video_ids.numel()} video rows; expected {expected_video_rows}")
    all_ids = torch.cat((video_ids, audio_ids, text_ids))
    if not all_ids.numel() or bool((all_ids < 0).any()):
        raise ValueError("packed modality indices must be nonempty and nonnegative")
    valid_rows = int(all_ids.max().item()) + 1
    if all_ids.numel() != valid_rows or torch.unique(all_ids).numel() != valid_rows:
        raise ValueError("modality indices must partition one contiguous valid prefix")
    layout_rows = int(layout.token_tags.numel())
    if layout_rows < valid_rows:
        raise ValueError("token_tags is shorter than the packed layout")
    total_rows = ((max(layout_rows, valid_rows) + pad_to - 1) // pad_to) * pad_to
    row_blocks = torch.full((total_rows,), PADDING_BLOCK, dtype=torch.long, device=device)
    row_blocks[text_ids] = STATIC_BLOCK
    row_blocks[audio_ids] = STATIC_BLOCK
    video_frames = torch.arange(frames // pt, device=device).repeat_interleave(rows_per_frame) * pt
    row_blocks[video_ids] = video_frames // block_frames
    row_times = torch.ones(total_rows, dtype=torch.float32, device=device)
    row_times[row_blocks >= current_block] = 1.0 - float(sigma)
    times, inverse = torch.unique(row_times, sorted=True, return_inverse=True)
    return TemporalPlan(times, inverse, row_blocks, valid_rows, current_block,
                        rows_per_frame, frames, block_frames)


def _apply_h3_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Upstream H3 RoPE: rotate only leading channels in [B,N,H,D]."""
    dim = cos.shape[-1]
    if dim % 2 or cos.shape != sin.shape or cos.shape[0] != x.shape[1] or dim > x.shape[-1]:
        raise ValueError("invalid H3 rotary geometry")
    rotating, rest = x[..., :dim], x[..., dim:]
    first, second = rotating.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    cos = cos.to(device=x.device, dtype=x.dtype)[None, :, None, :]
    sin = sin.to(device=x.device, dtype=x.dtype)[None, :, None, :]
    return torch.cat((rotating * cos + rotated * sin, rest), dim=-1).contiguous()


class H3BlockCausalProcessor:
    """Drop-in H3 attention processor with no future path through text/audio.

    Uses ordinary torch SDPA on explicit allowed K/V subsets, avoiding a full
    quadratic causal mask. ``mode='dense'`` is the unmasked valid-row reference.
    With ``detach_history``, *current-block queries* read detached older video
    K/V; static conditioning remains differentiable. Padding output is zero.

    This is full-prefix attention, not sequence-parallel or cached attention.
    Restore the generator plan and mode before checkpointed backward. When
    retaining graphs for multiple prefixes, the callback must capture/reinstall
    each prefix's plan during checkpoint replay. The rollout's default last-only
    gradient avoids that multi-prefix lifetime issue.
    """
    _attention_backend = None
    _parallel_config = None

    def __init__(self, plan: TemporalPlan | None = None, *, mode: str = "causal",
                 detach_history: bool = True, query_chunk_size: int = 256):
        if mode not in ("causal", "dense"):
            raise ValueError("mode must be causal or dense")
        if query_chunk_size < 1:
            raise ValueError("query_chunk_size must be positive")
        self.plan = plan
        self.mode = mode
        self.detach_history = detach_history
        self.query_chunk_size = query_chunk_size
        self.valid_rows = None if plan is None else plan.valid_rows

    def set_plan(self, plan: TemporalPlan) -> None:
        self.plan, self.valid_rows = plan, plan.valid_rows

    @staticmethod
    def _select_mask(mask: torch.Tensor | None, query_ids: torch.Tensor,
                     key_ids: torch.Tensor) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.ndim < 2:
            raise ValueError("attention_mask needs query/key dimensions")
        if mask.shape[-2] != 1:
            mask = mask.index_select(-2, query_ids)
        if mask.shape[-1] != 1:
            mask = mask.index_select(-1, key_ids)
        return mask

    def __call__(self, attn: Any, hidden_states: torch.Tensor,
                 rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
                 attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        plan = self.plan
        if plan is None:
            raise RuntimeError("set a TemporalPlan before attention")
        if hidden_states.ndim != 3 or hidden_states.shape[1] != plan.row_blocks.numel():
            raise ValueError("hidden_states sequence must match the full packed plan (including padding)")
        blocks = plan.row_blocks.to(hidden_states.device)
        if getattr(attn, "fused_projections", False):
            q, k, v = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            q, k, v = (fn(hidden_states) for fn in (attn.to_q, attn.to_k, attn.to_v))
        q, k, v = (x.unflatten(-1, (attn.heads, -1)) for x in (q, k, v))
        q, k = attn.norm_q(q), attn.norm_k(k)
        if rotary_emb is not None:
            q, k = (_apply_h3_rotary(x, *rotary_emb) for x in (q, k))
        chunk_outputs, chunk_indices = [], []
        valid = blocks != PADDING_BLOCK
        # These small block identifiers are structural metadata, not model data.
        query_groups = [int(b) for b in torch.unique(blocks[valid]).detach().cpu().tolist()]
        for block in query_groups:
            query_ids = torch.nonzero(blocks == block, as_tuple=False).flatten()
            allowed = valid if self.mode == "dense" else (
                (blocks == STATIC_BLOCK) if block == STATIC_BLOCK else
                ((blocks == STATIC_BLOCK) | ((blocks >= 0) & (blocks <= block))))
            key_ids = torch.nonzero(allowed, as_tuple=False).flatten()
            if not key_ids.numel():
                raise ValueError("an attention query has no allowed key")
            kg, vg = k.index_select(1, key_ids), v.index_select(1, key_ids)
            if self.detach_history and block == plan.current_block:
                history = ((blocks[key_ids] >= 0) & (blocks[key_ids] < block))[None, :, None, None]
                kg = torch.where(history, kg.detach(), kg)
                vg = torch.where(history, vg.detach(), vg)
            kg, vg = kg.transpose(1, 2).contiguous(), vg.transpose(1, 2).contiguous()
            for ids in query_ids.split(self.query_chunk_size):
                qg = q.index_select(1, ids).transpose(1, 2).contiguous()
                mask = self._select_mask(attention_mask, ids, key_ids)
                result = F.scaled_dot_product_attention(qg, kg, vg, attn_mask=mask,
                                                        dropout_p=0.0, is_causal=False)
                chunk_outputs.append(result.transpose(1, 2).to(q.dtype))
                chunk_indices.append(ids)
        # One scatter avoids copying a full sequence once per small query chunk.
        out = torch.zeros_like(q).index_copy(1, torch.cat(chunk_indices), torch.cat(chunk_outputs, dim=1))
        out = attn.to_out[1](attn.to_out[0](out.flatten(2, 3)))
        # Output projections may have bias, so zero padded rows after projection.
        return out.masked_fill((~valid)[None, :, None], 0)


@dataclass(frozen=True)
class RolloutResult:
    latent: torch.Tensor
    audit: dict[str, Any]


def self_forcing_rollout(
    upscaled: torch.Tensor,
    sigma: float,
    block_frames: int,
    forward_velocity: Callable[..., torch.Tensor],
    *,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    grad_blocks: str | Iterable[int] = "last",
) -> RolloutResult:
    """One refiner evaluation per block with the student's own detached history.

    ``forward_velocity(prefix, block_index=..., current_start=...,
    current_end=..., sigma=...)`` must apply the causal plan and return velocity
    with the prefix shape, or just the current-block shape. A tuple's first item
    is also accepted. ``prefix`` is [C,F,H,W] FP32: previous *generated* blocks
    followed by the current block's noisy upscaled input. No future is supplied.
    ``grad_blocks`` is 'last' (default), 'all', 'none', or block indices. Earlier
    generated results are detached when used as context, even in 'all' mode.
    The caller owns teacher/fake scoring, DMD loss, and optimizer updates.
    """
    if upscaled.ndim != 4 or min(upscaled.shape) <= 0 or block_frames < 1:
        raise ValueError("upscaled must be nonempty [C,F,H,W], block_frames positive")
    if not 0.0 < float(sigma) <= 1.0:
        raise ValueError("rollout sigma must be in (0,1]")
    source = upscaled.float()
    if noise is None:
        noise = torch.randn(source.shape, device=source.device, dtype=torch.float32, generator=generator)
    elif noise.shape != source.shape or noise.device != source.device:
        raise ValueError("noise must match upscaled shape and device")
    noise = noise.float()
    count = (source.shape[1] + block_frames - 1) // block_frames
    if isinstance(grad_blocks, str):
        choices = {"last": {count - 1}, "all": set(range(count)), "none": set()}
        if grad_blocks not in choices:
            raise ValueError("grad_blocks must be last, all, none, or block indices")
        retained = choices[grad_blocks]
    else:
        retained = set(grad_blocks)
        if any(not isinstance(i, int) or i < 0 or i >= count for i in retained):
            raise ValueError("grad_blocks contains an invalid block index")
    outer_grad = torch.is_grad_enabled()
    outputs, history, rows = [], [], []
    for index, start in enumerate(range(0, source.shape[1], block_frames)):
        end = min(start + block_frames, source.shape[1])
        with torch.set_grad_enabled(outer_grad and index in retained):
            state = (1.0 - float(sigma)) * source[:, start:end] + float(sigma) * noise[:, start:end]
            prefix = torch.cat([*history, state], dim=1) if history else state
            velocity = forward_velocity(prefix, block_index=index, current_start=start,
                                        current_end=end, sigma=float(sigma))
            if isinstance(velocity, tuple):
                velocity = velocity[0]
            if not torch.is_tensor(velocity):
                raise TypeError("forward_velocity must return a tensor or tensor-first tuple")
            if velocity.shape == prefix.shape:
                velocity = velocity[:, start:end]
            elif velocity.shape != state.shape:
                raise ValueError("velocity shape must match prefix or current block")
            prediction = state + float(sigma) * velocity.float()
        outputs.append(prediction)
        history.append(prediction.detach())
        rows.append({"block_index": index, "current_start": start, "current_end": end,
                     "history_frames": start, "prefix_frames": end,
                     "gradient_enabled": bool(outer_grad and index in retained),
                     "history_detached": all(not h.requires_grad for h in history[:-1])})
    return RolloutResult(torch.cat(outputs, dim=1), {
        "method": "one_step_block_self_forcing_prefix_recompute",
        "blocks": count, "model_forwards": count, "steps_per_block": 1,
        "sigma": float(sigma), "clean_history_model_time": 1.0,
        "current_model_time": 1.0 - float(sigma),
        "gradient_blocks": [r["block_index"] for r in rows if r["gradient_enabled"]],
        "self_generated_history": True, "future_frames_exposed": 0,
        "history_detached": all(r["history_detached"] for r in rows),
        "block_records": rows,
    })
