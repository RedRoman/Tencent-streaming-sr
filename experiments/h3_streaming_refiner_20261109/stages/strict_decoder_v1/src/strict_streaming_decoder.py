"""Experimental one-latent H3 decoder; full pretrained/NPU validation is pending.

CPU-backed per-layer KV retains a bounded seven-slot window. Two shared latents
are reprojected when that window shifts, plus one initial refresh. This bounded
refresh is reported explicitly; it is not full-prefix recomputation.
"""
from __future__ import annotations

import torch
from strict_decoder_core import IncrementalVAECore


class StrictStreamingVideoDecoder:
    def __init__(self, vae, *, token_budget=2600):
        if any(p.dtype != torch.float32 for p in vae.parameters()):
            raise ValueError("Preserve the frozen FP32 VAE; mixed precision is not admitted")
        if (vae.tokens_chunk_size, vae.token_overlap, vae.temporal_compression_ratio,
                vae.frame_pre_padding, vae.spatial_compression_ratio) != (5, 2, 4, 3, 16):
            raise ValueError("Unexpected H3 temporal/spatial geometry")
        if vae.training or any(p.requires_grad for p in vae.parameters()):
            raise ValueError("VAE must be frozen and in evaluation mode")
        self.vae = vae
        self.device = vae.decoder.proj_in.weight.device
        self.core = IncrementalVAECore(vae, compile=False, compensate=False,
                                      token_budget=token_budget, cache_device='cpu')
        self.mean = torch.tensor(vae.config.latents_mean, device=self.device).view(1,24,1,1,1)
        self.std = torch.tensor(vae.config.latents_std, device=self.device).view(1,24,1,1,1)
        if not bool(torch.isfinite(self.mean).all() and torch.isfinite(self.std).all()
                    and (self.std > 0).all()):
            raise ValueError("Invalid VAE latent statistics")
        self.pixel_mean = torch.tensor([.485,.456,.406], device=self.device).view(1,3,1,1,1)
        self.pixel_std = torch.tensor([.229,.224,.225], device=self.device).view(1,3,1,1,1)
        self.reset()

    def reset(self):
        self.core.reset()
        self.pixel_frames = 0
        self.poisoned = False

    @torch.inference_mode()
    def push(self, normalized, *, latent_index):
        if self.poisoned:
            raise RuntimeError("Previous decode failed; reset before reuse")
        if latent_index != self.core.received:
            raise ValueError("Absolute latent index must be the next unconsumed index")
        if tuple(normalized.shape) != (1,24,1,68,120):
            raise ValueError("Expected one full-spatial latent [1,24,1,68,120]")
        if not normalized.is_floating_point() or not bool(torch.isfinite(normalized).all()):
            raise ValueError("Latent must be finite and floating point")
        try:
            raw = normalized.to(self.device, torch.float32) * self.std + self.mean
            pixels = self.core.push(raw)
            count = self.core.frames_of(latent_index)
            if tuple(pixels.shape) != (1,3,count,1088,1920):
                raise ValueError("Decoder output geometry differs from the frozen contract")
            if not bool(torch.isfinite(pixels).all()):
                raise ValueError("Decoder produced nonfinite pixels")
            rgb = ((pixels.float() * self.pixel_std + self.pixel_mean).clamp(0,1) * 255).round()
            rgb = rgb[0,:, :,4:1084,:].permute(1,2,3,0).to('cpu',torch.uint8).contiguous()
            assert self.pixel_frames == self.core.first_frame(latent_index)
            self.pixel_frames += count
            return rgb
        except BaseException:
            self.poisoned = True
            raise

    def cache_stats(self):
        arrays = (self.core._k or []) + (self.core._v or [])
        return {
            'latent_count': self.core.received,
            'pixel_frames': self.pixel_frames,
            'kv_bytes': sum(x.numel()*x.element_size() for x in arrays),
            'kv_device': 'cpu',
            'retained_input_latents': len(self.core._latents),
            'slot_forward_count': self.core.slot_passes,
            'refreshed_past_slots': self.core.slot_passes-self.core.received,
            'rope_pattern_count': len(self.core._rope_cache),
            'compensation': False,
            'full_prefix_recompute': False,
        }
