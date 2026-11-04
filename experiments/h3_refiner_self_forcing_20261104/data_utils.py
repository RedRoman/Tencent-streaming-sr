"""Official H3 pixel/latent conventions, with device selection for Ascend.

No CUDA fast-VAE kernels are used. Loading this module never initializes an NPU.
"""
from __future__ import annotations

import gc
import math
import os
import shutil
import subprocess
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


def select_device(name: str) -> torch.device:
    if name.startswith("npu"):
        import torch_npu  # noqa: F401; registers torch.npu
        torch.npu.set_device(name)
    return torch.device(name)


def empty_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "npu":
        torch.npu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def released_decoded_frames(tokens: int) -> int:
    """Mirror released H3 clip_length=17, ratio=4, token_drop=3 decoder.

    In particular 13 tokens decode to 43 frames; 37 tokens decode to 124.
    The implementation has no special image decode path, so require >=3 tokens.
    """
    if tokens < 3:
        raise ValueError("Video decoding requires at least three H3 latent frames")
    pad = (-(tokens + 3)) % 5
    chunks = (tokens + 3 + pad) // 5 - 1
    frames = 17 * chunks + 5
    frames -= sum(1 if (tokens + k) % 5 == 0 else 4 for k in range(pad))
    if frames <= 0:
        raise ValueError(f"Unsupported latent length {tokens}")
    return frames


def tensor_load(path: str | Path):
    # Only locally curated experiment assets are accepted by these CLIs.
    return torch.load(path, map_location="cpu", weights_only=False)


def video_tensor(value, *, name: str = "latent") -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim == 5 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 4 or value.shape[0] != 24:
        raise ValueError(f"{name} must be [24,T,H,W], got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains nonfinite values")
    return value.detach().cpu().contiguous()


def latent_stats(vae, device):
    mean = torch.as_tensor(vae.config.latents_mean, dtype=torch.float32, device=device).view(1, 24, 1, 1, 1)
    std = torch.as_tensor(vae.config.latents_std, dtype=torch.float32, device=device).view(1, 24, 1, 1, 1)
    if not bool((std > 0).all()):
        raise ValueError("Invalid VAE latent standard deviations")
    return mean, std


def pixel_stats(device):
    return (torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1, 1),
            torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1, 1))


def native_attention():
    try:
        from diffusers.models.attention_dispatch import attention_backend
    except ImportError:
        return nullcontext()
    return attention_backend("native")


def ffmpeg_executable():
    executable = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("FFmpeg is required; put ffmpeg on PATH or set FFMPEG_BIN") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


@lru_cache(maxsize=4)
def ffmpeg_video_codec(executable):
    result = subprocess.run([executable, "-hide_banner", "-encoders"], check=True,
                            capture_output=True, text=True)
    encoders = {line.split()[1] for line in result.stdout.splitlines() if len(line.split()) >= 2}
    if "libx264" in encoders:
        return "libx264"
    if "mpeg4" in encoders:
        print("FFmpeg lacks libx264; exporting MP4 with native MPEG-4 Part 2 at q=2", flush=True)
        return "mpeg4"
    raise RuntimeError("FFmpeg has neither libx264 nor native mpeg4 encoder")


def load_video_vae(base_model: str | Path, device: torch.device):
    from diffusers import AutoencoderKLMiniMaxH3
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        str(base_model), subfolder="vae", torch_dtype=torch.float32, local_files_only=True
    ).to(device).eval().requires_grad_(False)
    if (vae.config.clip_length, vae.config.token_drop) != (17, 3):
        raise ValueError("Unexpected H3 VAE temporal geometry")
    vae.enable_tiling(tile_sample_min_height=256, tile_sample_min_width=256,
                      tile_sample_min_overlap_height=64, tile_sample_min_overlap_width=64)
    return vae


def read_video(path, *, width: int, height: int, frames: int, fps: float = 24.,
               resize_filter: str = "area") -> torch.Tensor:
    """Read first frames, resample fps, cover-resize/crop, hold final frame if short."""
    vf = (f"fps={fps:g},scale={width}:{height}:force_original_aspect_ratio=increase:flags={resize_filter},"
          f"crop={width}:{height},tpad=stop_mode=clone:stop_duration={frames / fps + 1:g}")
    result = subprocess.run([
        ffmpeg_executable(), "-v", "error", "-i", str(path), "-vf", vf,
        "-frames:v", str(frames), "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ], check=True, capture_output=True)
    expected = frames * height * width * 3
    if len(result.stdout) != expected:
        raise ValueError(f"{path}: decoded {len(result.stdout)} bytes, expected {expected}")
    arr = np.frombuffer(result.stdout, dtype=np.uint8).reshape(frames, height, width, 3).copy()
    return torch.from_numpy(arr)


def encode_pixels(vae, frames: torch.Tensor, device: torch.device, seed: int) -> torch.Tensor:
    """ImageNet-normalized FP32 VAE; CPU posterior sampling matches cached HR."""
    pixels = frames.permute(3, 0, 1, 2)[None].to(device, torch.float32) / 255.
    pm, ps = pixel_stats(device)
    with torch.inference_mode(), native_attention():
        posterior = vae.encode((pixels - pm) / ps, return_dict=False)[0]
        # CPU noise also works on NPU and matches the released pipeline's CPU generator.
        noise = torch.randn(posterior.mean.shape, generator=torch.Generator(device="cpu").manual_seed(seed),
                            dtype=torch.float32).to(device)
        raw = posterior.mean.float() + posterior.std.float() * noise
        mean, std = latent_stats(vae, device)
        normalized = ((raw - mean) / std)[0].float().cpu()
    return video_tensor(normalized)


def encode_silence(base_model, *, frames: int, fps: float, device: torch.device):
    """Released AudioVAE posterior mode over zero waveform, then normalize."""
    from diffusers import AutoencoderKLMiniMaxH3Audio
    vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
        str(base_model), subfolder="audio_vae", torch_dtype=torch.float32,
        local_files_only=True).to(device).eval().requires_grad_(False)
    samples = round(frames / fps * 32000)
    with torch.inference_mode():
        posterior = vae.encode(torch.zeros(2, 1, samples, device=device), return_dict=False)[0]
        raw = posterior.mode().float()
        mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1)
        std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1)
        result = ((raw - mean) / std).to(torch.bfloat16).cpu().contiguous()
    if result.ndim != 3 or result.shape[:2] != (2, 32) or not torch.isfinite(result).all():
        raise ValueError(f"Invalid encoded silence shape {result.shape}")
    del vae, posterior, raw
    empty_cache(device)
    return result, {"kind": "official_encoded_silence", "sample_rate": 32000,
                    "waveform_samples": samples, "pixel_frames": frames,
                    "fps": fps, "latent_frames": result.shape[-1],
                    "posterior": "mode", "normalization": "(raw-mean)/std"}


def decode_latent(vae, normalized: torch.Tensor, device: torch.device,
                  keep_frames: int | None = None) -> torch.Tensor:
    latent = video_tensor(normalized)[None].to(device, torch.float32)
    mean, std = latent_stats(vae, device)
    pm, ps = pixel_stats(device)
    with torch.inference_mode(), native_attention():
        pixels = vae.decode(latent * std + mean, return_dict=False)[0].float()
        expected = released_decoded_frames(latent.shape[2])
        if pixels.shape[2] != expected:
            raise ValueError(f"Released VAE decoded {pixels.shape[2]} frames, expected {expected}")
        if keep_frames is not None:
            if not 0 < keep_frames <= expected:
                raise ValueError(f"Cannot keep {keep_frames} from {expected} decoded frames")
            pixels = pixels[:, :, :keep_frames]
        pixels = ((pixels * ps + pm).clamp(0, 1)[0].permute(1, 2, 3, 0) * 255).round().byte().cpu()
    return pixels


def write_mp4(frames: torch.Tensor, path: str | Path, *, fps: float = 24, crf: int = 16):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = frames.detach().cpu().contiguous().numpy()
    height, width = frames.shape[1:3]
    executable = ffmpeg_executable()
    codec = ffmpeg_video_codec(executable)
    encoding = ["-c:v", "libx264", "-preset", "fast", "-crf", str(crf)] if codec == "libx264" else ["-c:v", "mpeg4", "-q:v", "2"]
    subprocess.run([
        executable, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-an", *encoding,
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ], input=frames.tobytes(), check=True)
    return codec
