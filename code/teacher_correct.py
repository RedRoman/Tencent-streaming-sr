"""Score the SR-DiT the way it is actually meant to be driven.

My earlier probes put the clean latent in the `noise` slot, which the model has
never seen -- it expects randn there and the conditioning latent separately.
Correct call: noise ~ N(0,1), condition = the (upsampled) Wan latent.

Two conditions are tried: the TRUE Wan HR latent (what a perfect mapper gives)
and the mapper's actual output, which separates teacher quality from mapper
quality.
"""
import importlib.util, json, os, random, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
sys.path.insert(0, str(ROOT / "h2w/vendor/UltraFlash/inference/sr"))
SRW = "/temp/rongxiang_workspace_backup_20260921/rongxiang/UltraFlash/inference/checkpoints/2K-Causal-Sparse-SR-Dit-fidelity-step_12800.pth"
TINY = "/temp/rongxiang_workspace_backup_20260921/rongxiang/UltraFlash/inference/checkpoints/tiny_decoder.pth"
MAP = "/temp/liusonghua/h3_to_wan_repro_32193/joint_mapper_upsampler_wan_dit_step_002000.pt"
WANDIR = Path("/temp/Wan2.1-T2V-1.3B")
PAIRS = ROOT / "data/wanpairs"; PIX = ROOT / "data/pixels"
MANIFEST = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/high/source_manifest.jsonl"


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def main():
    from h3wan.npu_attention import install
    from h3wan.sr_distill import forward_sr
    from h3wan.model import Mapper
    from tiny_decoder import WanTinyDecoder
    from vendor.wan_text_npu.t5 import umt5_xxl
    from vendor.wan_text_npu.tokenizers import HuggingfaceTokenizer
    mod = install(); dev = "npu:0"

    prompts = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(MANIFEST)}
    t5 = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=torch.bfloat16, device="meta")
    st = torch.load(str(WANDIR / "models_t5_umt5-xxl-enc-bf16.pth"), map_location="cpu",
                    mmap=True, weights_only=True)
    t5.load_state_dict(st, strict=True, assign=True); del st
    t5 = t5.eval().requires_grad_(False).to(device=dev, dtype=torch.bfloat16)
    tok = HuggingfaceTokenizer(str(WANDIR / "google/umt5-xxl"), seq_len=512, clean="whitespace")

    sd = torch.load(MAP, map_location="cpu", mmap=True, weights_only=False)["model"]
    mapper = Mapper(); mapper.load_state_dict({k[13:]: v for k, v in sd.items()
        if k.startswith("joint.mapper.")}, strict=True); mapper = mapper.to(dev).eval()
    spec = importlib.util.spec_from_file_location("uf",
        ROOT / "h2w/vendor/UltraFlash/inference/sr/latent_upsampler/ultra_latent_up_v1.py")
    um = importlib.util.module_from_spec(spec); spec.loader.exec_module(um)
    up = um.UltraLatentUpsampler(in_channels=16, out_channels=16, mid_channels=128,
        num_blocks=8, upsample_scale=3, activation="relu", residual=False,
        memory_init="replicate", zero_init_final=False, default_parallel=True)
    up.load_state_dict({k[16:]: v for k, v in sd.items()
        if k.startswith("joint.upsampler.")}, strict=True); up = up.to(dev).eval(); del sd

    with torch.device("meta"):
        dit = mod.Transformer3DModel(model_type="t2v", patch_size=(1, 2, 2), text_len=512,
            in_dim=32, dim=1536, ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16,
            num_heads=12, num_layers=30, window_size=(-1, -1), qk_norm=True,
            cross_attn_norm=True, eps=1e-6, use_sparse_attn=True, sparse_causal=True,
            sparse_block_size=(2, 8, 8), sparse_top_k_ratio=1., sparse_kv_ratio=3.,
            sparse_local_range=9, sparse_use_kernel=False, stream_chunk_size=2,
            sparse_ref_spatial_tokens=1560, rope_max_seq_len=1024, rope_theta=10000.,
            rope_cache_multiple=1024)
    st = torch.load(SRW, map_location="cpu", mmap=True, weights_only=False)["model"]
    dit.load_state_dict(st, strict=True, assign=True); del st
    dit = dit.to(dev).to(torch.bfloat16)
    dit.time_embedding.float(); dit.time_projection.float(); dit.head.float()
    dit.eval().requires_grad_(False)
    tiny = WanTinyDecoder(TINY, dtype=torch.float32).to(dev).eval()
    print("loaded", flush=True)

    ids = sorted(f[:-4] for f in os.listdir(f"{PAIRS}/wan")
                 if f.endswith(".npy") and not f.endswith(".tmp.npy"))
    random.Random(1234).shuffle(ids)
    rows = []
    for vid in ids[:4]:
        tid, mask = tok([prompts.get(vid, "")], return_mask=True, add_special_tokens=True)
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            cf = t5(tid.to(dev), mask.to(dev))
        ctx = cf[0, :int(mask.sum())].unsqueeze(0).to(torch.bfloat16)

        w = torch.from_numpy(np.load(f"{PAIRS}/wan/{vid}.npy")).float()[:, :5].unsqueeze(0).to(dev)
        h3 = torch.from_numpy(np.load(f"{PAIRS}/h3/{vid}.npy")).float()[:, :5].unsqueeze(0).to(dev)
        hrpx = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")

        g = torch.Generator(device="cpu").manual_seed(0)
        noise = torch.randn(w.shape, generator=g).to(dev, torch.bfloat16)
        with torch.no_grad():
            mapped = up(mapper(h3), parallel=True).to(torch.bfloat16)
            o_true = forward_sr(dit, noise, w.to(torch.bfloat16), ctx, stream=False)
            o_map = forward_sr(dit, noise, mapped, ctx, stream=False)
            if isinstance(o_true, tuple): o_true = o_true[0]
            if isinstance(o_map, tuple): o_map = o_map[0]
            px_t = tiny.decode_latents(w).float().clamp(-1, 1)
            px_dt = tiny.decode_latents(o_true.float()).float().clamp(-1, 1)
            px_dm = tiny.decode_latents(o_map.float()).float().clamp(-1, 1)
            px_m = tiny.decode_latents(mapped.float()).float().clamp(-1, 1)
        n = px_t.shape[2]
        gt = torch.from_numpy(np.asarray(hrpx[:n])).permute(3, 0, 1, 2).unsqueeze(0).to(dev).float() / 127.5 - 1
        rows.append(dict(clip=vid,
            true_latent=round(psnr(px_t, gt), 2),
            dit_from_true_cond=round(psnr(px_dt, gt), 2),
            mapper_cond_only=round(psnr(px_m, gt), 2),
            dit_from_mapper_cond=round(psnr(px_dm, gt), 2)))
        print(json.dumps(rows[-1]), flush=True)
        torch.npu.empty_cache()
    a = lambda k: round(sum(r[k] for r in rows) / len(rows), 2)
    print("\nMEAN " + json.dumps({k: a(k) for k in rows[0] if k != "clip"}), flush=True)
    os._exit(0)


main()
