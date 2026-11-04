"""Is the teacher broken, or am I calling it wrong?

Feed the SR-DiT a condition it must be able to handle: the TRUE Wan latent of
the HR frames (what a perfect mapper would produce). If the DiT still destroys
it, the call is wrong or the weights do not transfer; if it preserves it, the
mapper is the broken link and distillation is pointless either way.
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
PAIRS = ROOT / "data/wanpairs"; PIX = ROOT / "data/pixels"


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def main():
    from h3wan.npu_attention import install
    from h3wan.sr_distill import forward_sr
    from tiny_decoder import WanTinyDecoder
    mod = install(); dev = "npu:0"
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

    ids = sorted(f[:-4] for f in os.listdir(f"{PAIRS}/wan")
                 if f.endswith(".npy") and not f.endswith(".tmp.npy"))
    random.Random(1234).shuffle(ids)
    ctx = torch.zeros(1, 512, 4096, device=dev, dtype=torch.bfloat16)

    for vid in ids[:2]:
        w = torch.from_numpy(np.load(f"{PAIRS}/wan/{vid}.npy")).float()[:, :5].unsqueeze(0).to(dev)
        hrpx = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")
        wb = w.to(torch.bfloat16)
        with torch.no_grad():
            # the DiT is x0 = noise - flow; feeding the true latent as BOTH the
            # condition and the starting point is the easiest possible task
            o_self = forward_sr(dit, wb, wb, ctx, stream=True, offset=0)
            if isinstance(o_self, tuple):
                o_self = o_self[0]
            noise = torch.randn_like(wb)
            o_noise = forward_sr(dit, noise, wb, ctx, stream=True, offset=0)
            if isinstance(o_noise, tuple):
                o_noise = o_noise[0]
            px_true = tiny.decode_latents(w).float().clamp(-1, 1)
            px_self = tiny.decode_latents(o_self.float()).float().clamp(-1, 1)
            px_noise = tiny.decode_latents(o_noise.float()).float().clamp(-1, 1)
        n = px_true.shape[2]
        gt = torch.from_numpy(np.asarray(hrpx[:n])).permute(3, 0, 1, 2).unsqueeze(0).to(dev).float() / 127.5 - 1
        print(json.dumps(dict(clip=vid,
            latent_true_std=round(float(w.std()), 3),
            out_self_std=round(float(o_self.float().std()), 3),
            out_noise_std=round(float(o_noise.float().std()), 3),
            psnr_true_latent=round(psnr(px_true, gt), 2),
            psnr_dit_from_true=round(psnr(px_self, gt), 2),
            psnr_dit_from_noise=round(psnr(px_noise, gt), 2),
            latent_psnr_dit_vs_true=round(float(-10 * torch.log10(
                F.mse_loss(o_self.float(), w) + 1e-12)), 2))), flush=True)
        torch.npu.empty_cache()
    os._exit(0)


main()
