"""Re-test the SR-DiT with real text conditioning.

The earlier run passed an all-zero 512-row context. That is not "no prompt" --
it is 512 valid tokens that happen to be zero, which cross-attention has to
attend to. The repro encodes the real prompt with UMT5 and keeps only the valid
tokens before padding. If the DiT behaves once given that, the weights are fine
and the earlier verdict was my calling error.
"""
import json, os, random, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
sys.path.insert(0, str(ROOT / "h2w/vendor/UltraFlash/inference/sr"))
SRW = "/temp/rongxiang_workspace_backup_20260921/rongxiang/UltraFlash/inference/checkpoints/2K-Causal-Sparse-SR-Dit-fidelity-step_12800.pth"
TINY = "/temp/rongxiang_workspace_backup_20260921/rongxiang/UltraFlash/inference/checkpoints/tiny_decoder.pth"
WANDIR = Path("/temp/Wan2.1-T2V-1.3B")
PAIRS = ROOT / "data/wanpairs"; PIX = ROOT / "data/pixels"
MANIFEST = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/high/source_manifest.jsonl"


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def main():
    from h3wan.npu_attention import install
    from h3wan.sr_distill import forward_sr
    from tiny_decoder import WanTinyDecoder
    from vendor.wan_text_npu.t5 import umt5_xxl
    from vendor.wan_text_npu.tokenizers import HuggingfaceTokenizer
    mod = install(); dev = "npu:0"

    prompts = {}
    for line in open(MANIFEST):
        d = json.loads(line); prompts[d["id"]] = d["prompt"]

    t5 = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=torch.bfloat16, device="meta")
    st = torch.load(str(WANDIR / "models_t5_umt5-xxl-enc-bf16.pth"), map_location="cpu",
                    mmap=True, weights_only=True)
    t5.load_state_dict(st, strict=True, assign=True); del st
    t5 = t5.eval().requires_grad_(False).to(device=dev, dtype=torch.bfloat16)
    tok = HuggingfaceTokenizer(str(WANDIR / "google/umt5-xxl"), seq_len=512, clean="whitespace")
    print("T5 loaded", flush=True)

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
    zero_ctx = torch.zeros(1, 512, 4096, device=dev, dtype=torch.bfloat16)

    for vid in ids[:3]:
        p = prompts.get(vid, "")
        tid, mask = tok([p], return_mask=True, add_special_tokens=True)
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            ctx_full = t5(tid.to(dev), mask.to(dev))
        ctx = ctx_full[0, :int(mask.sum())].unsqueeze(0).to(torch.bfloat16)

        w = torch.from_numpy(np.load(f"{PAIRS}/wan/{vid}.npy")).float()[:, :5].unsqueeze(0).to(dev)
        wb = w.to(torch.bfloat16)
        hrpx = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")

        out = {}
        for tag, c in (("real_text", ctx), ("zero_text", zero_ctx)):
            with torch.no_grad():
                o = forward_sr(dit, wb, wb, c, stream=True, offset=0)
                if isinstance(o, tuple):
                    o = o[0]
            out[tag] = o.float()
        px_true = tiny.decode_latents(w).float().clamp(-1, 1)
        n = px_true.shape[2]
        gt = torch.from_numpy(np.asarray(hrpx[:n])).permute(3, 0, 1, 2).unsqueeze(0).to(dev).float() / 127.5 - 1

        row = dict(clip=vid, text_tokens=int(mask.sum()),
                   psnr_true_latent=round(psnr(px_true, gt), 2))
        for tag, o in out.items():
            px = tiny.decode_latents(o).float().clamp(-1, 1)
            row[f"psnr_{tag}"] = round(psnr(px, gt), 2)
            row[f"latent_match_{tag}"] = round(
                float(-10 * torch.log10(F.mse_loss(o, w) + 1e-12)), 2)
        print(json.dumps(row), flush=True)
        torch.npu.empty_cache()
    os._exit(0)


main()
