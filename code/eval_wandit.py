"""Evaluate the route with the CORRECT DiT: the fine-tuned WAN DiT from the
checkpoint, not UltraFlash's SR-DiT.

The checkpoint's `sr.dit.*` has all 825 WAN DiT keys, 824 identical in shape,
differing only at patch_embedding (32 channels instead of 16, to take noise and
condition concatenated) and by 12.5% in weights. That is a fine-tuned WAN DiT.
Everything I measured before used UltraFlash's unrelated SR-DiT weights.
"""
import importlib.util, json, math, os, random, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
sys.path.insert(0, str(ROOT / "h2w/vendor/UltraFlash/inference/sr"))
CK = "/temp/liusonghua/h3_to_wan_repro_32193/joint_mapper_upsampler_wan_dit_step_002000.pt"
TINY = "/temp/rongxiang_workspace_backup_20260921/rongxiang/UltraFlash/inference/checkpoints/tiny_decoder.pth"
WANDIR = Path("/temp/Wan2.1-T2V-1.3B")
PAIRS = ROOT / "data/wanpairs"; PIX = ROOT / "data/pixels"
MANIFEST = "/temp/yunfeng/h3_hrx0kd_transfer_20261008/high/source_manifest.jsonl"


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def main():
    from h3wan.npu_attention import install
    from h3wan.model import Mapper
    from tiny_decoder import WanTinyDecoder
    from vendor.wan_text_npu.t5 import umt5_xxl
    from vendor.wan_text_npu.tokenizers import HuggingfaceTokenizer
    mod = install(); dev = "npu:0"

    ck = torch.load(CK, map_location="cpu", mmap=True, weights_only=False)["model"]
    mapper = Mapper(); mapper.load_state_dict({k[13:]: v for k, v in ck.items()
        if k.startswith("joint.mapper.")}, strict=True); mapper = mapper.to(dev).eval()
    spec = importlib.util.spec_from_file_location("uf",
        ROOT / "h2w/vendor/UltraFlash/inference/sr/latent_upsampler/ultra_latent_up_v1.py")
    um = importlib.util.module_from_spec(spec); spec.loader.exec_module(um)
    up = um.UltraLatentUpsampler(in_channels=16, out_channels=16, mid_channels=128,
        num_blocks=8, upsample_scale=3, activation="relu", residual=False,
        memory_init="replicate", zero_init_final=False, default_parallel=True)
    up.load_state_dict({k[16:]: v for k, v in ck.items()
        if k.startswith("joint.upsampler.")}, strict=True); up = up.to(dev).eval()

    # WAN DiT geometry, with the checkpoint's 32-channel patch embedding
    with torch.device("meta"):
        dit = mod.Transformer3DModel(model_type="t2v", patch_size=(1, 2, 2), text_len=512,
            in_dim=32, dim=1536, ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16,
            num_heads=12, num_layers=30, window_size=(-1, -1), qk_norm=True,
            cross_attn_norm=True, eps=1e-6, use_sparse_attn=False,
            rope_max_seq_len=1024, rope_theta=10000., rope_cache_multiple=1024)
    sd = {k[len("sr.dit."):]: v for k, v in ck.items() if k.startswith("sr.dit.")}
    missing, unexpected = dit.load_state_dict(sd, strict=False, assign=True)
    print(json.dumps(dict(dit_missing=len(missing), dit_unexpected=len(unexpected),
                          sample_missing=[str(m) for m in list(missing)[:4]])), flush=True)
    dit = dit.to(dev).to(torch.bfloat16)
    dit.time_embedding.float(); dit.time_projection.float(); dit.head.float()
    dit.eval().requires_grad_(False)
    del ck, sd
    tiny = WanTinyDecoder(TINY, dtype=torch.float32).to(dev).eval()

    prompts = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(MANIFEST)}
    t5 = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=torch.bfloat16, device="meta")
    st = torch.load(str(WANDIR / "models_t5_umt5-xxl-enc-bf16.pth"), map_location="cpu",
                    mmap=True, weights_only=True)
    t5.load_state_dict(st, strict=True, assign=True); del st
    t5 = t5.eval().requires_grad_(False).to(device=dev, dtype=torch.bfloat16)
    tok = HuggingfaceTokenizer(str(WANDIR / "google/umt5-xxl"), seq_len=512, clean="whitespace")
    print("all loaded", flush=True)

    from h3wan.sr_distill import forward_sr
    ids = sorted(f[:-4] for f in os.listdir(f"{PAIRS}/wan")
                 if f.endswith(".npy") and not f.endswith(".tmp.npy"))
    random.Random(1234).shuffle(ids)
    rows = []
    for vid in ids[:4]:
        tid, mask = tok([prompts.get(vid, "")], return_mask=True, add_special_tokens=True)
        with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
            cf = t5(tid.to(dev), mask.to(dev))
        ctx = cf[0, :int(mask.sum())].unsqueeze(0).to(torch.bfloat16)

        h3 = torch.from_numpy(np.load(f"{PAIRS}/h3/{vid}.npy")).float()[:, :5].unsqueeze(0).to(dev)
        w_hr = torch.from_numpy(np.load(f"{PAIRS}/wan/{vid}.npy")).float()[:, :5].unsqueeze(0).to(dev)
        hrpx = np.load(f"{PIX}/hr/{vid}.npy", mmap_mode="r")
        lrpx = np.load(f"{PIX}/lr/{vid}.npy", mmap_mode="r")

        g = torch.Generator(device="cpu").manual_seed(0)
        def tri3(z):
            b, c, t, hh, ww = z.shape
            y = F.interpolate(z.permute(0, 2, 1, 3, 4).flatten(0, 1), scale_factor=3,
                              mode="bilinear", align_corners=False)
            return y.unflatten(0, (b, t)).permute(0, 2, 1, 3, 4).contiguous()

        with torch.no_grad():
            low = mapper(h3)
            variants = {
                "upsampler": up(low, parallel=True).to(torch.bfloat16),
                "trilinear": tri3(low).to(torch.bfloat16),
                "trueHR_cond": w_hr.to(torch.bfloat16),
            }
            outs = {}
            for tag, cond_v in variants.items():
                noise = torch.randn(cond_v.shape, generator=torch.Generator().manual_seed(0)).to(dev, torch.bfloat16)
                o = forward_sr(dit, noise, cond_v, ctx, stream=False)
                outs[tag] = (o[0] if isinstance(o, tuple) else o).float()
            cond = variants["upsampler"]
            out = outs["upsampler"]
            px_out = tiny.decode_latents(out).float().clamp(-1, 1)
            px_cond = tiny.decode_latents(cond.float()).float().clamp(-1, 1)
            px_true = tiny.decode_latents(w_hr).float().clamp(-1, 1)
        n = px_out.shape[2]
        gt = torch.from_numpy(np.asarray(hrpx[:n])).permute(3, 0, 1, 2).unsqueeze(0).to(dev).float() / 127.5 - 1
        bic = F.interpolate(
            torch.from_numpy(np.asarray(lrpx[:n])).permute(0, 3, 1, 2).to(dev).float() / 127.5 - 1,
            scale_factor=3, mode="bicubic", align_corners=False).clamp(-1, 1).permute(1, 0, 2, 3).unsqueeze(0)
        rows.append(dict(clip=vid,
            wan_dit_out=round(psnr(px_out, gt), 2),
            cond_only=round(psnr(px_cond, gt), 2),
            true_wan=round(psnr(px_true, gt), 2),
            bicubic=round(psnr(bic, gt), 2),
            latent_vs_trueHR=round(psnr(out.float(), w_hr), 2),
            **{f"dit_{k}": round(psnr(
                tiny.decode_latents(v).float().clamp(-1, 1)[:, :, :n], gt), 2)
               for k, v in outs.items()}))
        print(json.dumps(rows[-1]), flush=True)
        torch.npu.empty_cache()
    a = lambda k: round(sum(r[k] for r in rows) / len(rows), 2)
    print("\nMEAN " + json.dumps({k: a(k) for k in rows[0] if k != "clip"}), flush=True)
    os._exit(0)


main()
