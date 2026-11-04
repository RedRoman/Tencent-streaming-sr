"""Narrow down what breaks the SR-DiT: stream path, frame count, or grid padding.

Identity probe: feed the true Wan latent as both noise and condition. The model
predicts x0 = noise - flow, so a working model should return something close to
that latent. Vary one setting at a time.
"""
import json, os, random, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch_npu  # noqa: F401

HOME = os.environ["HOME"]; ROOT = Path(f"{HOME}/modelarts/user-job-dir/sol_h3")
sys.path.insert(0, str(ROOT / "h2w"))
SRW = "/temp/rongxiang_workspace_backup_20260921/rongxiang/UltraFlash/inference/checkpoints/2K-Causal-Sparse-SR-Dit-fidelity-step_12800.pth"
PAIRS = ROOT / "data/wanpairs"


def build(mod, dev, causal, blk, chunk):
    with torch.device("meta"):
        d = mod.Transformer3DModel(model_type="t2v", patch_size=(1, 2, 2), text_len=512,
            in_dim=32, dim=1536, ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16,
            num_heads=12, num_layers=30, window_size=(-1, -1), qk_norm=True,
            cross_attn_norm=True, eps=1e-6, use_sparse_attn=True, sparse_causal=causal,
            sparse_block_size=blk, sparse_top_k_ratio=1., sparse_kv_ratio=3.,
            sparse_local_range=9, sparse_use_kernel=False, stream_chunk_size=chunk,
            sparse_ref_spatial_tokens=1560, rope_max_seq_len=1024, rope_theta=10000.,
            rope_cache_multiple=1024)
    st = torch.load(SRW, map_location="cpu", mmap=True, weights_only=False)["model"]
    d.load_state_dict(st, strict=True, assign=True); del st
    d = d.to(dev).to(torch.bfloat16)
    d.time_embedding.float(); d.time_projection.float(); d.head.float()
    return d.eval().requires_grad_(False)


def main():
    from h3wan.npu_attention import install
    from h3wan.sr_distill import forward_sr
    mod = install(); dev = "npu:0"
    ids = sorted(f[:-4] for f in os.listdir(f"{PAIRS}/wan")
                 if f.endswith(".npy") and not f.endswith(".tmp.npy"))
    random.Random(1234).shuffle(ids)
    w_all = torch.from_numpy(np.load(f"{PAIRS}/wan/{ids[0]}.npy")).float().unsqueeze(0).to(dev)
    ctx = torch.zeros(1, 512, 4096, device=dev, dtype=torch.bfloat16)

    def match(o, ref):
        return round(float(-10 * torch.log10(F.mse_loss(o.float(), ref) + 1e-12)), 2)

    # A: current config, stream vs non-stream, several frame counts
    dit = build(mod, dev, True, (2, 8, 8), 2)
    for nf in (2, 4, 5, 7):
        w = w_all[:, :, :nf]; wb = w.to(torch.bfloat16)
        for stream in (True, False):
            try:
                with torch.no_grad():
                    o = forward_sr(dit, wb, wb, ctx, stream=stream, offset=0)
                    if isinstance(o, tuple):
                        o = o[0]
                print(json.dumps(dict(cfg="blk(2,8,8) chunk2", frames=nf, stream=stream,
                                      identity_psnr=match(o, w),
                                      out_std=round(float(o.float().std()), 3),
                                      ref_std=round(float(w.std()), 3))), flush=True)
            except Exception as e:
                print(json.dumps(dict(cfg="blk(2,8,8) chunk2", frames=nf, stream=stream,
                                      error=str(e)[:120])), flush=True)
    del dit; torch.npu.empty_cache()

    # B: non-causal / no sparse, to see whether the sparse machinery is at fault
    for causal, blk, chunk, tag in ((False, (2, 8, 8), 2, "non_causal"),
                                    (True, (1, 8, 8), 1, "blk(1,8,8) chunk1")):
        try:
            d2 = build(mod, dev, causal, blk, chunk)
            w = w_all[:, :, :5]; wb = w.to(torch.bfloat16)
            with torch.no_grad():
                o = forward_sr(d2, wb, wb, ctx, stream=False, offset=0)
                if isinstance(o, tuple):
                    o = o[0]
            print(json.dumps(dict(cfg=tag, frames=5, stream=False,
                                  identity_psnr=match(o, w),
                                  out_std=round(float(o.float().std()), 3))), flush=True)
            del d2; torch.npu.empty_cache()
        except Exception as e:
            print(json.dumps(dict(cfg=tag, error=str(e)[:140])), flush=True)
            torch.npu.empty_cache()
    os._exit(0)


main()
