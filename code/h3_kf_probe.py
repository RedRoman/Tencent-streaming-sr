"""Incremental cost of one HR keyframe latent from the H3 transformer.

This is the last unknown on the relay's critical path. The reference reader
needs one HR latent per interval; the HR VAE is already ruled out (5847 ms for
its 5-latent minimum chunk), so the question is purely what the transformer
costs to emit that latent.

Weights are fp32 on disk (61.7 GiB) and will not fit a 61 GiB card, so load in
bf16 (~31 GiB). Report what actually happens -- if it OOMs or the class will not
build on NPU, that is the finding.

Geometry: patch [1,2,2] on a 48x84 HR latent = 24x42 = 1008 tokens per latent
frame, against 8x14 = 112 for LR. One HR keyframe is ~9x the token count of one
LR step, which is the number the budget has to absorb.
"""
import json, os, sys, time
import torch
import torch_npu  # noqa: F401

CK = "ckpts/h3-base/transformer"


def main():
    print("tokens per latent frame: LR 8x14=112   HR 24x42=1008  (patch 2x2)", flush=True)
    t0 = time.time()
    try:
        from diffusers import MiniMaxH3Transformer3DModel as M
    except Exception as e:
        print("import failed:", str(e)[:200], flush=True); os._exit(1)
    try:
        m = M.from_pretrained(CK, torch_dtype=torch.bfloat16)
    except Exception as e:
        print("LOAD FAILED:", str(e)[:300], flush=True); os._exit(1)
    n = sum(p.numel() for p in m.parameters())
    print(f"loaded to CPU in {time.time()-t0:.0f}s  params {n/1e9:.2f}B  "
          f"bf16 size {n*2/2**30:.1f} GiB", flush=True)
    try:
        m = m.to("npu:0").eval()
    except Exception as e:
        print("TO-DEVICE FAILED:", str(e)[:300], flush=True); os._exit(1)
    print(f"on card, allocated {torch.npu.memory_allocated()/2**30:.1f} GiB", flush=True)

    import inspect
    sig = inspect.signature(m.forward)
    print("forward args:", list(sig.parameters)[:12], flush=True)
    json.dump(dict(params_b=round(n/1e9, 2), bf16_gib=round(n*2/2**30, 1),
                   loaded=True), open("results/h3_kf_probe.json", "w"), indent=2)
    os._exit(0)


main()
