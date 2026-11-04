"""Why is the refined latent so far from HR in pixel space?

Decompose the gap: compare trilinear(LR) (the refiner's own starting point),
the refiner output, and HR. If the refiner is worse than its input, it is
destroying signal -- an objective/sampler bug, not undertraining.
"""
import os, sys, random
import torch, torch.nn.functional as F
import torch_npu  # noqa: F401
from diffusers import AutoencoderKLMiniMaxH3
from solh3_model import SolH3Refiner
from solh3_train import Pairs, upsample_lr, HIGH, VAL_IDS

CK = sys.argv[1]
VAE_DIR = os.environ["HOME"] + "/modelarts/user-job-dir/sol_h3/ckpts/h3-base/vae"


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b) + 1e-12))


def roll(m, x0, lrup, nfe, dev):
    x = x0.clone()
    ts = torch.linspace(1, 0, nfe + 1, device=dev)
    with torch.no_grad():
        for i in range(nfe):
            with torch.autocast("npu", dtype=torch.bfloat16):
                v = m(x, lrup, ts[i].expand(1)).float()
            x = x + (ts[i + 1] - ts[i]) * v
    return x


def main():
    dev = "npu:0"
    blob = torch.load(CK, map_location="cpu", weights_only=False)
    a = blob["args"]
    obj = a.get("objective", "noise")
    m = SolH3Refiner(width=a["width"], time_taps=a["time_taps"],
                         cond_mode=a["cond"],
                         learned_upsample=a.get("learned_upsample", False)).to(dev)
    m.load_state_dict(blob["model"]); m.eval()
    print("ckpt step=%s chunk=%s objective=%s" % (blob["step"], a["chunk"], obj), flush=True)

    ids = sorted(f[:-8] for f in os.listdir(HIGH) if f.endswith(".pt.part"))
    random.Random(1234).shuffle(ids)
    ds = Pairs(ids[:VAL_IDS], a.get("lr_source") or "h3")
    lo, hi = ds.get(0)
    lo = lo[:, :5].unsqueeze(0).to(dev); hi = hi[:, :5].unsqueeze(0).to(dev)
    lrup = upsample_lr(lo)

    print("\nlatent  HR mean %+.3f std %.3f | LRup mean %+.3f std %.3f"
          % (hi.mean(), hi.std(), lrup.mean(), lrup.std()), flush=True)
    base = psnr(lrup, hi)
    print("latent PSNR  trilinear(LR) vs HR = %.3f dB  <-- the bar to beat" % base, flush=True)

    for nfe in (1, 2, 4, 8):
        x0 = lrup if obj == "bridge" else torch.randn_like(hi)
        x = roll(m, x0, lrup, nfe, dev)
        print("  NFE=%d: out mean %+.3f std %.3f  latent PSNR %6.3f dB  (delta %+.3f)"
              % (nfe, x.mean(), x.std(), psnr(x, hi), psnr(x, hi) - base), flush=True)

    vae = AutoencoderKLMiniMaxH3.from_pretrained(VAE_DIR, torch_dtype=torch.float32).to(dev).eval()
    mn = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
    sd = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)

    def dec(z):
        with torch.no_grad():
            return (vae.decode(z * sd + mn).sample.float().clamp(-1, 1) + 1) / 2

    x = roll(m, lrup if obj == "bridge" else torch.randn_like(hi), lrup, 2, dev)
    g, p, l = dec(hi), dec(x), dec(lrup)
    print("\npixel PSNR vs HR-decode:  refined %.3f dB | trilinear(LR)-decode %.3f dB"
          % (psnr(p, g), psnr(l, g)), flush=True)
    print("pixel  GT mean %.3f std %.3f | pred mean %.3f std %.3f"
          % (g.mean(), g.std(), p.mean(), p.std()), flush=True)
    os._exit(0)


main()
