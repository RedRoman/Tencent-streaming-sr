"""Native H3 temporal protocol and independently reset Wan main/tail calls."""
import sys, json
from pathlib import Path
import torch
from safetensors import safe_open
from .common import H3, WAN, geometry

sys.path.insert(0, str(H3.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

class VAEs:
    def __init__(self, device, h3=True, h3_token_drop=3):
        self.device = device
        if h3:
            from video_vae.minimax_h3_video_vae import _ensure_vae_parallel_state
            from video_vae.klvae import AutoencoderKLLegacy
            _ensure_vae_parallel_state()
            cfg = AutoencoderKLLegacy.load_config(str(H3/'source'))
            # Construct the official module on meta, then retain only the encoder.
            # No H3 decoder weights or any generation transformer are loaded.
            with torch.device('meta'):
                model = AutoencoderKLLegacy.from_config(cfg, clip_length=17, token_drop=h3_token_drop,
                    encoder_tiling=True, decoder_tiling=False, parallel_tiling=True,
                    tile_size=256, tile_overlap_min=64, encoder_parallel=0,
                    decoder_parallel=0, chunk_dim=-1)
            del model.decoder; del model.post_quant_conv
            with safe_open(str(H3/'source/model.safetensors'), framework='pt', device='cpu') as f:
                state = {k:f.get_tensor(k) for k in f.keys() if k.startswith(('encoder.','quant_conv.'))}
            model.load_state_dict(state, strict=True, assign=True)
            self.h3 = model.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)
            cfg = json.loads((H3/'config.json').read_text())
            self.hm = torch.tensor(cfg['latents_mean'], device=device).view(1,24,1,1,1)
            self.hs = torch.tensor(cfg['latents_std'], device=device).view(1,24,1,1,1)
            print(f'H3 encoder loaded; official clip_length=17 token_drop={h3_token_drop}; ImageNet pixels; posterior sampling', flush=True)
        from vendor.wan_vae import WanVAE
        self.wan = WanVAE(vae_pth=str(WAN), dtype=torch.bfloat16, device=device)
        self.wan.model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
        print('Wan VAE loaded; normalized (posterior_mean - channel_mean) / channel_std; fresh caches per call', flush=True)

    @torch.no_grad()
    def encode(self, rgb):
        rgb = rgb.to(self.device)
        pixels = self.h3.transform(rgb.permute(0,2,1,3,4)).permute(0,2,1,3,4).to(torch.bfloat16)
        with torch.autocast('npu', dtype=torch.bfloat16):
            hz = self.h3.encode_base(pixels, process_image=False)
        hz = (hz.float()-self.hm)/self.hs
        main = self.wan_encode(rgb[:,:,:17])
        tail = self.wan_encode(rgb[:,:,17:22])
        geometry(rgb, hz, main, tail)
        return hz, main, tail

    @torch.no_grad()
    def wan_encode(self, rgb):
        self.wan.model.clear_cache()
        with torch.autocast('npu', dtype=torch.bfloat16):
            z = self.wan.model.encode((rgb*2-1).to(torch.bfloat16), self.wan.scale).float()
        self.wan.model.clear_cache()
        return z

    @torch.no_grad()
    def decode(self, z):
        assert z.shape[1:3] == (16,7)
        parts = []
        for part, expected in ((z[:,:,:5],17), (z[:,:,5:],5)):
            self.wan.model.clear_cache()
            with torch.autocast('npu', dtype=torch.bfloat16):
                rgb = self.wan.model.decode(part.to(self.device,dtype=torch.bfloat16), self.wan.scale)
            self.wan.model.clear_cache()
            assert rgb.shape[2] == expected, rgb.shape
            parts.append((rgb.float().clamp(-1,1)+1)/2)
        return torch.cat(parts, dim=2)
