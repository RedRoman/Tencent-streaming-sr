"""Incremental 3-step refiner API. No lookahead or full-prefix recomputation.

Input/output: [1,24,chunk_latents,68,120]. User chunks are processed one latent
at a time; changing chunk boundaries does not change position-based noise.
Static prompt/silence must be known before stream start. An additional clean
cache-write forward follows the three denoising steps for each latent.
"""
import torch
from causal_attention import position_noise

class StreamingRefiner:
    SIGMAS=(.909375,6/7,2/3,0.)
    def __init__(self,runtime,sample,*,sample_id,seed=20261109):
        self.runtime=runtime;self.sample_id=sample_id;self.seed=seed
        self.runtime.start_stream(sample)
        self.model_forwards=0

    @torch.no_grad()
    def push(self,draft):
        if draft.ndim!=5 or draft.shape[:2]!=(1,24) or draft.shape[-2:]!=(68,120) or draft.shape[2]<1:
            raise ValueError('Require [1,24,T,68,120] with T>=1')
        outputs=[]
        for i in range(draft.shape[2]):
            clean=draft[0,:,i:i+1].float().to(self.runtime.device)
            noise=position_noise(clean.shape,self.sample_id,self.runtime.next_frame,self.seed).to(clean.device)
            x=(1-self.SIGMAS[0])*clean+self.SIGMAS[0]*noise
            for sigma,next_sigma in zip(self.SIGMAS,self.SIGMAS[1:]):
                x=x+(sigma-next_sigma)*self.runtime.stream_velocity(x,clean,sigma)
                self.model_forwards+=1
                if not torch.isfinite(x).all():raise RuntimeError('Nonfinite streaming state')
            self.runtime.stream_velocity(x,clean,0,commit=True);self.model_forwards+=1
            outputs.append(x.cpu())
        return torch.cat(outputs,1)[None]

    def audit(self):
        cache=self.runtime.cache or []
        return {'next_latent':self.runtime.next_frame,'denoise_steps_per_latent':3,
                'clean_cache_write_forwards_per_latent':1,'model_forwards':self.model_forwards,
                'lookahead_latents':0,'recomputed_history_latents':0,
                'cache_bytes':sum((v.k.numel()*v.k.element_size()+v.v.numel()*v.v.element_size()) for v in cache),
                'max_cached_video_frames':max((len(set(f for f in v.frames if f>=0)) for v in cache),default=0),
                'decoder_validated':False}
