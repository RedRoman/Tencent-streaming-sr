"""Numerical causality/cache/gradient/noise tests; synthetic tensors, no model claim."""
import json
from types import SimpleNamespace
import torch
from torch import nn
from causal_attention import CausalAttention,AttentionCall,append_cache,position_noise

def main():
    torch.manual_seed(901);torch.set_num_threads(2)
    a=SimpleNamespace(heads=2,to_q=nn.Linear(12,16,bias=False),to_k=nn.Linear(12,16,bias=False),to_v=nn.Linear(12,16,bias=False),norm_q=nn.RMSNorm(8),norm_k=nn.RMSNorm(8),to_out=[nn.Linear(16,12,bias=False),nn.Identity()])
    p=CausalAttention(chunk_size=3); ids=(-1,-1)+tuple(f for f in range(8) for _ in range(4))
    x=torch.randn(1,len(ids),12,requires_grad=True)
    p.call=AttentionCall(ids);full=p(a,x)
    changed=x.detach().clone();changed[:,14:]=torch.randn_like(changed[:,14:])*10
    future=(p(a,changed)[:,:14]-full[:,:14]).abs().max().item();assert future==0
    cache=None;outputs=[]
    for start,end,f in [(0,2,-1)]+[(2+4*i,6+4*i,i) for i in range(8)]:
        p.call=AttentionCall((f,)*(end-start),past=cache);p.capture=True
        outputs.append(p(a,x[:,start:end]));cache=append_cache(cache,p.captured,next_frame=f+1)
        assert len(cache.frames)<=2+8
    streamed=torch.cat(outputs,1);error=(full-streamed).abs().max().item();assert error<2e-6
    full.square().mean().backward();assert x.grad.abs().sum()>0
    n1=torch.stack([position_noise((24,68,120),'sample',f,42) for f in range(7)])
    n2=torch.cat([torch.stack([position_noise((24,68,120),'sample',f,42) for f in c]) for c in [range(2),range(2,6),range(6,7)]])
    assert torch.equal(n1,n2)
    print(json.dumps({'future_maxabs':future,'offline_cache_maxabs':error,'nonzero_gradient':True,'noise_chunking_exact':True,'bounded_cache_rows':len(cache.frames),'scope':'synthetic attention kernel only'}))
if __name__=='__main__':main()
