"""Whole-runtime test with a tiny random H3; full 68x120 spatial input.
This checks runtime orchestration, not pretrained production model quality.
"""
from types import SimpleNamespace
import json,torch
from torch import nn
from diffusers import MiniMaxH3Transformer3DModel
from paged_runtime import Runtime,AdapterLinear,canonicalize
from causal_attention import CausalAttention

torch.set_num_threads(4);torch.manual_seed(9)
r=Runtime.__new__(Runtime);r.device=torch.device('cpu');r.log=lambda *a,**kw:None;r.resident=0
r.model=MiniMaxH3Transformer3DModel(num_attention_heads=1,attention_head_dim=128,hidden_size=32,num_layers=1,num_refiner_layers=1,ffn_dim=64,time_embed_hidden_dim=32,time_embed_dim=16).float().requires_grad_(False)
# Match mixed precision of the actual model.
for name,child in r.model.named_children():
    if name not in ('proj_in','audio_proj_in','time_proj','time_embedder','proj_out','audio_proj_out','rope'):child.bfloat16()
r.host={};r.processors=[];r.layer_params=[]
for i,block in enumerate(r.model.transformer_blocks):
    for path in ('attn.to_q','attn.to_k','attn.to_v','attn.to_out.0'):
        parent,key=path.rsplit('.',1);setattr(block.get_submodule(parent),key,AdapterLinear(block.get_submodule(path),2))
    proc=CausalAttention(chunk_size=256);block.attn.set_processor(proc);r.processors.append(proc);canonicalize(block,proc)
    r.layer_params.append(tuple(p for p in block.parameters() if p.requires_grad))
    r.host[i]=[(m,k,p.detach()) for m in block.modules() for k,p in m._parameters.items() if p is not None and not p.requires_grad]
    r.host[i]+=[(m,k,b.detach()) for m in block.modules() for k,b in m._buffers.items() if b is not None]
r.curves=SimpleNamespace(coords=None)
def pre(mod,args,kwargs):r.curves.coords=torch.zeros(len(kwargs['timestep']),8)
r.curves.pre=pre
r.condition=nn.Linear(96,32,bias=False);nn.init.zeros_(r.condition.weight)
r.params=[p for group in r.layer_params for p in group]+list(r.condition.parameters());r.cache=None
sample={'audio_latents':torch.randn(2,32,2),'prompt_embeds':torch.empty(0,5120)}
x=torch.randn(24,2,68,120);draft=torch.randn_like(x)
with torch.no_grad():
    offline=r.offline(x,draft,sample,[0,1],.7)
    r.start_stream(sample);r.stream_velocity(x[:,:1],draft[:,:1],0,commit=True)
    stream=r.stream_velocity(x[:,1:],draft[:,1:],.7)
    error=(offline-stream).abs().max().item();assert error<=.03125
    poses=r.video_positions([0,1,5,66]);assert float(poses[2040,0])!=1.
pred=r.offline(x,draft,sample,[0,1],.7,train=True);pred.square().mean().backward()
assert r.condition.weight.grad is not None and r.condition.weight.grad.abs().sum()>0
state=r.adapter_state();r.load_adapter(state)
assert all(torch.equal(v,r.adapter_state()[k]) for k,v in state.items())
try:r.stream_velocity(x[:,:1],draft[:,:1],.7)
except ValueError:stale=True
else:stale=False
assert stale
print(json.dumps({'status':'passed','scope':'tiny random H3, real full spatial 68x120, 2 latents only','cache_max_abs':error,'nonzero_condition_gradient':True,'save_reload_exact':True,'stale_cache_rejected':True,'nonuniform_absolute_RoPE':True,'optimizer_updates':0}))
