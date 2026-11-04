import types,math
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
from .common import ROOT
from .npu_attention import install,sinusoidal_embedding_1d

RUN=ROOT/'sr_distill_round4'
WEIGHTS=ROOT/'assets/ultraflash/2K-Causal-Sparse-SR-Dit-fidelity-step_12800.pth'
MAPPER_CKPT=ROOT/'stages/03_joint_up/final.pt'
TEXT_ROOT=ROOT/'assets/wan'

def load_sr(device,student=False):
    mod=install()
    with torch.device('meta'):
        model=mod.Transformer3DModel(model_type='t2v',patch_size=(1,2,2),text_len=512,
            in_dim=32,dim=1536,ffn_dim=8960,freq_dim=256,text_dim=4096,out_dim=16,
            num_heads=12,num_layers=30,window_size=(-1,-1),qk_norm=True,cross_attn_norm=True,eps=1e-6,
            use_sparse_attn=True,sparse_causal=not student,sparse_block_size=(7,8,8) if student else (2,8,8),
            sparse_top_k_ratio=1.,sparse_kv_ratio=3.,sparse_local_range=9,sparse_use_kernel=False,
            stream_chunk_size=7 if student else 2,sparse_ref_spatial_tokens=1560,
            rope_max_seq_len=1024,rope_theta=10000.,rope_cache_multiple=1024)
    state=torch.load(WEIGHTS,map_location='cpu',mmap=True,weights_only=False)['model']
    model.load_state_dict(state,strict=True,assign=True)
    del state
    # FP32 master weights; original time/modulation/normalization computed FP32.
    # Linear/conv/attention are BF16 under explicit autocast.
    model=model.to(device).requires_grad_(student)
    if student:
        model.train()
    else:
        model.to(dtype=torch.bfloat16)
        model.time_embedding.float();model.time_projection.float();model.head.float()
        model.eval()
    print('SR DiT loaded strict',dict(student=student,params=sum(p.numel() for p in model.parameters())),flush=True)
    return model

def projected_context(model,context):
    if context.ndim==2:context=context[None]
    context=torch.nn.functional.pad(context,(0,0,0,512-context.shape[1]))
    return model.text_embedding(context.to(model.patch_embedding.weight.device))

def forward_sr(model,noise,condition,context,*,stream=False,offset=0,caches=None,checkpoint_blocks=False):
    """Native t=1000, one-step x0=noise-flow. Teacher state matches upstream."""
    b,c,t,h,w=noise.shape;assert b==1
    x=model.patch_embedding(torch.cat((noise,condition),1))
    grid=torch.tensor([x.shape[2:]],dtype=torch.long,device='cpu')
    model._ensure_rope_freqs(grid,temporal_offset=offset,device=x.device)
    x=x.flatten(2).transpose(1,2);seq=torch.tensor([x.shape[1]],dtype=torch.long)
    with torch.autocast('npu',enabled=False):
        emb=sinusoidal_embedding_1d(model.freq_dim,torch.full((b,),1000.,device=x.device))
        e=model.time_embedding(emb);e0=model.time_projection(e).unflatten(1,(6,model.dim))
    ctx=projected_context(model,context)
    kwargs=dict(e=e0,seq_lens=seq,grid_sizes=grid,freqs=model.freqs,context=ctx,context_lens=None,
                local_num=None,local_range=9,temporal_offset=offset,kv_len=3,train_img=False)
    _,gh,gw=grid[0].tolist();nb=math.ceil(gh/8)*math.ceil(gw/8)
    tb=2 if stream else 7
    kwargs['top_k']=min(max(int(nb*nb*1560/(gh*gw)),1),max(nb*nb*math.ceil(t/tb)-1,1))
    if stream:
        if caches is None:caches=[dict(k=None,v=None) for _ in model.blocks]
        if offset==0:model.clear_cross_kv();model.reinit_cross_kv(ctx)
        for i,block in enumerate(model.blocks):
            x,k,v=block(x,is_stream=True,pre_cache_k=caches[i]['k'],pre_cache_v=caches[i]['v'],**kwargs)
            caches[i]=dict(k=k,v=v)
    else:
        # Never reuse detached cross-attention caches in the trainable student.
        model.clear_cross_kv()
        for block in model.blocks:
            if checkpoint_blocks and torch.is_grad_enabled():
                x=checkpoint(block,x,use_reentrant=False,**kwargs)
            else:x=block(x,**kwargs)
    # Upstream Head is explicitly FP32. Return BF16 flow like official inference.
    with torch.autocast('npu',enabled=False):out=model.head(x.float(),e)
    flow=torch.stack(model.unpatchify(out,grid)).to(noise.dtype)
    return noise-flow,caches

class Student(torch.nn.Module):
    def __init__(self,device):super().__init__();self.dit=load_sr(device,True)
    def forward(self,noise,condition,context):
        return forward_sr(self.dit,noise,condition,context,checkpoint_blocks=self.training)[0]
