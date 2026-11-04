"""Exact per-head mask evaluation on Ascend; no union across heads.

Teacher keeps upstream block selection/window/cache code. Only its execution
backend and complex-number RoPE representation change. Student groups all seven
frames together within each spatial block.
"""
import math
import torch
import torch_npu
from torch.utils.checkpoint import checkpoint

class amp:
    @staticmethod
    def autocast(enabled=True,dtype=torch.bfloat16):
        return torch.autocast('npu',dtype=torch.bfloat16,enabled=enabled and dtype!=torch.float32)

def fused_attention(q,k,v,allowed=None):
    """BSND inputs, optional per-head True=allowed; NPU uses True=masked."""
    q,k,v=(z.to(torch.bfloat16).contiguous() for z in (q,k,v))
    blocked=None if allowed is None else (~allowed).contiguous()
    return torch_npu.npu_fusion_attention(q,k,v,q.shape[2],'BSND',
        atten_mask=blocked,scale=q.shape[-1]**-.5,keep_prob=1.,sparse_mode=0)[0]

def flash_attention(q,k,v,q_lens=None,k_lens=None,**kwargs):
    assert not kwargs.get('causal',False),'use explicit frame-causal path'
    # Upstream SR context is padded to 512 before text projection, and original
    # inference attends to these projected padded tokens. Preserve that behavior.
    return fused_attention(q,k,v).to(q.dtype)

def exact_block_attention(q_w,k_w,v_w,attention_mask,num_heads):
    nq,bs,c=q_w.shape;nk=k_w.shape[0];d=c//num_heads
    q=q_w.reshape(1,nq*bs,num_heads,d).to(torch.bfloat16)
    k=k_w.reshape(1,nk*bs,num_heads,d).to(torch.bfloat16)
    v=v_w.reshape(1,nk*bs,num_heads,d).to(torch.bfloat16)
    outputs=[]
    # Chunk only query computation. Every selected key remains visible and each
    # head retains its own mask. This is mathematically identical full attention.
    blocks_per_call=max(1,1024//bs)
    def run(qpart,kall,vall,small_mask):
        b,nh,nqb,nkb=small_mask.shape
        # Scalar repeat_interleave is pathologically slow for large bool masks
        # on this torch_npu build. Broadcast-copy is exactly equal and ~4ms.
        allow=small_mask.reshape(b,nh,nqb,1,nkb,1).expand(b,nh,nqb,bs,nkb,bs).reshape(b,nh,nqb*bs,nkb*bs)
        return fused_attention(qpart,kall,vall,allow)
    for b0 in range(0,nq,blocks_per_call):
        b1=min(nq,b0+blocks_per_call)
        args=(q[:,b0*bs:b1*bs],k,v,attention_mask[:,:,b0:b1])
        if torch.is_grad_enabled() and q.requires_grad:
            outputs.append(checkpoint(run,*args,use_reentrant=False))
        else:outputs.append(run(*args))
    return torch.cat(outputs,dim=1).reshape(1,nq*bs,c).to(q_w.dtype)

def _gather_plan(mask_1h, bs, device):
    """Group query blocks by their allowed key set.

    The block mask is small ([nqb, nkb]) but expanding it to token level costs
    4.54 GiB per layer, which is where the original path spent 93% of its time.
    Running each query block against only its allowed keys needs no token-level
    mask at all. Query blocks sharing an identical key set are batched together.
    """
    nqb = mask_1h.shape[0]
    rows = {}
    nz = [torch.nonzero(mask_1h[i], as_tuple=False).flatten() for i in range(nqb)]
    for i, r in enumerate(nz):
        rows.setdefault(tuple(r.tolist()), []).append(i)
    ar = torch.arange(bs, device=device)
    plan = []
    for keyset, qs in rows.items():
        qidx = torch.tensor(qs, device=device)
        qtok = (qidx[:, None] * bs + ar[None]).reshape(-1)
        kidx = torch.tensor(keyset, device=device)
        ktok = (kidx[:, None] * bs + ar[None]).reshape(-1)
        plan.append((qtok, ktok))
    return plan


GATHER_STATS = {"calls": 0, "heads_differ": 0, "plans_built": 0, "groups": 0}
_PLAN_CACHE = {}


def gather_block_attention(q_w, k_w, v_w, attention_mask, num_heads):
    """Exact block-sparse attention without materializing a token-level mask.

    Falls back to the original path when heads disagree, since the gather is
    only valid for a mask shared across heads.
    """
    nq, bs, c = q_w.shape
    nk = k_w.shape[0]
    d = c // num_heads
    m = attention_mask
    GATHER_STATS["calls"] += 1
    if m.dim() == 4:
        if not bool((m[0] == m[0, :1]).all()):
            GATHER_STATS["heads_differ"] += 1
            return exact_block_attention(q_w, k_w, v_w, attention_mask, num_heads)
        m1 = m[0, 0]
    else:
        m1 = m
    packed = m1.to(torch.uint8).cpu().numpy().tobytes()
    key = (hash(packed), nq, nk, bs)
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = _gather_plan(m1, bs, q_w.device)
        GATHER_STATS["plans_built"] += 1
        GATHER_STATS["groups"] += len(plan)
        if len(_PLAN_CACHE) > 64:
            _PLAN_CACHE.clear()
        _PLAN_CACHE[key] = plan

    q = q_w.reshape(1, nq * bs, num_heads, d).to(torch.bfloat16)
    k = k_w.reshape(1, nk * bs, num_heads, d).to(torch.bfloat16)
    v = v_w.reshape(1, nk * bs, num_heads, d).to(torch.bfloat16)
    out = torch.empty_like(q)
    for qtok, ktok in plan:
        o = torch_npu.npu_fusion_attention(
            q.index_select(1, qtok), k.index_select(1, ktok), v.index_select(1, ktok),
            num_heads, "BSND", scale=d ** -0.5, keep_prob=1.)[0]
        out.index_copy_(1, qtok, o)
    return out.reshape(1, nq * bs, c).to(q_w.dtype)


def rope_params(max_seq_len,dim,theta=10000,device=None):
    # Compute angles on CPU in official float64, then transfer cos/sin as FP32.
    angle=torch.outer(torch.arange(max_seq_len,dtype=torch.float64,device='cpu'),
        1./torch.pow(theta,torch.arange(0,dim,2,dtype=torch.float64,device='cpu')/dim))
    return torch.stack((angle.cos(),angle.sin()),-1).float().to(device)

def rope_apply(x,grid_sizes,freqs,t_offset=None):
    n,c=x.shape[2],x.shape[3]//2
    ft,fh,fw=freqs.split([c-2*(c//3),c//3,c//3],dim=1)
    grids=grid_sizes.cpu().tolist()
    offsets=[0]*len(grids) if t_offset is None else ([t_offset]*len(grids) if isinstance(t_offset,int) else list(t_offset))
    result=[]
    for i,(f,h,w) in enumerate(grids):
        off=int(offsets[i]);s=f*h*w
        factors=torch.cat([ft[off:off+f].view(f,1,1,-1,2).expand(f,h,w,-1,2),
                           fh[:h].view(1,h,1,-1,2).expand(f,h,w,-1,2),
                           fw[:w].view(1,1,w,-1,2).expand(f,h,w,-1,2)],dim=3).reshape(s,1,c,2)
        pairs=x[i,:s].float().reshape(s,n,c,2)
        re=pairs[...,0]*factors[...,0]-pairs[...,1]*factors[...,1]
        im=pairs[...,0]*factors[...,1]+pairs[...,1]*factors[...,0]
        y=torch.stack((re,im),-1).flatten(2)
        result.append(torch.cat((y,x[i,s:].float()),dim=0))
    return torch.stack(result).to(x.dtype)

def sinusoidal_embedding_1d(dim,position):
    # Official embedding calculated on CPU float64 avoids unsupported NPU FP64.
    pos=position.detach().cpu().double()
    f=torch.outer(pos,torch.pow(10000,-torch.arange(dim//2,dtype=torch.float64)/(dim//2)))
    return torch.cat((f.cos(),f.sin()),1).float().to(position.device)

def install():
    from vendor.ultraflash_npu import models as mod,attention as attn
    import os as _os
    _backend = _os.environ.get('SOLH3_ATTN', 'gather')
    attn._block_sparse_attn_fallback = (
        gather_block_attention if _backend == 'gather' else exact_block_attention)
    mod.flash_attention=flash_attention
    mod.rope_params=rope_params;mod.rope_apply=rope_apply
    mod.sinusoidal_embedding_1d=sinusoidal_embedding_1d
    return mod
