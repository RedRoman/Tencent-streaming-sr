"""H3 strict per-latent attention with real bounded incremental K/V.

Rows are [static, video]. Static queries never read video. Absolute video frame
IDs survive cache eviction and chunking. Cache records contain detached K/V,
never previous activations or inputs. No prefix recomputation is performed.
"""
from dataclasses import dataclass
import hashlib
import torch
import torch.nn.functional as F


def rotary(x, cos, sin):
    d = cos.shape[-1]
    a, b = x[..., :d].chunk(2, -1)
    c, s = cos.to(x)[None, :, None], sin.to(x)[None, :, None]
    return torch.cat((x[..., :d]*c + torch.cat((-b,a),-1)*s, x[...,d:]),-1).contiguous()


def position_noise(shape, sample, frame, seed):
    key = f'{seed}:{sample}:{frame}'.encode()
    g = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63-1))
    return torch.randn(shape, generator=g, dtype=torch.float32)


@dataclass(frozen=True)
class KV:
    k: torch.Tensor
    v: torch.Tensor
    frames: tuple

    def to(self, device):
        return KV(self.k.to(device), self.v.to(device), self.frames)


@dataclass(frozen=True)
class AttentionCall:
    frame_ids: tuple
    past: KV | None = None
    dense: bool = False
    sink: int = 1
    recent: int = 1


class CausalAttention:
    _attention_backend = None
    _parallel_config = None

    def __init__(self, chunk_size=256):
        self.call = None
        self.chunk_size = chunk_size
        self.captured = None
        self.capture = False

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        if attention_mask is not None:
            raise ValueError('External mask unsupported; use explicit frame IDs')
        call = self.call
        if call is None or len(call.frame_ids) != hidden_states.shape[1]:
            raise ValueError('Missing immutable attention call')
        q,k,v = [fn(hidden_states).unflatten(-1,(attn.heads,-1)) for fn in (attn.to_q,attn.to_k,attn.to_v)]
        q,k = attn.norm_q(q),attn.norm_k(k)
        if rotary_emb is not None:
            q,k = [rotary(x,*rotary_emb) for x in (q,k)]
        if self.capture:
            self.captured = KV(k.detach().cpu(),v.detach().cpu(),call.frame_ids)
        frames = call.frame_ids
        if call.past is not None:
            past = call.past.to(k.device)
            # Reject duplicate/late video rows; static+history must be finalized.
            current = [f for f in frames if f >= 0]
            if current and any(f >= min(current) for f in past.frames if f >= 0):
                raise ValueError('Cache contains current or future frames')
            k,v = torch.cat((past.k,k),1), torch.cat((past.v,v),1)
            keyframes = past.frames + frames
        else:
            keyframes = frames
        out = torch.zeros_like(q)
        for f in sorted(set(frames)):
            ids = torch.tensor([i for i,a in enumerate(frames) if a==f],device=q.device)
            permitted = [i for i,a in enumerate(keyframes) if call.dense or
                         (a<0 if f<0 else (a<0 or (0<=a<=f and (a<call.sink or a>=f-call.recent))))]
            if not permitted: raise ValueError('No attention keys')
            ki = torch.tensor(permitted,device=q.device)
            kg,vg = [z.index_select(1,ki).transpose(1,2).contiguous() for z in (k,v)]
            parts=[]
            for qi in ids.split(self.chunk_size):
                qg=q.index_select(1,qi).transpose(1,2).contiguous()
                parts.append(F.scaled_dot_product_attention(qg,kg,vg,dropout_p=0.0).transpose(1,2))
            out=out.index_copy(1,ids,torch.cat(parts,1))
        return attn.to_out[1](attn.to_out[0](out.flatten(2).to(q.dtype)))


def append_cache(old, new, *, next_frame, sink=1, recent=1):
    if old is None:
        merged=new
    else:
        merged=KV(torch.cat((old.k,new.k),1),torch.cat((old.v,new.v),1),old.frames+new.frames)
    keep=[i for i,f in enumerate(merged.frames) if f<0 or f<sink or f>=next_frame-recent]
    ids=torch.tensor(keep,dtype=torch.long,device=merged.k.device)
    return KV(merged.k.index_select(1,ids),merged.v.index_select(1,ids),tuple(merged.frames[i] for i in keep))
