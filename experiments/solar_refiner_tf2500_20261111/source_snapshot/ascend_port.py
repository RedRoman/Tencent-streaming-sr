"""Execution-only Ascend port: preserve upstream segments, RoPE, and clean-cache semantics."""
import torch
from solarwm.backends.minimax_h3 import omni_causal


def fused_attention(q,k,v):
    if q.device.type!='npu':
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),dropout_p=0.0).transpose(1,2)
    import torch_npu
    return torch_npu.npu_fusion_attention(q.contiguous(),k.contiguous(),v.contiguous(),
        q.shape[2],input_layout='BSND',scale=q.shape[-1]**-0.5,keep_prob=1.0,
        pre_tockens=2147483647,next_tockens=2147483647,sparse_mode=0)[0]


def segment_attention(query,key,value,plan,layer):
    if int(query.shape[1])!=plan.sequence_length:
        raise ValueError('Segment plan and query lengths differ')
    outputs=[]
    for start,stop,spans,cached,commit in plan.segments:
        keys=[key[:,a:b] for a,b in spans]
        values=[value[:,a:b] for a,b in spans]
        for chunk in cached:
            k,v=plan.cache.get(layer,chunk)
            keys.append(k.to(query.device));values.append(v.to(query.device))
        outputs.append(fused_attention(query[:,start:stop],torch.cat(keys,1),torch.cat(values,1)))
        if commit is not None:
            plan.cache.put(layer,commit,key[:,start:stop],value[:,start:stop])
    return torch.cat(outputs,1)


class HostChunkKVCache(omni_causal.H3ChunkKVCache):
    """Keep finalized K/V in host RAM; transfer only the layer currently attending."""
    def put(self,layer,name,key,value):
        if torch.is_grad_enabled():
            raise RuntimeError('A cache commit must not have an autograd graph')
        entries=self.layers.setdefault(layer,{})
        entries[name]=(key.detach().to('cpu',copy=True),value.detach().to('cpu',copy=True))
        kind,chunk=name
        for old in [n for n in entries if n[0]==kind and n[1]<=chunk-self.window]:
            entries.pop(old)


def install_attention_port():
    omni_causal.segment_attention=segment_attention
    omni_causal.H3ChunkKVCache=HostChunkKVCache


class PagedInference:
    """Frozen block paging; no changes to native block forward or model state names."""
    def __init__(self,model,device,log):
        self.model,self.device,self.log=model,torch.device(device),log
        self.calls=0
        self.host={}
        self.original=[]
        assert all(not p.requires_grad for p in model.parameters())
        for i,block in enumerate(model.transformer_blocks):
            self.original.append(block.forward)
            self.host[i]=[(m,k,p.detach()) for m in block.modules() for k,p in m._parameters.items() if p is not None]
            self.host[i]+=[(m,k,b.detach()) for m in block.modules() for k,b in m._buffers.items() if b is not None]
            def forward(*args,_index=i,**kwargs):
                if torch.is_grad_enabled():
                    raise RuntimeError('This paging implementation is inference-only')
                if _index==0:
                    self.calls+=1
                    self.log('model_forward_start',forward=self.calls,sequence=int(args[0].shape[1]))
                self.load(_index)
                try:
                    result=self.original[_index](*args,**kwargs)
                    if _index==len(self.original)-1:
                        self.log('model_forward_complete',forward=self.calls)
                    return result
                finally:
                    self.release(_index)
            block.forward=forward
        for name,child in model.named_children():
            if name!='transformer_blocks':
                child.to(self.device)

    def load(self,index):
        for module,key,host in self.host[index]:
            if key in module._parameters:
                module._parameters[key]=torch.nn.Parameter(host.to(self.device),requires_grad=False)
            else:
                module._buffers[key]=host.to(self.device)

    def release(self,index):
        for module,key,host in self.host[index]:
            if key in module._parameters:
                module._parameters[key]=torch.nn.Parameter(host,requires_grad=False)
            else:
                module._buffers[key]=host
