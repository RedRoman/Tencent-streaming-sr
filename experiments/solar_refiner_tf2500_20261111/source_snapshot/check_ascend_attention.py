"""Compare the Ascend segment kernel, gradients, and host-cache reuse against SDPA."""
import json
import time
from pathlib import Path
import torch
import torch_npu
from solarwm.backends.minimax_h3.omni_causal import H3SegmentPlan,segment_attention as reference
from ascend_port import segment_attention,HostChunkKVCache

torch.set_num_threads(4)
torch.npu.set_device('npu:0')
torch.npu.conv.allow_hf32=False
torch.npu.matmul.allow_hf32=False
torch.manual_seed(20261109)
plan=H3SegmentPlan(segments=((0,4,((0,4),),(),None),(4,12,((0,12),),(),None)),sequence_length=12)
source=[torch.randn(1,12,2,128).bfloat16() for _ in range(3)]
ref=[x.float().requires_grad_() for x in source]
dev=[x.to('npu:0').requires_grad_() for x in source]
a=reference(*ref,plan,0)
b=segment_attention(*dev,plan,0)
delta=(a.detach()-b.float().cpu().detach()).abs().max().item()
grad=torch.randn_like(a)
ga=torch.autograd.grad(a,ref,grad)
gb=torch.autograd.grad(b,dev,grad.to(b))
gradient_errors=[(x-y.float().cpu()).abs().max().item() for x,y in zip(ga,gb)]
assert delta<.02,(delta,gradient_errors)
assert max(gradient_errors)<.04,gradient_errors
with torch.no_grad():
    changed=[x.detach().clone() for x in dev]
    changed[1][:,4:]*=-7
    changed[2][:,4:]+=20
    output=segment_attention(*changed,plan,0)
    leak=(output[:,:4]-b[:,:4]).abs().max().item()
    assert leak==0,leak
    cache=HostChunkKVCache(window=6)
    commit_plan=H3SegmentPlan(((0,4,((0,4),),(),('clean',0)),),4,cache)
    segment_attention(*[x[:,:4].detach() for x in dev],commit_plan,0)
    read_plan=H3SegmentPlan(((0,8,((0,8),),(('clean',0),),None),),8,cache)
    streamed=segment_attention(*[x[:,4:].detach() for x in dev],read_plan,0)
    cache_error=(streamed-b[:,4:]).float().abs().max().item()
    assert cache_error<.02,cache_error
    assert all(x.device.type=='cpu' for pair in cache.layers[0].values() for x in pair)
record={'status':'passed','forward_max_abs':delta,'gradient_max_abs':gradient_errors,
        'future_leak_max_abs':leak,'cache_max_abs':cache_error,'time':time.time(),
        'scope':'small random attention kernel and cache test; not a model quality result'}
Path('/opt/refiner-cache/h3-pretrained-causal-refiner-20261109/evidence/npu_attention_check.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
