"""CPU check of one full-width real H3 block, not full-model/NPU validation."""
import argparse,json,time,hashlib,inspect
from pathlib import Path
import torch
from safetensors.torch import load_file
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock
from causal_attention import AttentionCall,CausalAttention,append_cache
from paged_runtime import AdapterLinear,LayerCall,Recompute
from open_h3.adapters import CurveContext,CurveLinear,CURVE_FILE,CURVE_TABLE

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);args=p.parse_args()
root=Path(args.root);out=Path(args.output);out.mkdir(exist_ok=False,parents=True)
torch.set_num_threads(4);torch.manual_seed(99);start=time.time()
def log(event,**data):print(json.dumps({'event':event,'seconds':time.time()-start,**data}),flush=True)
config=json.loads((root/'merged_transformer/config.json').read_text())
with torch.device('meta'):
    block=MiniMaxH3TransformerBlock(hidden_size=config['hidden_size'],num_attention_heads=config['num_attention_heads'],attention_head_dim=config['attention_head_dim'],ffn_dim=config['ffn_dim'],time_embed_dim=config['time_embed_dim'],norm_eps=config['norm_eps'],qk_norm_eps=config['qk_norm_eps'])
index=json.loads((root/'merged_transformer/diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map'];prefix='transformer_blocks.0.'
state={}
for file in sorted(set(v for k,v in index.items() if k.startswith(prefix))):
    shard=load_file(str(root/'merged_transformer'/file))
    state.update({k[len(prefix):]:v for k,v in shard.items() if k.startswith(prefix)})
block.load_state_dict(state,strict=True,assign=True);block.eval().requires_grad_(False)
log('actual_block_loaded',parameters=sum(p.numel() for p in block.parameters()),dtypes=sorted(set(str(p.dtype) for p in block.parameters())))
ctx=CurveContext(root/'assets'/CURVE_TABLE);ctx.strength=1.;ctx.pre(None,(),{'timestep':torch.tensor([1.,.1])})
curves=load_file(str(root/'assets'/CURVE_FILE));key='transformer.transformer_blocks.0.adaln_proj.linear'
# Discover the exact released projection key rather than guessing module depth.
keys=[k for k in curves if k.startswith('transformer.transformer_blocks.0.') and k.endswith('.curve_A.weight')];assert len(keys)==1
name=keys[0][len('transformer.transformer_blocks.0.'):-len('.curve_A.weight')];parent,leaf=name.rsplit('.',1)
setattr(block.get_submodule(parent),leaf,CurveLinear(block.get_submodule(name),curves[keys[0]].float(),curves[keys[0].replace('.curve_A.weight','.curve_B.weight')].float(),ctx));del curves
for path in ('attn.to_q','attn.to_k','attn.to_v','attn.to_out.0'):
    parent,key=path.rsplit('.',1);setattr(block.get_submodule(parent),key,AdapterLinear(block.get_submodule(path),8))
processor=CausalAttention(chunk_size=3);block.attn.set_processor(processor)
params=tuple(p for p in block.parameters() if p.requires_grad)
x=torch.randn(1,6,config['hidden_size'],dtype=torch.bfloat16,requires_grad=True)
temb=torch.randn(2,config['time_embed_dim']);indices=torch.tensor([1,2,0,0,3,3]);rope=(torch.ones(6,96),torch.zeros(6,96));ids=(-1,-1,0,0,1,1)
processor.call=AttentionCall(ids)
y=block(x,temb,indices,rope);loss=y.float().square().mean();loss.backward()
grad=sum(float(p.grad.float().square().sum()) for p in params)**.5;assert grad>0
reference_grads=[p.grad.detach().clone() for p in params];reference_xgrad=x.grad.clone()
for pp in params:pp.grad=None
x.grad=None
# Exercise the exact custom recompute autograd implementation against ordinary
# autograd on real 33B-model block weights, with CPU paging methods as no-ops.
class CPU:
    device=torch.device('cpu')
    def load_layer(self,i):pass
    def release_layer(self,i):pass
    def layer(self,i,h,call,release=True):
        ctx.coords=call.curve;processor.call=call.attention
        return block(h,call.temb,call.adaln,call.rope)
r=CPU();call=LayerCall(temb,indices,rope,ctx.coords,AttentionCall(ids))
z=Recompute.apply(r,0,call,x,*params);z.float().square().mean().backward()
param_grad_error=max(float((p.grad-ref).abs().max()) for p,ref in zip(params,reference_grads));input_grad_error=float((x.grad-reference_xgrad).abs().max())
assert param_grad_error==0 and input_grad_error==0
log('real_block_recompute_backward_parity',param_grad_max_abs=param_grad_error,input_grad_max_abs=input_grad_error,gradient_norm=grad)
with torch.no_grad():
    processor.call=AttentionCall(ids);baseline=block(x.detach(),temb,indices,rope)
    changed=x.detach().clone();changed[:,4:]*=-5
    future=block(changed,temb,indices,rope);leak=float((future[:,:4]-baseline[:,:4]).abs().max());assert leak==0
    cache=None;pieces=[]
    for lo,hi,f in [(0,2,-1),(2,4,0),(4,6,1)]:
        processor.call=AttentionCall((f,)*(hi-lo),past=cache);processor.capture=True
        pieces.append(block(x[:,lo:hi],temb,indices[lo:hi],tuple(v[lo:hi] for v in rope)))
        cache=append_cache(cache,processor.captured,next_frame=f+1)
    error=float((torch.cat(pieces,1)-baseline).abs().max())
    # BF16 GEMMs can vary by sequence batch shape; preserve observed error.
    assert error<=.03125
report={'status':'passed','scope':'one full-width block from actual merged OpenH3 weights plus Sharp curve; CPU six rows only; NOT full model or production geometry or NPU test','parameters':sum(p.numel() for p in block.parameters()),'gradient_norm':grad,'parameter_gradient_parity_maxabs':param_grad_error,'input_gradient_parity_maxabs':input_grad_error,'future_maxabs':leak,'cache_maxabs':error,'optimizer_updates':0,'seconds':time.time()-start,'implementation_sha256':hashlib.sha256(Path(inspect.getfile(MiniMaxH3TransformerBlock)).read_bytes()).hexdigest()}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');log('complete',**report)
