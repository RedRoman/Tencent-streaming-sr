"""Trace observed full-model parity failure without any optimizer update."""
import argparse,json,time,hashlib
from pathlib import Path
import torch,torch_npu
from torch import nn
from paged_runtime import Runtime
from causal_attention import position_noise
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);args=p.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);torch.npu.set_device('npu:0');torch.npu.conv.allow_hf32=False;torch.npu.matmul.allow_hf32=False;torch.set_float32_matmul_precision('highest');torch.manual_seed(20261109)
started=time.time()
def log(e,**v):print(json.dumps({'event':e,'seconds':time.time()-started,**v}),flush=True)
data=Path(args.data);m=json.loads((data/'manifest.json').read_text());sample=torch.load(m['samples'][0]['path'],weights_only=True);cond=torch.load(str(data/'conditioning.pt'),weights_only=True)
rt=Runtime(args.root,resident=24,log=log);phase='offline';captured={};handles=[]
static_count=2*cond['audio_latents'].shape[-1];last=static_count+2040
paths=['proj_in','time_proj','time_embedder','condition','norm_out','proj_out']
paths += ['transformer_blocks.0.'+s for s in ['adaln_proj','norm1','attn.to_q','attn.to_k','attn.to_v','attn.norm_q','attn.norm_k','attn.to_out.0','attn','norm2','ff']]
paths += [f'transformer_blocks.{i}' for i in range(50)]
def pick(value,path,phase):
    if isinstance(value,tuple):value=value[0]
    if not isinstance(value,torch.Tensor):return None
    value=value.detach()
    if value.ndim==2:
        if phase=='offline' and value.shape[0]==2:value=value[1:2]
        return value.cpu()
    if value.ndim>=3:
        n=value.shape[1]
        if phase=='offline':
            start=2040 if n==4080 else last if n==last+2040 else 0
        else:start=0
        return value[:1,start:start+4].cpu()
    return value.cpu()
for path in paths:
    mod=rt.condition if path=='condition' else rt.model.get_submodule(path)
    def hook(m,a,z,path=path):
        inp=pick(a[0],path,phase);val=pick(z,path,phase)
        if val is not None:captured[phase+':'+path]={'in':inp,'out':val,'shape':list(z[0].shape if isinstance(z,tuple) else z.shape)}
    handles.append(mod.register_forward_hook(hook))
hr=sample['reference'];draft=sample['upscaled'];sigma=.909375;noise=position_noise((24,1,68,120),sample['sample_id'],1,999);noisy=(1-sigma)*hr[:,1:2]+sigma*noise
with torch.no_grad():
    full=rt.offline(torch.cat((hr[:,:1],noisy),1),draft[:,:2],cond,[0,1],sigma)
    phase='static';rt.start_stream(cond)
    phase='clean';rt.stream_velocity(hr[:,:1],draft[:,:1],0,commit=True)
    phase='current';stream=rt.stream_velocity(noisy,draft[:,1:2],sigma)
for h in handles:h.remove()
def diff(a,b):
    if a is None or b is None or a.shape!=b.shape:return {'shape_mismatch':[list(a.shape) if a is not None else None,list(b.shape) if b is not None else None]}
    d=a.float()-b.float();return {'maxabs':float(d.abs().max()),'rms':float(d.square().mean().sqrt()),'relative_rms':float(d.square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-8))}
report={'optimizer_updates':0,'scope':'full real model two latent spatial68x120 selected-row trace; sampled rows cannot prove full tensor parity','full_output':diff(full.cpu(),stream.cpu()),'modules':{}}
for path in paths:
    a,b=captured.get('offline:'+path),captured.get('current:'+path)
    if a and b:report['modules'][path]={'input':diff(a['in'],b['in']),'output':diff(a['out'],b['out']),'shapes':[a['shape'],b['shape']]}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');torch.save(captured,out/'private_trace.pt');log('trace_complete',full_output=report['full_output'])
