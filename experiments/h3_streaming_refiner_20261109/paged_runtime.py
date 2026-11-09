"""Frozen Open H3 33B base; FP32 LoRA and clean-draft conditioning.

Single-device layer paging plus custom recompute backward keeps all frozen
weights on CPU between layer calls. A bounded resident prefix is optional.
Static/audio are known-in-advance silence only in the first pilot. Text can be
supplied explicitly; no mismatched cached caption is substituted.
"""
from dataclasses import dataclass
import contextlib
import torch
from torch import nn
from safetensors.torch import load_file
from diffusers import MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents, _temporal_position_grid
from open_h3.geometry import MiniMaxH3Geometry, build_t2va_layout, pack_audio_latents, unpatchify_video_rows
from open_h3.adapters import CurveContext, CurveLinear, CURVE_FILE, CURVE_TABLE
from pathlib import Path
from causal_attention import CausalAttention, AttentionCall, KV, append_cache


class AdapterLinear(nn.Module):
    def __init__(self,base,rank=8):
        super().__init__();self.base=base
        self.a=nn.Parameter(torch.randn(rank,base.in_features,dtype=torch.float32)*.01)
        self.b=nn.Parameter(torch.zeros(base.out_features,rank,dtype=torch.float32))
    @property
    def weight(self):return self.base.weight
    def forward(self,x):
        return self.base(x.to(self.base.weight.dtype)) + ((x.float()@self.a.T)@self.b.T).to(x.dtype)


@dataclass(frozen=True)
class LayerCall:
    temb: torch.Tensor
    adaln: torch.Tensor
    rope: tuple
    curve: torch.Tensor
    attention: AttentionCall


class Recompute(torch.autograd.Function):
    @staticmethod
    def forward(ctx,runtime,index,call,hidden,*params):
        ctx.runtime,ctx.index,ctx.call=runtime,index,call
        ctx.params=params
        ctx.save_for_backward(hidden.detach().cpu())
        return runtime.layer(index,hidden,call)
    @staticmethod
    def backward(ctx,grad):
        runtime=ctx.runtime
        hidden=ctx.saved_tensors[0].to(runtime.device).requires_grad_(True)
        runtime.load_layer(ctx.index)
        try:
            with torch.enable_grad():
                out=runtime.layer(ctx.index,hidden,ctx.call,release=False)
                values=torch.autograd.grad(out,(hidden,*ctx.params),grad,allow_unused=True)
            return (None,None,None,*values)
        finally:
            runtime.release_layer(ctx.index)


class Runtime:
    def __init__(self,root,device='npu:0',rank=8,resident=0,log=lambda *a,**k:None):
        self.root=Path(root);self.device=torch.device(device);self.log=log
        self.model=MiniMaxH3Transformer3DModel.from_pretrained(str(self.root/'merged_transformer'),torch_dtype=torch.bfloat16,local_files_only=True).eval().requires_grad_(False)
        log('base_loaded_cpu',parameter_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters()))
        self.curves=CurveContext(self.root/'assets'/CURVE_TABLE);self.curves.strength=1.0
        curves=load_file(str(self.root/'assets'/CURVE_FILE))
        roots=sorted(k[len('transformer.'):-len('.curve_A.weight')] for k in curves if k.endswith('.curve_A.weight'))
        assert len(roots)==50
        for name in roots:
            parent,key=name.rsplit('.',1)
            a,b=[curves[f'transformer.{name}.curve_{v}.weight'].float() for v in ('A','B')]
            setattr(self.model.get_submodule(parent),key,CurveLinear(self.model.get_submodule(name),a,b,self.curves))
        del curves
        self.resident=resident;self.host={};self.processors=[];self.layer_params=[]
        for i,block in enumerate(self.model.transformer_blocks):
            for path in ('attn.to_q','attn.to_k','attn.to_v','attn.to_out.0'):
                parent,key=path.rsplit('.',1)
                setattr(block.get_submodule(parent),key,AdapterLinear(block.get_submodule(path),rank))
            processor=CausalAttention();block.attn.set_processor(processor);self.processors.append(processor)
            # Move trainable tensors exactly once, before optimizer creation.
            for module in block.modules():
                if isinstance(module,AdapterLinear):
                    module.a=nn.Parameter(module.a.to(self.device));module.b=nn.Parameter(module.b.to(self.device))
            self.layer_params.append(tuple(p for p in block.parameters() if p.requires_grad))
            self.host[i]=[(module,key,p.detach()) for module in block.modules() for key,p in module._parameters.items() if p is not None and not p.requires_grad]
            self.host[i]+=[(module,key,b.detach()) for module in block.modules() for key,b in module._buffers.items() if b is not None]
            if i<resident:self.load_layer(i)
        for name,child in self.model.named_children():
            if name!='transformer_blocks':child.to(self.device)
        self.condition=nn.Linear(96,self.model.config.hidden_size,bias=False,dtype=torch.float32,device=self.device)
        nn.init.zeros_(self.condition.weight)
        self.params=[p for group in self.layer_params for p in group]+list(self.condition.parameters())
        self.cache=None;self.next_frame=0;self.cache_version=None
        log('runtime_ready',trainable=sum(p.numel() for p in self.params),resident_layers=resident)

    def load_layer(self,i):
        for module,key,host in self.host[i]:
            if key in module._parameters:
                p=module._parameters[key]
                if p.device!=self.device:module._parameters[key]=nn.Parameter(host.to(self.device),requires_grad=False)
            elif module._buffers[key].device!=self.device:module._buffers[key]=host.to(self.device)

    def release_layer(self,i):
        if i<self.resident:return
        for module,key,host in self.host[i]:
            if key in module._parameters:module._parameters[key]=nn.Parameter(host,requires_grad=False)
            else:module._buffers[key]=host

    def layer(self,i,x,call,release=True,capture=False):
        self.load_layer(i)
        self.curves.coords=call.curve
        p=self.processors[i];p.call=call.attention;p.capture=capture;p.captured=None
        try:return self.model.transformer_blocks[i](x,call.temb,call.adaln,call.rope)
        finally:
            p.capture=False
            if release:self.release_layer(i)

    def time(self,values):
        times=torch.tensor(values,device=self.device,dtype=torch.float32)
        with torch.no_grad():
            temb=self.model.time_embedder(self.model.time_proj(times).float())
            self.curves.pre(None,(),{'timestep':times});coords=self.curves.coords
        return temb,coords

    def static_inputs(self,sample):
        # No text pilot is an explicit configuration, never a zero tensor prompt.
        audio=sample['audio_latents'].to(self.device)
        if audio.ndim==3:audio=audio[None]
        text=sample['prompt_embeds'].to(self.device)
        if text.ndim==2:text=text[None]
        with torch.no_grad():
            if text.shape[1]:
                embeds=self.model.token_refiner(self.model.context_embedder(text.bfloat16()))
            else:embeds=torch.empty((1,0,self.model.config.hidden_size),device=self.device,dtype=torch.bfloat16)
            a=self.model.audio_proj_in(pack_audio_latents(audio).float()).bfloat16()
        # Use upstream exact spatial/audio position convention. Dummy text used
        # only to construct geometry, then removed; no dummy text hidden state.
        tags=torch.ones(max(1,text.shape[1]),dtype=torch.long)
        layout=build_t2va_layout(tags,MiniMaxH3Geometry(1,68,120,audio.shape[-1]))
        n=tags.numel();positions=layout.position_ids[:n+a.shape[1]]
        if not text.shape[1]:
            positions=positions[n:].clone();positions[:,0]-=n
        self.text_count=text.shape[1]
        token_tags=torch.cat((torch.ones(text.shape[1],dtype=torch.long),torch.full((a.shape[1],),2,dtype=torch.long))).to(self.device)
        return torch.cat((embeds,a),1),positions.to(self.device),token_tags

    def video_positions(self,frames):
        # Read released position grid, replacing temporal axis by absolute frame.
        layout=build_t2va_layout(torch.ones(1,dtype=torch.long),MiniMaxH3Geometry(1,68,120,1))
        grid=layout.position_ids[layout.video_indices].to(self.device)
        values=[]
        for f in frames:
            p=grid.clone();p[:,0]=_temporal_position_grid(f+1,float(self.text_count))[-1];values.append(p)
        return torch.cat(values)

    def embeds(self,x,condition):
        rows=patchify_video_latents(x[None],self.model.config.patch_size).reshape(1,-1,96)
        cond=patchify_video_latents(condition[None],self.model.config.patch_size).reshape(1,-1,96)
        return (self.model.proj_in(rows.bfloat16().float()).bfloat16()+self.condition(cond.float()).bfloat16())

    def offline(self,x,condition,sample,frames,sigma,*,train=False,dense=False,return_full=False):
        if x.shape!=(24,len(frames),68,120) or condition.shape!=x.shape:raise ValueError('Exact geometry required')
        static,sp,tags=self.static_inputs(sample)
        vs=self.embeds(x.to(self.device),condition.to(self.device))
        h=torch.cat((static,vs),1)
        positions=torch.cat((sp,self.video_positions(frames)))
        rotary=self.model.rope(positions)
        time_indices=torch.zeros(h.shape[1],dtype=torch.long,device=self.device)
        # clean history, noisy current; text/audio known static clean t=1.
        time_indices[-2040:]=1
        temb,curve=self.time([1.,1.-sigma])
        modalities=torch.cat((tags,torch.zeros(vs.shape[1],dtype=torch.long,device=self.device)))
        adaln=time_indices*3+modalities
        frameids=(-1,)*static.shape[1]+tuple(f for f in frames for _ in range(2040))
        call=LayerCall(temb,adaln,rotary,curve,AttentionCall(frameids,dense=dense))
        for i in range(len(self.model.transformer_blocks)):
            h=Recompute.apply(self,i,call,h,*self.layer_params[i]) if train else self.layer(i,h,call)
        # Select current frame before output projection; output norm is rowwise.
        n=2040*len(frames) if return_full else 2040
        last=h[:,-n:]
        out=self.model.proj_out(self.model.norm_out(last,temb,time_indices[-n:]).float())
        return unpatchify_video_rows(out,MiniMaxH3Geometry(len(frames) if return_full else 1,68,120,1))[0].float()

    @torch.no_grad()
    def start_stream(self,sample):
        h,pos,tags=self.static_inputs(sample);temb,curve=self.time([1.]);rope=self.model.rope(pos)
        call=LayerCall(temb,tags,rope,curve,AttentionCall((-1,)*h.shape[1]))
        self.cache=[]
        for i in range(len(self.model.transformer_blocks)):
            h=self.layer(i,h,call,capture=True);self.cache.append(self.processors[i].captured)
        self.next_frame=0;self.cache_version=tuple(p._version for p in self.params)

    @torch.no_grad()
    def stream_velocity(self,x,condition,sigma,*,commit=False):
        if self.cache is None or self.cache_version!=tuple(p._version for p in self.params):raise ValueError('Missing/stale stream cache')
        if commit and sigma!=0:raise ValueError('Only clean history may be committed')
        if x.shape!=(24,1,68,120) or condition.shape!=x.shape:raise ValueError('One full spatial latent only')
        f=self.next_frame;h=self.embeds(x.to(self.device),condition.to(self.device))
        temb,curve=self.time([1.-sigma]);rope=self.model.rope(self.video_positions([f]))
        ids=torch.zeros(2040,dtype=torch.long,device=self.device)
        pending=[]
        for i in range(len(self.model.transformer_blocks)):
            call=LayerCall(temb,ids,rope,curve,AttentionCall((f,)*2040,past=self.cache[i]))
            h=self.layer(i,h,call,capture=commit)
            if commit:pending.append(append_cache(self.cache[i],self.processors[i].captured,next_frame=f+1))
        out=self.model.proj_out(self.model.norm_out(h,temb,ids).float())
        if commit:
            if sigma!=0:raise ValueError('Only clean history may be committed')
            self.cache=pending
            self.next_frame+=1
        return unpatchify_video_rows(out,MiniMaxH3Geometry(1,68,120,1))[0].float()

    def adapter_state(self):
        out={k:v.detach().cpu().clone() for k,v in self.model.named_parameters() if v.requires_grad}
        out['condition.weight']=self.condition.weight.detach().cpu().clone()
        return out

    def load_adapter(self,state):
        dest={k:v for k,v in self.model.named_parameters() if v.requires_grad};dest['condition.weight']=self.condition.weight
        assert set(dest)==set(state)
        with torch.no_grad():
            for k,v in dest.items():v.copy_(state[k].to(v))
        self.cache=None
