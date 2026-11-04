"""Round 5: three independently encoded chunks, direct GT, one previous chunk per layer."""
import math,contextlib,random
import numpy as np
import cv2
import torch
from torch.utils.checkpoint import checkpoint
from .common import ROOT
from .sr_distill import load_sr,projected_context,MAPPER_CKPT
from .npu_attention import exact_block_attention,sinusoidal_embedding_1d
from .vaes import VAEs
from .joint_x3 import JointX3,spatial_resize
from .pixel_losses import PixelLoss
RUN=ROOT/'sr_gt_3chunks_round5'
NCHUNKS=3

def temporal_mask(*,num_query_frames,num_key_frames,spatial_blocks,device,chunk_size=5,key_prefix_frames=0,temporal_block_size=5,**kw):
    assert chunk_size==temporal_block_size==5
    q=torch.arange(num_query_frames//5,device=device)+key_prefix_frames//5
    k=torch.arange(num_key_frames//5,device=device)
    m=(k[None]<=q[:,None]) & (k[None]>=q[:,None]-1)
    return m[:,None,:,None].expand(-1,spatial_blocks,-1,spatial_blocks).reshape(len(q)*spatial_blocks,len(k)*spatial_blocks)

def windowed_attention(q,k,v,mask,num_heads):
    # Each temporal block contains a whole 5-frame chunk. Slice out guaranteed
    # masked keys before fused attention; no detach, full gradient across chunks.
    assert len(q)==len(k) and len(q)%NCHUNKS==0
    n=len(q)//NCHUNKS;out=[]
    for i in range(NCHUNKS):
        qs,qe=i*n,(i+1)*n;ks,ke=max(0,i-1)*n,(i+1)*n
        out.append(exact_block_attention(q[qs:qe],k[ks:ke],v[ks:ke],mask[:,:,qs:qe,ks:ke],num_heads))
    return torch.cat(out,1)

class ChunkStudent(torch.nn.Module):
    def __init__(self,device,offload=True):
        super().__init__();self.dit=load_sr(device,student=True);self.offload=offload
        from vendor.ultraflash_npu import attention
        attention.build_chunked_temporal_causal_mask=temporal_mask
        attention._block_sparse_attn_fallback=windowed_attention
        for block in self.dit.blocks:
            att=block.self_attn;att.sparse_block_size=(5,8,8);att.sparse_causal=True;att.stream_chunk_size=5
    def forward(self,noise,condition,context):
        m=self.dit;assert noise.shape==condition.shape and noise.shape[:3]==(1,16,5*NCHUNKS)
        x=m.patch_embedding(torch.cat((noise,condition),1))
        grid=torch.tensor([x.shape[2:]],dtype=torch.long);m._ensure_rope_freqs(grid,temporal_offset=0,device=x.device)
        x=x.flatten(2).transpose(1,2);seq=torch.tensor([x.shape[1]])
        with torch.autocast('npu',enabled=False):
            e=m.time_embedding(sinusoidal_embedding_1d(m.freq_dim,torch.full((1,),1000.,device=x.device)))
            e0=m.time_projection(e).unflatten(1,(6,m.dim))
        ctx=projected_context(m,context);m.clear_cross_kv()
        _,h,w=grid[0].tolist();nb=math.ceil(h/8)*math.ceil(w/8)
        topk=min(max(int(nb*nb*1560/(h*w)),1),max(2*nb*nb-1,1))
        kw=dict(e=e0,seq_lens=seq,grid_sizes=grid,freqs=m.freqs,context=ctx,context_lens=None,
                top_k=topk,local_num=None,local_range=9,temporal_offset=0,train_img=False)
        for block in m.blocks:
            if self.training and torch.is_grad_enabled():
                # Offload only checkpoint boundary tensors, retaining exact BPTT.
                scope=torch.autograd.graph.save_on_cpu(pin_memory=True,device_type='npu') if self.offload else contextlib.nullcontext()
                with scope:x=checkpoint(block,x,use_reentrant=False,**kw)
            else:x=block(x,**kw)
        with torch.autocast('npu',enabled=False):out=m.head(x.float(),e)
        return noise-torch.stack(m.unpatchify(out,grid)).to(noise.dtype)

def read_sequence(record,rng,crop=None):
    h,w=672,1152
    if crop is None:crop=dict(start=rng.randint(0,record['num_frames']-17*NCHUNKS),top=rng.randint(0,record['height']-h),left=rng.randint(0,record['width']-w),height=h,width=w)
    cap=cv2.VideoCapture(record['video_path']);cap.set(cv2.CAP_PROP_POS_FRAMES,crop['start']);frames=[]
    for _ in range(17*NCHUNKS):
        ok,x=cap.read();assert ok and x.shape==(record['height'],record['width'],3)
        y,l=crop['top'],crop['left'];frames.append(cv2.cvtColor(x[y:y+h,l:l+w],cv2.COLOR_BGR2RGB))
    cap.release();return torch.from_numpy(np.stack(frames)).permute(3,0,1,2).unsqueeze(0),crop

class ChunkData:
    def __init__(self,device):
        self.device=device;self.vaes=VAEs(device,h3_token_drop=0)
        self.joint=JointX3(pretrained=False)
        self.joint.load_state_dict(torch.load(MAPPER_CKPT,map_location='cpu',weights_only=False)['model'],strict=True)
        self.joint.to(device).eval().requires_grad_(False);self.pixels=PixelLoss(device);self.text={}
    @torch.no_grad()
    def make(self,record,seed,crop=None):
        rng=random.Random(seed);rgb,crop=read_sequence(record,rng,crop);d=self.device
        torch.npu.manual_seed_all(seed)
        context=torch.load(ROOT/'sr_distill_round4/text'/f'{record["id"]}.pt',map_location='cpu',weights_only=False)
        assert context['context'].shape[0]>0
        conds=[];targets=[];geometry=[]
        for i in range(NCHUNKS):
            hr=rgb[:,:,17*i:17*(i+1)].to(d).float()/255
            lr=spatial_resize(hr,224,384).clamp(0,1)
            pix=self.vaes.h3.transform(lr.permute(0,2,1,3,4)).permute(0,2,1,3,4).bfloat16()
            with torch.autocast('npu',dtype=torch.bfloat16):hz=self.vaes.h3.encode_base(pix,process_image=False)
            hz=(hz.float()-self.vaes.hm)/self.vaes.hs
            wz=self.vaes.wan_encode(hr)
            with torch.autocast('npu',dtype=torch.bfloat16):cond=self.joint.upsampler(self.joint.mapper(hz),parallel=True)
            assert hz.shape==(1,24,5,14,24) and wz.shape==cond.shape==(1,16,5,84,144),(hz.shape,wz.shape,cond.shape)
            conds.append(cond.bfloat16());targets.append(wz.bfloat16())
            geometry.append(dict(chunk=i,original_frames=[crop['start']+17*i,crop['start']+17*i+16],rgb=list(hr.shape),lr=list(lr.shape),h3=list(hz.shape),wan=list(wz.shape)))
        condition=torch.cat(conds,2);target=torch.cat(targets,2)
        noise=torch.randn(condition.shape,generator=torch.Generator().manual_seed(seed+1000000)).bfloat16().to(d)
        return dict(condition=condition,target=target,noise=noise,rgb=rgb,context=context['context'].to(d),prompt=context.get('prompt'),record=record,crop=crop,seed=seed,geometry=geometry)

    def loss_and_gradient(self,pred,item,backward=True,lpips_weight=.25):
        """Exact sum of three chunk losses; release each frozen decoder graph early.
        Compute dL/d(pred) chunkwise then backprop once through the complete DiT.
        This does NOT truncate gradients through previous-chunk attention.
        """
        stats={};grads=[];visual=[]
        for i in range(NCHUNKS):
            z=pred[:,:,i*5:(i+1)*5].detach().requires_grad_(backward)
            gt=item['rgb'][:,:,i*17:(i+1)*17].to(pred.device).float()/255
            with torch.set_grad_enabled(backward):
                rgb=checkpoint(self.pixels._decode_part,z,use_reentrant=False) if backward else self.pixels._decode_part(z)
                assert rgb.shape==gt.shape and rgb.shape[2]==17
                mse=(z.float()-item['target'][:,:,i*5:(i+1)*5].float()).square().mean()
                l1=(rgb.float()-gt).abs().mean()
                p=rgb[0].permute(1,0,2,3);t=gt[0].permute(1,0,2,3);ls=[]
                for f in range(0,17,2):
                    args=(p[f:f+2],t[f:f+2])
                    ls.append(checkpoint(self.pixels._lpips_frames,*args,use_reentrant=False) if backward else self.pixels._lpips_frames(*args))
                lp=torch.cat(ls).mean();loss=mse+l1+lpips_weight*lp
                if backward:(loss/NCHUNKS).backward();assert z.grad is not None and torch.isfinite(z.grad).all();grads.append(z.grad.detach())
            values=dict(loss_total=loss.item(),latent_mse=mse.item(),pixel_l1=l1.item(),lpips=lp.item())
            for k,v in values.items():stats[k]=stats.get(k,0)+v/NCHUNKS;stats[f'chunk_{i}/{k}']=v
            if not backward:visual.append(rgb.detach().cpu())
            del z,rgb,gt,p,t,ls,loss,mse,l1,lp
        if backward:pred.backward(torch.cat(grads,2))
        return stats,None if backward else torch.cat(visual,2)
