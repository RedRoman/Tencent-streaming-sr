"""Ultra Flash Eq.3 loss family, explicitly adapted to H3's temporal grid.

Local choices (not an official training recipe): frozen OpenCV Farneback flow
on quarter-resolution GT RGB, L1 warp residual, mean reduction, in-bounds mask,
and no warp supervision on latent anchors created by VAE tail padding.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

FLOW_VERSION = 'h3-anchors17x5-farneback-quarter-backward-inbounds-v1'

def anchors(frames=37):
    return [17*(j//5)+4*(j%5) for j in range(frames)]

def flow_from_rgb(rgb, latent_hw, latent_frames=37):
    import cv2
    if isinstance(rgb, torch.Tensor): rgb=rgb.cpu().numpy()
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError('Expected uint8 [T,H,W,3] GT RGB')
    lh,lw=latent_hw; h,w=rgb.shape[1:3]; qh,qw=h//4,w//4
    times=anchors(latent_frames)
    grey={i:cv2.cvtColor(cv2.resize(rgb[i],(qw,qh),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY)
          for i in times if i<len(rgb)}
    flows=[]; valid_pairs=[]
    for previous,current in zip(times,times[1:]):
        valid=previous<len(rgb) and current<len(rgb)
        if valid:
            # Current coordinates -> previous coordinates; warp samples previous.
            value=cv2.calcOpticalFlowFarneback(grey[current],grey[previous],None,
                pyr_scale=.5,levels=4,winsize=21,iterations=5,poly_n=7,poly_sigma=1.5,flags=0)
            value=cv2.resize(value,(lw,lh),interpolation=cv2.INTER_LINEAR)
            value[...,0] *= lw/qw; value[...,1] *= lh/qh
        else: value=np.zeros((lh,lw,2),dtype=np.float32)
        flows.append(value);valid_pairs.append(valid)
    flow=torch.from_numpy(np.stack(flows)).permute(0,3,1,2).contiguous()
    return flow, torch.tensor(valid_pairs,dtype=torch.bool), {
        'version':FLOW_VERSION,'flow_direction':'current_to_previous',
        'latent_anchors':times,'real_source_frames':len(rgb),
        'valid_latent_frames':[i for i,t in enumerate(times) if t<len(rgb)],
        'valid_warp_pairs':valid_pairs,'source_flow_hw':[qh,qw],
        'latent_flow_hw':[lh,lw],'vector_scale_xy':[lw/qw,lh/qh],
        'anchor_meaning':'last causal encoder source-frame dependency, not a point-sampled RGB latent',
        'optical_flow':'OpenCV Farneback on fixed GT RGB; no flow or GT input to upscaler',
        'occlusion_mask':False,'spatial_mask':'inbounds only',
        'boundary_pairs':'included across 17-frame encoder blocks using actual anchor spacing',
    }

def warp_previous(previous, flow):
    if previous.ndim!=4 or flow.shape!=(previous.shape[0],2,*previous.shape[-2:]):
        raise ValueError('Expected previous [N,C,H,W], backward flow [N,2,H,W]')
    n,c,h,w=previous.shape
    yy,xx=torch.meshgrid(torch.arange(h,device=previous.device),torch.arange(w,device=previous.device),indexing='ij')
    xy=torch.stack((xx,yy),dim=-1).float()[None]+flow.permute(0,2,3,1).float()
    mask=((xy[...,0]>=0)&(xy[...,0]<=w-1)&(xy[...,1]>=0)&(xy[...,1]<=h-1)).unsqueeze(1)
    grid=torch.stack((2*xy[...,0]/max(w-1,1)-1,2*xy[...,1]/max(h-1,1)-1),dim=-1)
    return F.grid_sample(previous.float(),grid,mode='bilinear',padding_mode='border',align_corners=True),mask

def losses(pred,target,flow,valid_pairs,warp_weight=.1,valid_frames=36):
    if pred.shape!=target.shape or pred.ndim!=5: raise ValueError('Expected equal B,C,T,H,W')
    b,c,t,h,w=pred.shape
    if flow.ndim==4:flow=flow[None]
    if valid_pairs.ndim==1:valid_pairs=valid_pairs[None]
    if flow.shape!=(b,t-1,2,h,w) or valid_pairs.shape!=(b,t-1):raise ValueError('Invalid flow geometry')
    if not 1<valid_frames<=t:raise ValueError('Invalid real-frame latent count')
    mse=(pred[:,:,:valid_frames].float()-target[:,:,:valid_frames].float()).square().mean()
    previous=pred[:,:,:-1].permute(0,2,1,3,4).reshape(b*(t-1),c,h,w)
    current=pred[:,:,1:].permute(0,2,1,3,4).reshape(b*(t-1),c,h,w)
    warped,inbounds=warp_previous(previous,flow.reshape(b*(t-1),2,h,w))
    mask=inbounds & valid_pairs.reshape(-1,1,1,1)
    count=mask.float().sum((1,2,3))*c
    per_pair=((current.float()-warped).abs()*mask).sum((1,2,3))/count.clamp_min(1)
    has_pixels=count>0
    ewarp=per_pair[has_pixels].mean() if has_pixels.any() else pred.sum()*0
    return {'loss':mse+warp_weight*ewarp,'latent_mse':mse,'ewarp_l1':ewarp,
            'valid_warp_fraction':mask.float().sum()/(valid_pairs.float().sum()*h*w).clamp_min(1)}

def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda:handle.read(8*1024*1024),b''):digest.update(part)
    return digest.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--num-shards',type=int,default=1)
    args=p.parse_args();torch.set_num_threads(2)
    output=args.pairs/'flow';output.mkdir(exist_ok=True)
    samples=sorted((args.pairs/'samples').glob('*.pt'))
    records=[]
    for i,path in enumerate(samples):
        if i%args.num_shards!=args.shard_index:continue
        target=output/path.name;digest=file_sha(path)
        if target.exists():
            saved=torch.load(target,map_location='cpu',weights_only=False)
            if saved['pair_sha256']!=digest or saved['metadata']['version']!=FLOW_VERSION:raise ValueError('Stale flow cache')
            continue
        pair=torch.load(path,map_location='cpu',weights_only=False)
        flow,valid,meta=flow_from_rgb(pair['source_pixels'],pair['reference'].shape[-2:],pair['reference'].shape[1])
        assert len(meta['valid_latent_frames'])==36 and sum(meta['valid_warp_pairs'])==35
        result={'flow':flow,'valid_pairs':valid,'metadata':meta,'pair_sha256':digest}
        tmp=target.with_suffix('.tmp');torch.save(result,tmp);tmp.replace(target)
        row={'sample_id':path.stem,'flow_abs_mean':float(flow.abs().mean()),'flow_abs_max':float(flow.abs().max()),'valid_pairs':int(valid.sum())}
        records.append(row);print(json.dumps(row),flush=True)
    (output/f'summary_shard{args.shard_index}.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
