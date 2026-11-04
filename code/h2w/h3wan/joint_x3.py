"""Round 3: normalized H3 -> normalized Wan -> official UltraFlash 3x.

Loss family: UltraFlash paper Eq.3. Unreleased recipe choices are explicit:
mean squared latent reconstruction + 0.1 * mean absolute backward-flow warp.
Flow: fixed OpenCV Farneback on GT RGB at quarter resolution, then /2 to
Wan latent resolution. Independent main/tail segments; never warp across reset.
"""
import importlib.util
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from .common import ROOT
from .model import Mapper

RUN = ROOT/'joint_x3_round3'
UPSAMPLER = ROOT/'assets/ultraflash/2K-ultraLatentUpsampler-step_28100.pth'
MAPPER = ROOT/'pixel_lpips_round2/checkpoints/pixel_train_step_010000.pt'
HR_H, HR_W = 672, 1152
ANCHORS = [0,4,8,12,16,17,21]
PAIRS = [(0,1),(1,2),(2,3),(3,4),(5,6)]

def spatial_resize(x, height, width):
    """Only resize 4D batches of RGB/latent frames, never the temporal axis."""
    b,c,t,h,w=x.shape
    y=F.interpolate(x.permute(0,2,1,3,4).reshape(b*t,c,h,w).float(),
                    size=(height,width),mode='bicubic',align_corners=False,antialias=True)
    return y.reshape(b,t,c,height,width).permute(0,2,1,3,4).contiguous()

class JointX3(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        src=Path(__file__).resolve().parents[1]/'vendor/UltraFlash/inference/sr/latent_upsampler/ultra_latent_up_v1.py'
        spec=importlib.util.spec_from_file_location('official_ultraflash_up_v1',src)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        self.mapper=Mapper()
        self.upsampler=mod.UltraLatentUpsampler(in_channels=16,out_channels=16,
            mid_channels=128,num_blocks=8,upsample_scale=3,activation='relu',
            residual=False,memory_init='replicate',zero_init_final=False,default_parallel=True)
        if pretrained:
            self.mapper.load_state_dict(torch.load(MAPPER,map_location='cpu',weights_only=False)['model'],strict=True)
            self.upsampler.load_state_dict(torch.load(UPSAMPLER,map_location='cpu',weights_only=False)['model'],strict=True)

    def forward(self,x):
        low=self.mapper(x)
        # Fresh causal memory for independently encoded main and tail.
        high=torch.cat([self.upsampler(low[:,:,:5],parallel=True),
                        self.upsampler(low[:,:,5:],parallel=True)],dim=2)
        assert high.shape[:3]==(x.shape[0],16,7)
        assert high.shape[-2:]==(x.shape[-2]*6,x.shape[-1]*6)
        return high

def flow_from_rgb(rgb):
    frames=(rgb[0].permute(1,2,3,0).numpy()*255).round().astype(np.uint8)
    h,w=frames.shape[1:3];lh,lw=h//8,w//8
    grey={i:cv2.cvtColor(cv2.resize(frames[i],(w//4,h//4),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY) for i in ANCHORS}
    flows=[]
    for prev,cur in PAIRS:
        # Backward flow at current coordinates: current RGB -> previous RGB.
        flow=cv2.calcOpticalFlowFarneback(grey[ANCHORS[cur]],grey[ANCHORS[prev]],None,
            pyr_scale=.5,levels=4,winsize=21,iterations=5,poly_n=7,poly_sigma=1.5,flags=0)
        flow=cv2.resize(flow,(lw,lh),interpolation=cv2.INTER_LINEAR)*.5
        flows.append(flow)
    return torch.from_numpy(np.stack(flows)).permute(0,3,1,2).contiguous()

def warp_previous(previous, flow):
    n,c,h,w=previous.shape
    yy,xx=torch.meshgrid(torch.arange(h,device=previous.device),torch.arange(w,device=previous.device),indexing='ij')
    xy=torch.stack((xx,yy),dim=-1).float()[None]+flow.permute(0,2,3,1).float()
    mask=((xy[...,0]>=0)&(xy[...,0]<=w-1)&(xy[...,1]>=0)&(xy[...,1]<=h-1)).float().unsqueeze(1)
    grid=torch.stack((2*xy[...,0]/(w-1)-1,2*xy[...,1]/(h-1)-1),dim=-1)
    return F.grid_sample(previous.float(),grid,mode='bilinear',padding_mode='border',align_corners=True),mask

def losses(pred,target,flow,warp_weight=.1):
    pred,target=pred.float(),target.float()
    mse_main=F.mse_loss(pred[:,:,:5],target[:,:,:5]);mse_tail=F.mse_loss(pred[:,:,5:],target[:,:,5:])
    # Flatten B and the five valid temporal pairs for spatial-only warping.
    b,c,_,h,w=pred.shape
    prev=pred[:,:,[p[0] for p in PAIRS]].permute(0,2,1,3,4).reshape(b*5,c,h,w)
    cur=pred[:,:,[p[1] for p in PAIRS]].permute(0,2,1,3,4).reshape(b*5,c,h,w)
    warped,mask=warp_previous(prev,flow.reshape(b*5,2,h,w))
    per_pair=((cur-warped).abs()*mask).sum((1,2,3))/(mask.sum((1,2,3))*c).clamp_min(1)
    per_pair=per_pair.reshape(b,5)
    ew_main=per_pair[:,:4].mean();ew_tail=per_pair[:,4:].mean()
    lm=mse_main+warp_weight*ew_main;lt=mse_tail+warp_weight*ew_tail
    mse=F.mse_loss(pred,target)
    return dict(loss_total=(lm+lt)/2,loss_main=lm,loss_tail=lt,
        mse_balanced=(mse_main+mse_tail)/2,mse_main=mse_main,mse_tail=mse_tail,
        ewarp=(ew_main+ew_tail)/2,ewarp_main=ew_main,ewarp_tail=ew_tail,
        latent_mse=mse,latent_l1=F.l1_loss(pred,target),latent_psnr_like=-10*torch.log10(mse.clamp_min(1e-12)),
        cosine=F.cosine_similarity(pred.flatten(1),target.flatten(1)).mean())
