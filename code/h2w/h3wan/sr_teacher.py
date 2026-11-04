"""Online original UltraFlash teacher; fresh shared noise on every train step."""
import copy,time,random
from pathlib import Path
import torch
from .common import ROOT
from .joint_x3 import JointX3,spatial_resize
from .vaes import VAEs
from .pixel_losses import PixelLoss
from .sr_distill import RUN,MAPPER_CKPT,load_sr,forward_sr
from scripts.cache_sr_teacher import read29

class Teacher:
    def __init__(self,device):
        self.device=device;self.dit=load_sr(device)
        self.joint=JointX3()
        self.up=copy.deepcopy(self.joint.upsampler).to(device,dtype=torch.bfloat16).eval().requires_grad_(False)
        ck=torch.load(MAPPER_CKPT,map_location='cpu',weights_only=False)
        self.joint.load_state_dict(ck['model'],strict=True);del ck
        self.joint=self.joint.to(device).eval().requires_grad_(False)
        # Cached official H3 latents are reused; no H3 model/T5 resides on train NPUs.
        self.vaes=VAEs(device,h3=False);self.pixels=PixelLoss(device)
        self.text={}

    def base(self,record,split,rng=None,fixed=False):
        paths=sorted((ROOT/'joint_x3_round3/latents'/split/record['id']).glob('*.pt'))
        assert len(paths)>=8
        if rng is not None and not fixed:rng.shuffle(paths)
        # Use actual extra original frames; never pad repeated RGB at video end.
        for path in paths:
            x=torch.load(path,map_location='cpu',weights_only=False)
            if x['crop']['start']+29<=record['num_frames']:
                x['source_pair']=str(path);return x
        raise RuntimeError('No cached crop has 29 consecutive real frames: '+record['id'])

    @torch.no_grad()
    def make(self,base,noise_seed):
        d=self.device;record=base['record'];rgb,crop=read29(record,base['crop'])
        assert crop==base['crop']
        low=spatial_resize(rgb,rgb.shape[-2]//3,rgb.shape[-1]//3).clamp(0,1)
        if record['id'] not in self.text:
            self.text[record['id']]=torch.load(RUN/'text'/f'{record["id"]}.pt',map_location='cpu',weights_only=False)
        context=self.text[record['id']]['context'].to(d)
        with torch.autocast('npu',dtype=torch.bfloat16):condition=self.joint(base['h3'].to(d))
        lr=self.vaes.wan_encode(low.to(d));assert lr.shape[2]==8
        gen=torch.Generator(device='cpu').manual_seed(noise_seed)
        noise=torch.randn(1,16,8,*condition.shape[-2:],generator=gen).bfloat16().to(d)
        up_cache=sr_cache=None;parts=[]
        for start in range(0,8,2):
            with torch.autocast('npu',dtype=torch.bfloat16):
                cond,up_cache=self.up(lr[:,:,start:start+2].bfloat16(),parallel=False,caches=up_cache,return_caches=True)
                clean,sr_cache=forward_sr(self.dit,noise[:,:,start:start+2],cond,context,stream=True,offset=start,caches=sr_cache)
            parts.append(clean)
        z=torch.cat(parts,2)
        teacher_rgb=self.pixels._decode_part(z).float().clamp(0,1)
        assert teacher_rgb.shape[2]==29
        teacher_rgb=teacher_rgb[:,:,:22].contiguous()
        main=self.vaes.wan_encode(teacher_rgb[:,:,:17]);tail=self.vaes.wan_encode(teacher_rgb[:,:,17:])
        target=torch.cat((main,tail),2)
        assert condition.shape==target.shape==noise[:,:,:7].shape
        assert main.shape[2]==5 and tail.shape[2]==2
        assert all(torch.isfinite(x).all() for x in (target,teacher_rgb,condition,z))
        return dict(condition=condition.detach(),noise=noise[:,:,:7].contiguous(),target=target.detach(),
            teacher_rgb=teacher_rgb.detach(),context=context,record=record,crop=crop,noise_seed=noise_seed,
            teacher_noise8_shape=list(noise.shape),source_pair=base['source_pair'],
            noise_exact_prefix_equal=torch.equal(noise[:,:,:7],noise[:,:,:7].contiguous()))

def objective(pred,item,pixels,lpips_weight=.25):
    p,t=pred.float(),item['target'].float()
    main=(p[:,:,:5]-t[:,:,:5]).square().mean();tail=(p[:,:,5:]-t[:,:,5:]).square().mean()
    rgb=pixels.decode(pred)
    m=pixels.losses(rgb,item['teacher_rgb'])
    m.update(latent_mse_main=main,latent_mse_tail=tail,latent_mse=(main+tail)/2,
        latent_mse_frame_mean=(p-t).square().mean())
    m['loss_main']=main+m['pixel_l1_main']+lpips_weight*m['lpips_main']
    m['loss_tail']=tail+m['pixel_l1_tail']+lpips_weight*m['lpips_tail']
    m['loss_total']=(m['loss_main']+m['loss_tail'])/2
    return m,rgb
