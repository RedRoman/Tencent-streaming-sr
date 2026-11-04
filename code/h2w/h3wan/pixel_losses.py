"""Differentiable frozen UltraFlash decoder and calibrated LPIPS-v0.1/VGG."""
import importlib.util
from pathlib import Path
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import lpips
from .common import ROOT

RUN = ROOT / 'pixel_lpips_round2'
SOURCE = Path(__file__).resolve().parents[1] / 'vendor/UltraFlash/inference/sr/tiny_decoder.py'

class PixelLoss(nn.Module):
    def __init__(self, device, frame_batch=2):
        super().__init__()
        spec=importlib.util.spec_from_file_location('ultraflash_tiny_decoder',SOURCE)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.tiny=module.WanTinyDecoder(str(ROOT/'assets/ultraflash/tiny_decoder.pth'),dtype=torch.bfloat16).to(device)
        self.perceptual=lpips.LPIPS(net='vgg',version='0.1',pretrained=True,pnet_rand=False,verbose=True).to(device)
        self.eval().requires_grad_(False)
        self.frame_batch=frame_batch
        print('UltraFlash tiny decoder and LPIPS-VGG frozen; differentiable input path enabled',flush=True)

    def _decode_part(self,z):
        # Official decoder consumes the normalized Wan generator representation directly.
        # A fresh decode_video invocation creates zero memory; native TGrow + one trim only.
        with torch.autocast('npu',dtype=torch.bfloat16):
            return (self.tiny.decode_latents(z,parallel=True)+1)/2

    def decode(self,z,checkpointed=True):
        assert z.shape[1:3]==(16,7)
        pieces=[]
        for part,expected in ((z[:,:,:5],17),(z[:,:,5:],5)):
            if checkpointed and torch.is_grad_enabled() and part.requires_grad:
                rgb=checkpoint(self._decode_part,part,use_reentrant=False)
            else: rgb=self._decode_part(part)
            assert rgb.shape[2]==expected,rgb.shape
            pieces.append(rgb)
        rgb=torch.cat(pieces,dim=2)
        assert rgb.shape[2]==22
        return rgb

    def _lpips_frames(self,pred,gt):
        # LPIPS's calibrated VGG receives RGB in [-1,1], evaluated in FP32.
        with torch.autocast('npu',enabled=False):
            return self.perceptual(pred.float().clamp(0,1)*2-1,gt.float()*2-1).flatten()

    def losses(self,rgb,gt):
        assert rgb.shape==gt.shape and rgb.shape[2]==22,(rgb.shape,gt.shape)
        result={}
        for name,sl in [('main',slice(0,17)),('tail',slice(17,22))]:
            p=rgb[:,:,sl]; t=gt[:,:,sl]
            result[f'pixel_l1_{name}']=(p.float()-t.float()).abs().mean()
            p=p.permute(0,2,1,3,4).flatten(0,1)
            t=t.permute(0,2,1,3,4).flatten(0,1)
            values=[]
            for start in range(0,len(p),self.frame_batch):
                a,b=p[start:start+self.frame_batch],t[start:start+self.frame_batch]
                if torch.is_grad_enabled() and a.requires_grad:
                    values.append(checkpoint(self._lpips_frames,a,b,use_reentrant=False))
                else: values.append(self._lpips_frames(a,b))
            result[f'lpips_{name}']=torch.cat(values).mean()
        result['pixel_l1_total']=result['pixel_l1_main']+result['pixel_l1_tail']
        result['lpips_total']=result['lpips_main']+result['lpips_tail']
        result['pixel_l1_frame_mean']=(rgb.float()-gt.float()).abs().mean()
        result['lpips_frame_mean']=(17*result['lpips_main']+5*result['lpips_tail'])/22
        return result
