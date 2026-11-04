import json, subprocess
import torch
from .common import ROOT, dump
from .data import read_clip

def write_mp4(path, rgb, fps):
    frames=(rgb[0].detach().float().clamp(0,1).permute(1,2,3,0).cpu()*255).round().to(torch.uint8).numpy()
    t,h,w,c=frames.shape
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}',
         '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(path)]
    subprocess.run(cmd,input=frames.tobytes(),check=True)

@torch.no_grad()
def save_visuals(vaes, pred, item, step, index=0):
    gt,_=read_clip(item['record'],crop=item['crop'])
    recon=vaes.decode(item['wan']); predicted=vaes.decode(pred)
    assert gt.shape==recon.shape==predicted.shape
    out=ROOT/'visualizations'/f'step_{step:06d}'/f'sample_{index:02d}'; out.mkdir(parents=True,exist_ok=True)
    for name,rgb in [('gt',gt),('pred',predicted),('true_wan',recon)]: write_mp4(out/f'{name}.mp4',rgb,item['record']['fps'])
    # Side-by-side GT | prediction | true Wan; no temporal/spatial resampling.
    write_mp4(out/'comparison.mp4',torch.cat((gt,predicted.cpu(),recon.cpu()),dim=-1),item['record']['fps'])
    def changes(x):
        return (x[:,:,1:]-x[:,:,:-1]).abs().mean((0,1,3,4)).cpu().tolist()
    dump(out/'metadata.json',dict(record=item['record'],crop=item['crop'],boundary_frame_pair=[16,17],
                                per_transition_rgb_l1=dict(gt=changes(gt),pred=changes(predicted),true_wan=changes(recon))))
