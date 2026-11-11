"""Decode identical complete step1000 latents with released causal ViT24 and H3 VAE."""
import argparse,contextlib,gc,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/opt/refiner-cache/h3-pretrained-causal-refiner-20261109')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'decoder_source/vendor'),str(HERE/'decoder_source/native'),str(ROOT/'media_step500_v1')]
import torch
import torch_npu
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from safetensors.torch import load_file
from causal_vit_decoder_ascend_v2 import CausalViTDecoder
from data_utils import load_video_vae,decode_latent

WEIGHT_SHA='a1455dd6fedb61b1b835c3e1d2e5bad4a132c02a45585f992372ca2d25e22889'
BASE='/opt/runtime-temp/h3_training_assets_20260921/VDN-MiniMax-H3/h3-base'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)

class Writer:
    def __init__(self,path,w=1920,h=1080):
        self.path=path;self.frames=0
        self.p=subprocess.Popen(['/usr/local/ffmpeg/bin/ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n',
            '-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}','-framerate','24','-i','pipe:0',
            '-an','-c:v','mpeg4','-q:v','2','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    def push(self,rgb):
        a=np.ascontiguousarray(rgb.numpy() if isinstance(rgb,torch.Tensor) else rgb,dtype=np.uint8)
        self.p.stdin.write(a.tobytes());self.frames+=len(a)
    def finish(self):
        self.p.stdin.close();assert self.p.wait()==0
        assert self.frames==226,self.frames
        probe=json.loads(subprocess.check_output(['/usr/local/ffmpeg/bin/ffprobe','-v','error','-count_frames',
            '-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,avg_frame_rate','-of','json',str(self.path)]))
        assert int(probe['streams'][0]['nb_read_frames'])==226
        return dict(path=str(self.path),sha256=sha(self.path),bytes=self.path.stat().st_size,probe=probe)

def load_decoder(device):
    cfg=json.loads((HERE/'decoder_source/decoder_config.json').read_text())
    basecfg=json.loads((Path(BASE)/'vae/config.json').read_text())
    assert all(cfg[k]==basecfg[k] for k in ('latents_mean','latents_std'))
    weight=ROOT/'weights/streamrender_vit24/causal_vit24_step6000.safetensors'
    assert sha(weight)==WEIGHT_SHA
    model=CausalViTDecoder.from_config(cfg,chunk=2,cache_latents=4)
    state=load_file(str(weight),device='cpu')
    loaded=model.load_state_dict(state,strict=True,assign=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    assert all(not p.is_meta for p in model.parameters())
    model=model.float().eval().requires_grad_(False)
    return model.to(device)

def compare(a,b):
    e=(a.float()-b.float()).abs()
    return dict(max_abs=float(e.max()),mean_abs=float(e.mean()),reference_rms=float(a.float().square().mean().sqrt()))

@torch.no_grad()
def validate_decoder(model,device,out,log):
    # Real pretrained weights on an odd spatial grid exercise shifted-window masks,
    # a partially filled history and eviction beyond four cached latents.
    from causal_vit_decoder import CausalViTDecoder as Reference
    z=torch.randn(1,8,24,9,11,generator=torch.Generator().manual_seed(343)).float()*.5
    model.cpu();sr=model.init_state();sp=model.init_state()
    ref=[];port=[]
    for i in range(0,8,2):
        ref.append(Reference.decode_chunk(model,z[:,i:i+2],sr))
        port.append(model.decode_chunk(z[:,i:i+2],sp))
    reference=torch.cat(ref,2);cpu_port=torch.cat(port,2)
    assert torch.equal(reference,cpu_port)
    log('cpu_reference_passed',exact_equal=True,frames=reference.shape[2])
    model.to(device);s=model.init_state();fp=[]
    for i in range(0,8,2):fp.append(model.decode_chunk(z[:,i:i+2].to(device),s).cpu())
    fp=torch.cat(fp,2);fp_metric=compare(reference,fp)
    assert fp_metric['mean_abs']<.003 and fp_metric['max_abs']<.05,fp_metric
    s=model.init_state();bf=[]
    with torch.autocast('npu',dtype=torch.bfloat16):
        for i in range(0,8,2):bf.append(model.decode_chunk(z[:,i:i+2].to(device),s).cpu())
    bf=torch.cat(bf,2);bf_metric=compare(reference,bf)
    assert bf_metric['mean_abs']<.08 and bf_metric['max_abs']<1.0,bf_metric
    assert all(x.shape[1]==4 for x in s['k']+s['v'])
    # Fresh streams with an altered future must produce identical first two pairs.
    z2=z.clone();z2[:,4:]+=4
    a=model.init_state();b=model.init_state()
    with torch.autocast('npu',dtype=torch.bfloat16):
        for i in (0,2):
            aa=model.decode_chunk(z[:,i:i+2].to(device),a)
            bb=model.decode_chunk(z2[:,i:i+2].to(device),b)
            assert torch.equal(aa,bb)
    receipt=dict(status='passed',weights_sha256=WEIGHT_SHA,strict_load=True,cpu_reference_exact=True,
        npu_fp32_vs_cpu=fp_metric,npu_bf16_vs_cpu_fp32=bf_metric,cache_latents=4,
        future_pair_isolation=True,test_grid=[9,11],test_latents=8,optimizer_updates=0)
    write(out/'decoder_validation.json',receipt);log('decoder_validation_passed',result=receipt)
    del sr,sp,s,a,b;gc.collect();torch.npu.empty_cache()

@torch.no_grad()
def decode_causal(model,z,out,name,device,log):
    state=model.init_state();pending=[];writer=Writer(out/(name+'_causal1080p.mp4'))
    pm=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1,1)
    ps=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1,1)
    previews=[];started=time.time();records=[]
    def pair(part,keep=None):
        t=state['t']
        with torch.autocast('npu',dtype=torch.bfloat16):pixels=model.decode_chunk(part.to(device),state)
        assert torch.isfinite(pixels).all()
        if keep is not None:pixels=pixels[:,:,:keep]
        rgb=((pixels.float()*ps+pm).clamp(0,1)*255).round().to(torch.uint8)
        rgb=rgb[0,:,:,4:1084,:].permute(1,2,3,0).cpu().contiguous()
        writer.push(rgb)
        for frame in rgb:
            previews.append(np.array(Image.fromarray(frame.numpy()).resize((640,360),Image.Resampling.LANCZOS)))
        records.append(dict(latent_start=t,latent_end=min(t+2,67),frames=len(rgb),total_frames=writer.frames,
            cached_latents=max(x.shape[1] for x in state['k']),tail_replicated=keep is not None))
        log('causal_pair_complete',arm=name,**records[-1],seconds=time.time()-started)
    # Replay actual DiT arrival chunks (5+5+...+2), buffer into decoder pairs.
    for start in range(0,67,5):
        for i in range(start,min(start+5,67)):pending.append(z[:,i:i+1].permute(1,0,2,3)[None])
        while len(pending)>=2:
            p=torch.cat(pending[:2],1);del pending[:2];pair(p)
    assert len(pending)==1 and state['t']==66
    own=4-(3 if state['t']%5==0 else 0)
    pair(torch.cat([pending[0],pending[0]],1),keep=own)
    receipt=writer.finish();receipt.update(seconds=time.time()-started,decoder='StreamRender pretrained causal ViT24',
        precision='BF16 autocast with FP32 norms, residual scales, embedder and projection',pair_records=records)
    np.save(out/(name+'_causal_preview.npy'),np.stack(previews))
    write(out/(name+'_causal_result.json'),receipt)
    del state,pending;gc.collect();torch.npu.empty_cache()
    return receipt

def comparison(out,names,filename,labels):
    arrays=[np.load(out/(n+'_preview.npy'),mmap_mode='r') for n in names]
    assert all(len(a)==226 for a in arrays)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
    writer=Writer(out/filename,w=640*len(names),h=410)
    sheet=Image.new('RGB',(640*len(names),410*3))
    for i in range(226):
        canvas=Image.new('RGB',(640*len(names),410),(16,20,28));d=ImageDraw.Draw(canvas)
        for j,(a,label) in enumerate(zip(arrays,labels)):
            canvas.paste(Image.fromarray(a[i]),(640*j,31));d.text((640*j+8,4),label,font=font,fill='white')
        d.text((8,390),f'Charge train | step1000 | {i+1}/226 frames | 24fps | full 67 latents',font=font,fill=(205,210,220))
        writer.push(np.asarray(canvas)[None])
        if i in (0,112,225):sheet.paste(canvas,(0,(0 if i==0 else 1 if i==112 else 2)*410))
    sheet.save(out/(Path(filename).stem+'_contactsheet.jpg'),quality=94)
    return writer.finish()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prediction',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--mode',choices=['causal','original'],required=True);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    def log(event,**kw):
        row=dict(event=event,time=time.time(),**kw);print(json.dumps(row),flush=True)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    try:
        torch.set_num_threads(8);torch.npu.set_device('npu:0');device=torch.device('npu:0')
        torch.npu.conv.allow_hf32=False;torch.npu.matmul.allow_hf32=False
        prediction=Path(args.prediction);meta=json.loads((prediction.parent/'result.json').read_text())
        assert sha(prediction)==meta['refined_sha256']
        pred=torch.load(prediction,map_location='cpu',weights_only=True)
        assert pred['step']==1000 and pred['latent_slice']==[0,67]
        sample=ROOT/'data/Charge_train_3856_4081.pt'
        assert sha(sample)=='5dc1df60fa60fc67c9712a181e283e3147ac6b2d209d1f876174dcb1bfefa778'
        data=torch.load(sample,map_location='cpu',weights_only=True)
        arms={'refiner':pred['refined'],'draft':data['upscaled'],'decoded_gt':data['reference']}
        assert all(z.shape==(24,67,68,120) and torch.isfinite(z).all() for z in arms.values())
        results={}
        if args.mode=='causal':
            log('load_causal_decoder');model=load_decoder('cpu');validate_decoder(model,device,out,log)
            for name,z in arms.items():results[name]=decode_causal(model,z,out,name,device,log)
            results['comparison']=comparison(out,['draft_causal','refiner_causal','decoded_gt_causal'],
                'draft_refiner_gt_causal.mp4',['Draft | causal decoder','Step1000 | causal decoder','Decoded GT | causal decoder'])
        else:
            log('load_original_vae');vae=load_video_vae(BASE,device)
            for name,z in arms.items():
                start=time.time();log('original_decode_start',arm=name)
                rgb=decode_latent(vae,z,device)[:,4:1084].contiguous()
                assert rgb.shape==(226,1080,1920,3)
                w=Writer(out/(name+'_original1080p.mp4'));w.push(rgb);results[name]=w.finish()
                previews=np.stack([np.array(Image.fromarray(f.numpy()).resize((640,360),Image.Resampling.LANCZOS)) for f in rgb])
                np.save(out/(name+'_original_preview.npy'),previews)
                results[name]['seconds']=time.time()-start;log('original_decode_complete',arm=name,seconds=time.time()-start)
                del rgb,previews;gc.collect();torch.npu.empty_cache()
            results['comparison']=comparison(out,['draft_original','refiner_original','decoded_gt_original'],
                'draft_refiner_gt_original.mp4',['Draft | original VAE','Step1000 | original VAE','Decoded GT | original VAE'])
        write(out/'result.json',dict(status='complete',mode=args.mode,checkpoint_step=1000,
            frames=226,fps=24,duration_seconds=226/24,latent_count=67,prediction_sha256=sha(prediction),
            training_sample_only=True,GT_scope='decoded target latent, not native RGB',
            source_sha256=sha(Path(__file__)),optimizer_updates=0,files=results))
        log('complete')
    except BaseException as e:
        write(out/'failure.json',dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
