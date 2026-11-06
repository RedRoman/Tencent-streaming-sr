"""Two-clip LR VAE roundtrip probe: two decodes, no encode/upscaler/training.

Use existing native z192, original raw LR derived from cached source384, and
the already saved lossless HR384 reference RGB cache. Metrics never read MP4.
The caller holds NPU5 locks. This script reads, and never changes, HF32 flags.
An error here locates damage in the VAE roundtrip, not the encoder alone.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

_boot=argparse.ArgumentParser(add_help=False)
_boot.add_argument('--root',type=Path,default=Path(os.environ.get('EXP_ROOT','${EXPERIMENT_ROOT}')))
_early,_=_boot.parse_known_args();sys.path.insert(0,str(_early.root.resolve()/'experiment'))

import torch
import torch.nn.functional as F
from PIL import Image
from data_utils import decode_latent,empty_cache,load_video_vae,select_device,synchronize,tensor_load
from decode_results import comparisons,label_image
from evaluate_h3_cmn import audit_video,file_record,module_source,read_json,require,vae_fingerprint
from evaluate_h3_conv3d_test16 import immutable_json
from train_h3_v5_motion import execution_precision
from train_upscaler_trial import sha256

IDS=('VIDGEN5S_0421364','VIDGEN5S_0665341')
LABELS=('raw source384','raw LR bicubic','LR VAE roundtrip bicubic','HR384 VAE roundtrip')


def record(path):
    p=Path(path).resolve();return dict(path=str(p),**file_record(p))


def rgb_record(value):
    h=hashlib.sha256()
    for frame in value:h.update(frame.contiguous().numpy().tobytes())
    return dict(sha256=h.hexdigest(),shape=list(value.shape),dtype=str(value.dtype),hash_scope='Raw contiguous RGB bytes, frame order')


def resize(frames,down=False):
    result=[]
    for start in range(0,120,4):
        x=frames[start:start+4].permute(0,3,1,2).float()
        x=F.avg_pool2d(x,2,2) if down else F.interpolate(x,size=(384,672),mode='bicubic',align_corners=False,antialias=True)
        result.append(x.clamp(0,255).round().byte().permute(0,2,3,1).contiguous())
    return torch.cat(result)


def metrics(output,target):
    """All120 frames/119 pairs; fixed target-only high-change ROI, no LPIPS job."""
    require(output.shape==target.shape and len(output)==120 and output.dtype==target.dtype==torch.uint8,'Metric RGB mismatch')
    sums=dict(abs=0.,square=0.,delta_abs=0.,delta_square=0.,roi_abs=0.);count=pairs=mask_count=0
    for start in range(0,120,4):
        a=output[start:start+4].float()/255.;b=target[start:start+4].float()/255.;err=a-b
        sums['abs']+=float(err.abs().sum(dtype=torch.float64));sums['square']+=float(err.square().sum(dtype=torch.float64));count+=err.numel()
        left=max(0,start-1);stop=min(start+4,120)
        a=output[left:stop].float()/255.;b=target[left:stop].float()/255.
        truth=b[1:]-b[:-1];delta=(a[1:]-a[:-1])-truth;mask=truth.abs().mean(-1)>=.05
        endpoint=(a[1:]-b[1:]).abs()+(a[:-1]-b[:-1]).abs()
        sums['delta_abs']+=float(delta.abs().sum(dtype=torch.float64));sums['delta_square']+=float(delta.square().sum(dtype=torch.float64))
        sums['roi_abs']+=float((endpoint*mask[...,None]).sum(dtype=torch.float64));pairs+=delta.numel();mask_count+=int(mask.sum())
    mse=sums['square']/count
    return dict(pixel_l1=sums['abs']/count,mse=mse,psnr_db=-10*math.log10(max(mse,1e-12)),
        delta_l1=sums['delta_abs']/pairs,temporal_difference_mse=sums['delta_square']/pairs,
        motion_roi_l1=sums['roi_abs']/max(mask_count*2*3,1),motion_pair_pixels=mask_count,
        motion_roi_fraction=mask_count/(pairs/3),frames=120,adjacent_pairs=119,roi_threshold=.05)


def context(args):
    old=read_json(args.history/'contract.json');old_sha=sha256(args.history/'contract.json');files={}
    def bind(key,path,digest=None):
        rec=record(path)
        if digest is not None:require(rec['sha256']==digest,'Fingerprint differs: '+key)
        files[key]=rec;return rec
    bind('script',__file__);bind('historical_contract',args.history/'contract.json',old_sha)
    bind('dev_manifest',args.pairs/'dev.jsonl',old['manifests']['dev']['sha256'])
    rows={r['sample_id']:r for r in [json.loads(x) for x in (args.pairs/'dev.jsonl').read_text().splitlines() if x.strip()]}
    require(set(IDS)<=set(rows) and all(rows[i]['split']=='dev' for i in IDS),'Unexpected sample split')
    sources={Path(m.__file__).name for m in tuple(sys.modules.values()) if getattr(m,'__file__',None) and
        Path(m.__file__).suffix=='.py' and Path(m.__file__).resolve().parent==(args.root/'experiment').resolve()}
    for name in sorted(sources):
        rec=bind('source/'+name,module_source(name))
        if name in old['source_sha256']:require(rec['sha256']==old['source_sha256'][name],'Frozen helper changed: '+name)
    require(vae_fingerprint(args.base_model)==old['decoder_fingerprint']['weights'],'VAE files differ')
    for name,rec in old['decoder_fingerprint']['weights'].items():bind('vae/'+name,args.base_model/'vae'/name,rec['sha256'])
    from diffusers import AutoencoderKLMiniMaxH3
    bind('vae/implementation',inspect.getfile(AutoencoderKLMiniMaxH3),old['decoder_fingerprint']['implementation_sha256'])
    data={}
    for ident in IDS:
        row=rows[ident];path=Path(row['path']);path=path if path.is_absolute() else args.pairs/path
        pair=bind('pair/'+ident,path,row['sha256']);folder=args.history/'baseline_cache/dev'/ident
        br=read_json(folder/'done.json')
        require(br['status']=='completed' and br['contract_sha256']==old_sha and br['sample_id']==ident and br['split']=='dev' and
            br['pair_sha256']==row['sha256'] and br['execution_precision']==old['execution_precision'],'HR cache identity/precision differs')
        require(set(br['files'])=={'rgb.pt'},'Unexpected HR cache file coverage')
        for key in ('weights','implementation_sha256'):require(br['decoder_fingerprint'][key]==old['decoder_fingerprint'][key],'HR cache decoder differs')
        rgb=bind('hr_rgb/'+ident,folder/'rgb.pt',br['files']['rgb.pt']['sha256']);receipt=bind('hr_receipt/'+ident,folder/'done.json')
        data[ident]=dict(pair=pair,hr_rgb=rgb,hr_receipt=receipt,source_video_sha256=row['source_video_sha256'])
    return dict(schema_version=1,diagnostic='Post-hoc two-dev LR VAE roundtrip probe',sample_ids=list(IDS),data=data,files=files,
        decoder_fingerprint=old['decoder_fingerprint'],execution_precision=old['execution_precision'],
        compute=dict(new_decodes=2,new_encodes=0,upscaler_forwards=0,training_updates=0,comparison_mp4=2),
        frames=120,fps=24,raw_lr_pixels=[336,192],common_pixels=[672,384],comparison_order=list(LABELS),
        resizing='RawLR=exact2x2 average of cached source384, round/clamp uint8; both LR displays use identical per-frame bicubic align_corners=False antialias=True, clamp/round uint8.',
        metric_scope='Uncompressed uint8/255. Native LR reconstruction versus rawLR; common384 all paths versus SAME raw source384. No LPIPS or lossy-MP4 metrics.',
        limitations=['Two previously inspected problem dev clips, post-hoc and not representative/blind.',
            'A bad LR reconstruction locates damage in the VAE encoder/decoder roundtrip, not the encoder alone.',
            'A clean LR reconstruction does not prove preservation of all fine information needed by the upscaler.',
            'No new encode, no upscaler, no motion-loss change, no saved model weights. Frozen decoder remains noncausal.'],output=str(args.output.resolve()))


def verify(args):
    c=context(args);path=args.output/'contract.json'
    if path.exists():require(read_json(path)==c,'Probe contract changed')
    else:
        require(not args.output.exists(),'Use a fresh probe output directory');args.output.mkdir(parents=True);immutable_json(path,c)
    return c,sha256(path)


def check_receipt(folder,digest):
    if not (folder/'done.json').exists():return None
    r=read_json(folder/'done.json');require(r['status']=='completed' and r['contract_sha256']==digest and r['sample_id']==folder.name,'Probe receipt differs')
    require(set(r['files'])=={'comparison.mp4','comparison.png','motion_contacts.png','metrics.json'},'Probe artifact coverage differs')
    for name,value in r['files'].items():require(file_record(folder/name)==value,'Probe artifact changed')
    return r


@torch.no_grad()
def run(args,c,digest):
    todo=[i for i in IDS if check_receipt(args.output/'clips'/i,digest) is None]
    if todo:
        device=select_device(args.device);require(execution_precision(device)==c['execution_precision'],'Math flags differ; do not alter HF32')
        vae=load_video_vae(args.base_model,device).eval().requires_grad_(False)
        require(sha256(inspect.getfile(type(vae)))==c['decoder_fingerprint']['implementation_sha256'],'Loaded VAE source differs')
        before=tuple((id(t),t._version) for t in (*vae.parameters(),*vae.buffers()))
        require(all(not p.requires_grad and p.grad is None and p.dtype==torch.float32 for p in vae.parameters()),'VAE is not frozen FP32')
        for ident in todo:
            folder=args.output/'clips'/ident;require(not folder.exists(),'Incomplete probe preserved; choose a fresh output before retry')
            folder.mkdir(parents=True);pair=tensor_load(c['data'][ident]['pair']['path']);meta=pair['metadata']
            require(meta['split']=='dev' and meta.get('sample_id',meta.get('id'))==ident and meta['source_video_sha256']==c['data'][ident]['source_video_sha256'] and
                meta['posterior']=='sample' and meta['posterior_seed']==42,'Pair provenance differs')
            source=pair['source_pixels'];draft=pair['draft']
            require(source.dtype==torch.uint8 and tuple(source.shape)==(120,384,672,3) and tuple(draft.shape)==(24,37,12,21) and
                bool(torch.isfinite(draft).all()),'LR pair geometry/values differ')
            raw_lr=resize(source,down=True);start=time.perf_counter();reconstructed=decode_latent(vae,draft,device,keep_frames=120);synchronize(device)
            seconds=time.perf_counter()-start;require(tuple(reconstructed.shape)==(120,192,336,3) and reconstructed.dtype==torch.uint8,'Native LR decode geometry differs')
            old=tensor_load(c['data'][ident]['hr_rgb']['path'])
            require(old['sample_id']==ident and old['split']=='dev' and old['pair_sha256']==c['data'][ident]['pair']['sha256'],'HR raw-cache payload differs')
            hr=old['reference'];require(tuple(hr.shape)==(120,384,672,3) and hr.dtype==torch.uint8,'HR reference shape differs')
            outputs=dict(zip(LABELS,(source,resize(raw_lr),resize(reconstructed),hr)))
            values={label:metrics(rgb,source) for label,rgb in outputs.items() if label!=LABELS[0]}
            entry=dict(sample_id=ident,frames=120,native_lr_roundtrip=metrics(reconstructed,raw_lr),common384_vs_raw_source=values,
                native_raw_lr=rgb_record(raw_lr),native_lr_reconstruction=rgb_record(reconstructed),
                displayed_raw_rgb={label:rgb_record(rgb) for label,rgb in outputs.items()},decode_seconds=seconds,
                no_upscaler_or_encoder_invoked=True,metric_scope=c['metric_scope'])
            comparisons(outputs,folder,fps=24.,crf=16,comparison_video=True)
            indices=(14,17,19,22,60);sheet=Image.new('RGB',(672*4,412*len(indices)))
            for row,index in enumerate(indices):
                for col,(label,rgb) in enumerate(outputs.items()):sheet.paste(label_image(rgb[index],f'{label} | frame {index}',(672,384)),(672*col,412*row))
            sheet.save(folder/'motion_contacts.png');media=audit_video(folder/'comparison.mp4',2688,412)
            immutable_json(folder/'metrics.json',entry)
            require(tuple((id(t),t._version) for t in (*vae.parameters(),*vae.buffers()))==before and
                all(not p.requires_grad and p.grad is None for p in vae.parameters()),'Frozen decoder state changed')
            require(execution_precision(device)==c['execution_precision'],'Precision changed')
            immutable_json(folder/'done.json',dict(status='completed',sample_id=ident,contract_sha256=digest,
                frozen_decoder_unchanged=True,execution_precision=c['execution_precision'],media_audit=media,
                files={name:file_record(folder/name) for name in ('comparison.mp4','comparison.png','motion_contacts.png','metrics.json')}))
            print(json.dumps(dict(stage='lr_roundtrip_completed',sample_id=ident,decode_seconds=seconds,native_lr_metrics=entry['native_lr_roundtrip'],
                common384_metrics=values)),flush=True)
            del pair,source,draft,raw_lr,reconstructed,old,hr,outputs
        del vae;empty_cache(device)
    result={};receipts={}
    for ident in IDS:
        folder=args.output/'clips'/ident;r=check_receipt(folder,digest)
        require(r is not None and r['frozen_decoder_unchanged'] and r['execution_precision']==c['execution_precision'],'Probe freeze evidence incomplete')
        a=r['media_audit'];require(a['passed'] and a['decoded_frames']==120 and a['fps']==24 and a['duration_seconds']==5,'Media coverage differs')
        result[ident]=read_json(folder/'metrics.json');receipts[ident]=record(folder/'done.json')
    require(context(args)==c,'Final frozen inputs/source/decoder files changed')
    summary=dict(status='completed',contract_sha256=digest,sample_ids=list(IDS),samples=result,receipts=receipts,
        new_decodes=2,new_encodes=0,upscaler_forwards=0,training_updates=0,full_decoded_mp4_files=2,
        frozen_inputs_and_sources_unchanged=True,limitations=c['limitations'])
    immutable_json(args.output/'summary.json',summary);print(json.dumps(dict(stage='completed',new_decodes=2,mp4_files=2)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,parents=[_boot]);p.add_argument('--stage',choices=('check','run'),default='check')
    for name in ('pairs','history','base-model','output'):p.add_argument('--'+name,type=Path)
    p.add_argument('--device',default='npu:0');args=p.parse_args();args.root=args.root.resolve()
    require(args.root==_early.root.resolve(),'Import root differs');home=args.root/'upscaler_v5_motion_20261106'
    args.pairs=args.pairs or args.root/'upscaler_ultraflash/pairs';args.history=args.history or home/'evaluation'
    args.base_model=args.base_model or Path('${H3_BASE_MODEL}')
    args.output=args.output or home/'diagnostics/lr_roundtrip_v1';torch.set_num_threads(4)
    c,digest=verify(args)
    if args.stage=='check':print(json.dumps(dict(status='checked',contract_sha256=digest,compute=c['compute'],no_model_loaded=True)),flush=True)
    else:run(args,c,digest)


if __name__=='__main__':main()
