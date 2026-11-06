"""Post-hoc, frozen H3 resolution/codec-chain diagnostic, never training.

check/prepare: CPU-only contracts and strict native-source reads.
latent: six VAE encodes and ten upscaler forwards; no pixel decoding.
pixels: only two preselected clips; common384 raw-RGB metrics, native768
auxiliary metrics/crops, LR VAE roundtrip, ordinary interpolation controls.
No source edits, optimizer, checkpoint selection, or saved model weights.
The caller must hold the established NPU5 locks; this script never sets HF32.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

_boot=argparse.ArgumentParser(add_help=False)
_boot.add_argument('--root',type=Path,default=Path(os.environ.get('EXP_ROOT','${EXPERIMENT_ROOT}')))
_early,_=_boot.parse_known_args();sys.path.insert(0,str(_early.root.resolve()/'experiment'))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
from data_utils import (decode_latent,empty_cache,encode_pixels,ffmpeg_executable,
    load_video_vae,select_device,synchronize,tensor_load)
from decode_results import comparisons
from evaluate_h3_cmn import audit_video,file_record,module_source,read_json,require,vae_fingerprint
from evaluate_h3_conv3d_hybrid import load_lpips
from evaluate_h3_conv3d_test16 import archive_incomplete,immutable_json
from evaluate_h3_v5_motion import SOURCE_NAMES,frame_regions,rgb_metrics,rgb_transition_metrics,transition_pair_groups
from h3_conv3d_hybrid import load_student_checkpoint
from prepare_data import load_author_upscaler
from prepare_upscaler_5s_pairs import read_first_120_strict
from train_h3_v5_motion import execution_precision
from train_upscaler_trial import atomic_torch_save,sha256,wrapper_forward

IDS=('VIDGEN5S_0421364','VIDGEN5S_0665341','VIDGEN5S_0589299','VIDGEN5S_0760711')
PIXEL_IDS=IDS[:2]
PREDICTIONS={'teacher_native':IDS,'motion_native':IDS,'teacher_information_matched':PIXEL_IDS}
CROP_BOXES={'VIDGEN5S_0421364':(140,80,560,380),'VIDGEN5S_0665341':(180,10,540,370)}
CROP_FRAMES=(17,19,60)
COMMON_LABELS=('raw_source384','rgb_bicubic192_to384','lr_vae192_roundtrip_bicubic',
    'vae384_roundtrip','vae768_roundtrip_down','teacher192_to384','motion192_to384',
    'teacher384_to768_down','motion384_to768_down','teacher_bicubic192_to384_to768_down')
POSTERIOR_SEED=42


def record(path):
    p=Path(path).resolve();return dict(path=str(p),**file_record(p))


def pixel_record(x):
    require(x.dtype==torch.uint8 and x.ndim==4 and x.shape[-1]==3,'Expected uint8 THWC pixels')
    h=hashlib.sha256()
    for frame in x:h.update(frame.contiguous().numpy().tobytes())
    return dict(sha256=h.hexdigest(),shape=list(x.shape),dtype='uint8',hash_scope='Contiguous RGB24 frames in temporal order; no header')


def frozen_versions(*models):
    return tuple((id(t),t._version,str(t.dtype),tuple(t.shape)) for m in models for t in (*m.parameters(),*m.buffers()))


def check_frozen(*models):
    require(all(not m.training for m in models),'Frozen model entered training mode')
    require(all(not p.requires_grad and p.grad is None and (not p.is_floating_point() or p.dtype==torch.float32)
        for m in models for p in m.parameters()),'Parameter dtype/freeze/gradient changed')


def precision(device,contract):
    actual=execution_precision(device)
    require(actual==contract['execution_precision'],'Execution precision differs; flags are never changed here')
    return actual


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def context(args):
    old=read_json(args.history/'contract.json');old_sha=sha256(args.history/'contract.json')
    summary=read_json(args.history/'summary.json')
    require(summary['status']=='completed' and summary['contract_sha256']==old_sha,'Completed original evaluation differs')
    train=read_json(args.root/'upscaler_v5_motion_20261106/motion_steps2000/contract.json')
    require(train['execution_precision']==old['execution_precision'],'Training/evaluation precision differs')
    original=train['original_upscaler_fingerprint'];files={}
    def bind(key,path,digest=None):
        rec=record(path)
        if digest is not None:require(rec['sha256']==digest,'Input fingerprint differs: '+key)
        files[key]=rec;return rec
    bind('script',__file__);bind('historical_contract',args.history/'contract.json',old_sha)
    bind('historical_summary',args.history/'summary.json')
    bind('motion_training_contract',args.root/'upscaler_v5_motion_20261106/motion_steps2000/contract.json')
    bind('upscaler_weights',args.weights,original['base_weights_sha256'])
    bind('upscaler_loader',args.upstream/'open_h3/upscaler.py',original['loader_sha256'])
    bind('upscaler_node',args.upstream/'open_h3/upscaler_node.py',original['loaded_node_sha256'])
    imported={Path(m.__file__).name for m in tuple(sys.modules.values()) if getattr(m,'__file__',None) and
        Path(m.__file__).suffix=='.py' and Path(m.__file__).resolve().parent==(args.root/'experiment').resolve()}
    for name in sorted(set(SOURCE_NAMES)|imported):
        rec=bind('source/'+name,module_source(name))
        if name in old['source_sha256']:require(rec['sha256']==old['source_sha256'][name],'Frozen source changed: '+name)
    ready=read_json(args.pairs/'READY.json');plan=read_json(args.pairs/'plan.json')
    require(ready['status']=='ready' and ready['plan_sha256']==old['plan_sha256'] and plan['posterior_seed']==42,'Old data/seed differs')
    bind('ready',args.pairs/'READY.json',old['ready_sha256']);bind('plan',args.pairs/'plan.json',old['plan_sha256'])
    bind('dev_manifest',args.pairs/'dev.jsonl',old['manifests']['dev']['sha256'])
    dev=rows(args.pairs/'dev.jsonl');by_id={r['sample_id']:r for r in dev};plan_rows={r['sample_id']:r for r in plan['entries']}
    require(len(dev)==len(by_id)==16 and all(r['split']=='dev' for r in dev) and set(IDS)<=set(by_id),'Wrong diagnostic dev IDs')
    pairs={}
    for ident in IDS:
        row=by_id[ident];source=plan_rows[ident]
        require(source['split']=='dev' and source['source_video_sha256']==row['source_video_sha256'] and
            source['source_frames_available']>=120 and float(source['source_fps'])==24.,'Source plan differs')
        path=Path(row['path']);path=path if path.is_absolute() else args.pairs/path
        pairs[ident]=dict(pair=bind('pair/'+ident,path,row['sha256']),
            source=bind('source_video/'+ident,source['source_video_path'],row['source_video_sha256']),manifest=row,plan=source)
    require(vae_fingerprint(args.base_model)==old['decoder_fingerprint']['weights']==train['decoder_fingerprint']['weights'],'VAE identity differs')
    for name,rec in old['decoder_fingerprint']['weights'].items():bind('vae/'+name,Path(args.base_model)/'vae'/name,rec['sha256'])
    from diffusers import AutoencoderKLMiniMaxH3
    bind('vae/implementation',inspect.getfile(AutoencoderKLMiniMaxH3),old['decoder_fingerprint']['implementation_sha256'])
    import lpips
    for name,path,key in (('backbone',Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth','backbone_sha256'),
        ('linear',Path(lpips.__file__).resolve().parent/'weights/v0.1/alex.pth','linear_sha256'),
        ('implementation',Path(inspect.getfile(lpips.LPIPS)),'implementation_sha256')):
        bind('lpips/'+name,path,old['lpips_fingerprint'][key])
    motion=old['selected_checkpoints']['motion'];bind('motion_checkpoint',motion['path'],motion['sha256'])
    require(motion==summary['selected_checkpoints']['motion'] and motion['selected_step']==2000,'Selected motion checkpoint changed')
    caches={}
    teacher_ready=read_json(args.teacher_cache/'READY.json');bind('teacher_ready',args.teacher_cache/'READY.json')
    require(teacher_ready['status']=='ready' and teacher_ready['contract']['teacher_sha256']==original['base_weights_sha256'] and
        teacher_ready['contract']['plan_sha256']==old['plan_sha256'],'Teacher cache differs')
    for ident in IDS:
        teacher=teacher_ready['records'][ident]
        require(teacher['split']=='dev' and teacher['pair_sha256']==by_id[ident]['sha256'],'Teacher pair differs')
        teacher_file=bind('teacher_cache/'+ident,args.teacher_cache/teacher['path'],teacher['sha256'])
        folder=args.history/'predictions/dev/motion'/ident;pr=read_json(folder/'done.json')
        require(pr['status']=='completed' and pr['contract_sha256']==old_sha and pr['sample_id']==ident and pr['split']=='dev' and
            pr['variant']=='motion' and pr['pair_sha256']==by_id[ident]['sha256'] and pr['checkpoint_sha256']==motion['sha256'],
            'Old motion prediction provenance differs')
        require(pr['frozen_student_unchanged'] and pr['execution_precision_start']==pr['execution_precision_end']==old['execution_precision'],
            'Old motion precision/freeze differs')
        motion_file=bind('motion_cache/'+ident,folder/'latent.pt',pr['files']['latent.pt']['sha256'])
        motion_receipt=bind('motion_receipt/'+ident,folder/'done.json')
        caches[ident]=dict(teacher=teacher_file,motion=motion_file,motion_receipt=motion_receipt)
        if ident in PIXEL_IDS:
            cache=args.history/'baseline_cache/dev'/ident;br=read_json(cache/'done.json')
            require(br['status']=='completed' and br['contract_sha256']==old_sha and br['sample_id']==ident and
                br['split']=='dev' and br['pair_sha256']==by_id[ident]['sha256'] and br['execution_precision']==old['execution_precision'],
                'Old raw reference cache differs')
            for key in ('weights','implementation_sha256'):require(br['decoder_fingerprint'][key]==old['decoder_fingerprint'][key],'Old decoder cache identity differs')
            require(set(br['files'])=={'rgb.pt'},'Unexpected raw RGB cache files')
            caches[ident]['reference_rgb']=bind('reference_rgb/'+ident,cache/'rgb.pt',br['files']['rgb.pt']['sha256'])
            caches[ident]['reference_rgb_receipt']=bind('reference_rgb_receipt/'+ident,cache/'done.json')
    return dict(schema_version=2,diagnostic='Post-hoc resolution and codec-chain diagnosis; no training or checkpoint selection',
        sample_ids=list(IDS),pixel_sample_ids=list(PIXEL_IDS),pairs=pairs,caches=caches,files=files,
        original_upscaler_fingerprint=original,decoder_fingerprint=old['decoder_fingerprint'],lpips_fingerprint=old['lpips_fingerprint'],
        motion_checkpoint=motion,execution_precision=old['execution_precision'],posterior='sample',posterior_seed=42,
        frames=120,fps=24,native_pixels=[1344,768],common_pixels=[672,384],lr_pixels=[336,192],
        new_encodings={'native768':list(IDS),'information_matched384':list(PIXEL_IDS)},new_predictions={k:list(v) for k,v in PREDICTIONS.items()},
        expected_compute=dict(native768_encodes=4,information_matched384_encodes=2,teacher_native_forwards=4,
            motion_native_forwards=4,teacher_information_matched_forwards=2,pixel_clips=2,new_pixel_decodes_per_clip=7,
            reused_reference384_decodes_per_clip=1,comparison_mp4_files=4),
        reader='First120 native1344x768 RGB24 frames, trim only, original24fps PTS, no scale/fps/interpolation/padding/loop',
        reader_alignment='CPU replay of the frozen original384 reader must exactly reproduce cached source_pixels. Native reader independently requires consecutive zero-based24fps PTS and the same source SHA. RGB avg2 projection is diagnostic only, not required bit-exact to the old FFmpeg resize/color-conversion path.',
        resize='LR=exact2x2 average of cached source384, round/clamp uint8. Bicubic=PyTorch float32 align_corners=False antialias=True then clamp/round uint8. Common projection768->384=exact2x2 average then round/clamp uint8.',
        common_metric_target='The same original cached pair.source_pixels at672x384 for EVERY node; GT motion masks derived only from these raw pixels, not decoded HR.',
        native_metric_target='Original1344x768 RGB first120; auxiliary L1/MSE/PSNR/delta metrics only, not ranked against common384 scores.',
        latent_metric_scope='First36 of37 normalized latents, within each target scale only. No cross-resolution latent MSE ranking; no resized old optical flow.',
        frame_regions=frame_regions(),transition_pair_groups=transition_pair_groups(),common_labels=list(COMMON_LABELS),
        native_crop_frames=list(CROP_FRAMES),native_crop_boxes384={k:list(v) for k,v in CROP_BOXES.items()},
        limitations=['Four already inspected problem dev clips; two predeclared pixel clips; post-hoc diagnosis, not blind or representative.',
            '384 input has more source information and still does not equal the upstream explicit576/768 input modes; no claim of official-domain validation.',
            'Information-matched branch adds pixels by interpolation, not true source detail; compared at the same384 input grid with the same posterior seed/noise shape.',
            'Same seed across different resolutions is not the same noise field; no posterior-mode substitution.',
            'VAE roundtrip errors identify the encoder/decoder chain, not the encoder alone. Upscaler output errors may interact with the decoder.',
            'Original H3 VAE remains noncausal; this is not an end-to-end streaming test.',
            'No true1152 reference is created; only real source768 and its derived lower resolutions.'],output=str(args.output.resolve()))


def contract_for(args):
    current=context(args);path=args.output/'contract.json'
    if path.exists():require(read_json(path)==current,'Resolution diagnostic contract changed')
    else:
        require(not args.output.exists(),'Use a fresh output directory')
        args.output.mkdir(parents=True);immutable_json(path,current)
    return current,sha256(path)


def load_pair(contract,ident):
    rec=contract['pairs'][ident];value=tensor_load(rec['pair']['path']);meta=value['metadata']
    require(meta['split']=='dev' and meta.get('sample_id',meta.get('id'))==ident and meta['source_video_sha256']==rec['source']['sha256'] and
        meta['posterior']=='sample' and meta['posterior_seed']==42,'Cached pair provenance/sampling differs')
    require(tuple(value['draft'].shape)==(24,37,12,21) and tuple(value['reference'].shape)==(24,37,24,42) and
        tuple(value['source_pixels'].shape)==(120,384,672,3) and value['source_pixels'].dtype==torch.uint8,'Cached pair geometry differs')
    require(bool(torch.isfinite(value['draft']).all()) and bool(torch.isfinite(value['reference']).all()),'Nonfinite cached pair')
    return value


def resize_rgb(frames,size,mode):
    """Fixed per-frame spatial operators, no temporal mixing; bounded memory."""
    result=[]
    for start in range(0,len(frames),4):
        x=frames[start:start+4].permute(0,3,1,2).float()
        if mode=='area2':
            require(tuple(x.shape[-2:])==(2*size[0],2*size[1]),'Expected exact2x downsample')
            x=F.avg_pool2d(x,2,2)
        elif mode=='bicubic':x=F.interpolate(x,size=size,mode='bicubic',align_corners=False,antialias=True)
        else:raise ValueError(mode)
        result.append(x.clamp(0,255).round().byte().permute(0,2,3,1).contiguous())
    return torch.cat(result)


def simple_metrics(output,target):
    """Whole120 RGB metrics with bounded CPU memory; all119 real transitions."""
    require(output.shape==target.shape and len(output)==120 and output.dtype==target.dtype==torch.uint8,'RGB comparison mismatch')
    absolute=squared=delta_absolute=delta_squared=0.;count=delta_count=0
    for start in range(0,120,4):
        a=output[start:start+4].float()/255.;b=target[start:start+4].float()/255.;diff=a-b
        absolute+=float(diff.abs().sum(dtype=torch.float64));squared+=float(diff.square().sum(dtype=torch.float64));count+=diff.numel()
        left=max(0,start-1);stop=min(120,start+4)
        a=output[left:stop].float()/255.;b=target[left:stop].float()/255.
        delta=(a[1:]-a[:-1])-(b[1:]-b[:-1])
        delta_absolute+=float(delta.abs().sum(dtype=torch.float64));delta_squared+=float(delta.square().sum(dtype=torch.float64));delta_count+=delta.numel()
    mse=squared/count
    return dict(pixel_l1=absolute/count,mse=mse,psnr_db=-10*math.log10(max(mse,1e-12)),
        delta_l1=delta_absolute/delta_count,temporal_difference_mse=delta_squared/delta_count,frames=120,adjacent_pairs=119)


def read_native(source):
    command=[ffmpeg_executable(),'-hide_banner','-nostdin','-loglevel','info','-nostats','-i',str(source),
        '-map','0:v:0','-an','-sn','-dn','-vf','trim=end_frame=120,showinfo','-frames:v','120','-vsync','0',
        '-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    result=subprocess.run(command,capture_output=True,timeout=300)
    log=result.stderr.decode('utf-8',errors='replace')
    require(result.returncode==0,'Native source read failed: '+log[-1500:])
    pattern=r'\[Parsed_showinfo[^\]]*\].*?\bn:\s*(\d+)\s+pts:\s*-?\d+\s+pts_time:([-+0-9.eE]+).*?\bs:(\d+)x(\d+)'
    events=[(int(n),float(t),int(w),int(h)) for n,t,w,h in re.findall(pattern,log)]
    require(len(events)==120 and [v[0] for v in events]==list(range(120)),'Native reader did not expose exactly first120 frames')
    require(all((w,h)==(1344,768) for _,_,w,h in events),'Source is not native1344x768')
    require(all(abs(t-i/24.)<=1e-5 for i,t,_,_ in events),'Source PTS are not consecutive zero-based24fps')
    expected=120*768*1344*3;require(len(result.stdout)==expected,'Native raw RGB byte count differs')
    pixels=torch.from_numpy(np.frombuffer(result.stdout,dtype=np.uint8).reshape(120,768,1344,3).copy())
    proof=dict(source=record(source),frames=120,width=1344,height=768,fps=24,first_pts=events[0][1],last_pts=events[-1][1],
        raw_bytes=len(result.stdout),rgb=pixel_record(pixels),command=command,no_external_time_or_spatial_resampling=True)
    return pixels,proof,log


def prepare(args,contract,digest):
    done=args.output/'prepare.json'
    if done.exists():return check_prepared(args,contract,digest)
    evidence={}
    for ident in IDS:
        directory=args.output/'readers'/ident
        if (directory/'proof.json').exists():
            saved=read_json(directory/'proof.json')
            require(saved['contract_sha256']==digest and saved['sample_id']==ident,'Reader proof identity changed')
            require(file_record(directory/'reader.log')==saved['reader_log'],'Reader log changed');evidence[ident]=record(directory/'proof.json');continue
        archive_incomplete(directory,args.output,'Incomplete native source reader evidence retained');directory.mkdir(parents=True)
        pair=load_pair(contract,ident);native,proof,log=read_native(contract['pairs'][ident]['source']['path'])
        legacy_dir=directory/'legacy_reader';legacy_dir.mkdir()
        legacy_pixels,legacy_proof=read_first_120_strict(contract['pairs'][ident]['source']['path'],legacy_dir,ident)
        require(torch.equal(legacy_pixels,pair['source_pixels']),'Frozen original384 reader does not reproduce cached source_pixels')
        legacy=dict(reader=legacy_proof,rgb=pixel_record(legacy_pixels),cached_source384_bitexact=True,
            files={str(p.relative_to(directory)):file_record(p) for p in sorted(legacy_dir.iterdir()) if p.is_file()})
        del legacy_pixels
        low=resize_rgb(pair['source_pixels'],(192,336),'area2');projected=resize_rgb(native,(384,672),'area2')
        info=resize_rgb(low,(384,672),'bicubic') if ident in PIXEL_IDS else None
        (directory/'reader.log').write_text(log)
        value=dict(sample_id=ident,contract_sha256=digest,reader=proof,reader_log=file_record(directory/'reader.log'),
            legacy384_replay=legacy,
            cached_source384=pixel_record(pair['source_pixels']),derived_lr192=pixel_record(low),
            projected_native768_to384=pixel_record(projected),projection_vs_cached_source384=simple_metrics(projected,pair['source_pixels']),
            projection_max_abs_uint8=int((projected.to(torch.int16)-pair['source_pixels'].to(torch.int16)).abs().max()),
            information_matched384=pixel_record(info) if info is not None else None)
        immutable_json(directory/'proof.json',value);evidence[ident]=record(directory/'proof.json')
        print(json.dumps(dict(stage='source_prepared',sample_id=ident,projection_vs_cached_source384=value['projection_vs_cached_source384'])),flush=True)
        del pair,native,low,projected,info
    require(context(args)==contract,'Frozen inputs changed during prepare')
    value=dict(status='prepared',contract_sha256=digest,sample_ids=list(IDS),readers=evidence,no_accelerator_model_loaded=True)
    immutable_json(done,value);return value


def check_prepared(args,contract,digest):
    value=read_json(args.output/'prepare.json')
    require(value['status']=='prepared' and value['contract_sha256']==digest and value['sample_ids']==list(IDS),'Preparation incomplete')
    require(set(value['readers'])==set(IDS),'Preparation reader set differs')
    for ident,rec in value['readers'].items():
        require(record(args.output/'readers'/ident/'proof.json')==rec,'Reader proof changed')
        proof=read_json(rec['path']);require(file_record(args.output/'readers'/ident/'reader.log')==proof['reader_log'],'Reader log changed')
        require(proof['sample_id']==ident and proof['contract_sha256']==digest,'Reader proof identity differs')
        require(proof['reader']['source']==contract['pairs'][ident]['source'],'Reader/source contract differs')
        legacy=proof['legacy384_replay'];expected_files={f'legacy_reader/{ident}.read.log',f'legacy_reader/{ident}.reader.json'}
        require(set(legacy['files'])==expected_files and legacy['cached_source384_bitexact'] and
            legacy['rgb']==proof['cached_source384'],'Original384 replay identity/parity differs')
        for name,expected in legacy['files'].items():
            require(file_record(args.output/'readers'/ident/name)==expected,'Original384 reader evidence changed')
        require(read_json(args.output/'readers'/ident/f'legacy_reader/{ident}.reader.json')==legacy['reader'],
            'Original384 reader proof changed')
        require(legacy['reader']['exact_120_frames'] and legacy['reader']['fps_generated_duplicate_frames']==0 and
            legacy['reader']['fps_filter']['input_frames']>=120,'Original384 reader coverage differs')
    return value


def fresh_native(args,contract,ident):
    pixels,proof,_=read_native(contract['pairs'][ident]['source']['path'])
    expected=read_json(args.output/'readers'/ident/'proof.json')['reader']
    require(proof==expected,'Repeated strict source read differs from frozen preparation')
    return pixels


def artifact(directory,digest,kind):
    path=directory/'done.json'
    if not path.exists():return None
    rec=read_json(path)
    require(rec['status']=='completed' and rec['contract_sha256']==digest and rec['kind']==kind and rec['sample_id']==directory.name,
        'Artifact receipt identity differs')
    require(set(rec['files'])=={'latent.pt'},'Unexpected latent artifact files')
    require(file_record(directory/'latent.pt')==rec['files']['latent.pt'],'Latent artifact changed')
    output=directory.parents[2];contract=read_json(output/'contract.json')
    require(sha256(output/'contract.json')==digest,'Latent artifact contract file differs')
    expected=contract['execution_precision'];label=directory.parent.name;ident=directory.name
    require(rec['variant']==label,'Latent artifact variant differs')
    require(rec['execution_precision_start']==rec['execution_precision_end']==expected,'Latent artifact precision differs')
    require(rec['frozen_vae_unchanged'] if kind=='encoding' else rec['frozen_upscaler_unchanged'], 'Latent artifact freeze check failed')
    if kind=='encoding':
        require(label in contract['new_encodings'] and ident in contract['new_encodings'][label],'Unexpected encoding identity')
        proof=read_json(output/'readers'/ident/'proof.json')
        expected_rgb=proof['reader']['rgb'] if label=='native768' else proof['information_matched384']
        require(rec['posterior']==contract['posterior'] and rec['posterior_seed']==contract['posterior_seed'] and
            rec['input_rgb']==expected_rgb,'Encoding input/sampling provenance differs')
    else:
        require(kind=='prediction' and label in contract['new_predictions'] and ident in contract['new_predictions'][label],
            'Unexpected prediction identity')
        model='motion' if label=='motion_native' else 'teacher'
        checkpoint=contract['motion_checkpoint']['sha256'] if model=='motion' else contract['original_upscaler_fingerprint']['base_weights_sha256']
        require(rec['model']==model and rec['checkpoint_sha256']==checkpoint,'Prediction model/checkpoint differs')
        if label=='teacher_information_matched':
            parent=output/'encodings/information_matched384'/ident
            require(artifact(parent,digest,'encoding') is not None,'Missing prediction input encoding')
            origin=record(parent/'done.json')
        else:origin=contract['pairs'][ident]['pair']
        require(rec['input_origin']==origin,'Prediction input receipt differs')
    return rec


def latent_receipts(args,contract,digest):
    receipts={}
    for group,key,kind in (('encodings','new_encodings','encoding'),('predictions','new_predictions','prediction')):
        for label,idents in contract[key].items():
            for ident in idents:
                folder=args.output/group/label/ident
                require(artifact(folder,digest,kind) is not None,'Incomplete latent evidence chain')
                receipts[label+'/'+ident]=record(folder/'done.json')
    require(len(receipts)==16,'Expected six encoding and ten prediction receipts')
    return receipts


def verify_latent_summary(args,contract,digest):
    value=read_json(args.output/'latent_summary.json')
    require(value['status']=='latent_completed' and value['contract_sha256']==digest and
        value['no_pixel_decoder_invoked'] and value['frozen_inputs_and_sources_unchanged'],'Latent summary identity differs')
    require(value['receipts']==latent_receipts(args,contract,digest),'Latent summary receipt chain changed')
    return value


def load_new(args,digest,group,label,ident):
    folder=args.output/group/label/ident;rec=artifact(folder,digest,'encoding' if group=='encodings' else 'prediction')
    require(rec is not None and rec['variant']==label,'Missing/mismatched new latent')
    value=tensor_load(folder/'latent.pt')
    require(value['sample_id']==ident and value['variant']==label and value['contract_sha256']==digest,'New latent metadata differs')
    z=value['latent'];expected=(24,37,24,42) if label=='information_matched384' else (24,37,48,84)
    require(tuple(z.shape)==expected and bool(torch.isfinite(z).all()),'New latent geometry/values differ')
    return z.float(),record(folder/'done.json')


def old_prediction(contract,label,ident):
    rec=contract['caches'][ident][label];require(record(rec['path'])==rec,'Old prediction changed')
    value=tensor_load(rec['path'])
    if label=='teacher':
        require(value['metadata']['split']=='dev' and value['metadata']['pair_sha256']==contract['pairs'][ident]['pair']['sha256'],'Old teacher identity differs')
    else:require(value['sample_id']==ident and value['split']=='dev' and value['pair_sha256']==contract['pairs'][ident]['pair']['sha256'],'Old motion identity differs')
    require(tuple(value['latent'].shape)==(24,37,24,42) and bool(torch.isfinite(value['latent']).all()),'Old prediction shape/values differ')
    return value['latent'].float()


@torch.no_grad()
def encode_new(args,contract,digest,device):
    todo=[(label,ident) for label,idents in contract['new_encodings'].items() for ident in idents
        if artifact(args.output/'encodings'/label/ident,digest,'encoding') is None]
    if not todo:return
    vae=load_video_vae(args.base_model,device).eval().requires_grad_(False);check_frozen(vae)
    require(sha256(inspect.getfile(type(vae)))==contract['decoder_fingerprint']['implementation_sha256'],'Loaded VAE implementation differs')
    before=frozen_versions(vae);mode=precision(device,contract)
    for label,ident in todo:
        directory=args.output/'encodings'/label/ident;archive_incomplete(directory,args.output,'Incomplete new encoding retained');directory.mkdir(parents=True)
        if label=='native768':pixels=fresh_native(args,contract,ident)
        else:
            pair=load_pair(contract,ident);low=resize_rgb(pair['source_pixels'],(192,336),'area2')
            pixels=resize_rgb(low,(384,672),'bicubic');del pair,low
            require(pixel_record(pixels)==read_json(args.output/'readers'/ident/'proof.json')['information_matched384'],'Information-matched pixels changed')
        start=time.perf_counter();z=encode_pixels(vae,pixels,device,42);synchronize(device)
        shape=(24,37,48,84) if label=='native768' else (24,37,24,42)
        require(tuple(z.shape)==shape and bool(torch.isfinite(z).all()),'New encoded shape/values differ')
        atomic_torch_save(directory/'latent.pt',dict(latent=z,sample_id=ident,variant=label,contract_sha256=digest))
        check_frozen(vae);require(frozen_versions(vae)==before,'Frozen encoder/decoder state changed')
        immutable_json(directory/'done.json',dict(status='completed',kind='encoding',variant=label,sample_id=ident,
            contract_sha256=digest,posterior='sample',posterior_seed=42,input_rgb=pixel_record(pixels),seconds=time.perf_counter()-start,
            execution_precision_start=mode,execution_precision_end=precision(device,contract),frozen_vae_unchanged=True,
            files={'latent.pt':file_record(directory/'latent.pt')}))
        print(json.dumps(dict(stage='encoded',variant=label,sample_id=ident,shape=list(z.shape))),flush=True);del pixels,z
    del vae;empty_cache(device)


def latent_metric(pred,target):
    require(pred.shape==target.shape and pred.ndim==4 and pred.shape[1]==37,'Latent comparison geometry differs')
    a=pred[:,:36].float();b=target[:,:36].float();delta=a-b
    result=dict(latent_mse=float(delta.square().mean()),latent_mae=float(delta.abs().mean()),
        temporal_difference_mse=float(((a[:,1:]-a[:,:-1])-(b[:,1:]-b[:,:-1])).square().mean()),
        valid_latent_frames=36,reference_std=float(b.std(unbiased=False)),latent_height=pred.shape[2],latent_width=pred.shape[3])
    require(all(math.isfinite(v) for v in result.values()),'Nonfinite latent metric');return result


@torch.no_grad()
def infer_new(args,contract,digest,device):
    for model_label in ('teacher','motion'):
        labels=('teacher_native','teacher_information_matched') if model_label=='teacher' else ('motion_native',)
        todo=[(label,ident) for label in labels for ident in PREDICTIONS[label]
            if artifact(args.output/'predictions'/label/ident,digest,'prediction') is None]
        if not todo:continue
        if model_label=='teacher':
            node=load_author_upscaler(args.upstream,args.weights,device)
            model=node.load_model('upscaler',device,'fp32').eval().requires_grad_(False)
        else:
            model=load_student_checkpoint(contract['motion_checkpoint']['path'],args.upstream,args.weights,device).eval().requires_grad_(False)
            node=model.adapter.node
        require(sum(p.numel() for p in model.parameters())==345280216,'Unexpected upscaler parameter count')
        check_frozen(model);before=frozen_versions(model);mode=precision(device,contract)
        for label,ident in todo:
            directory=args.output/'predictions'/label/ident;archive_incomplete(directory,args.output,'Incomplete high-resolution prediction retained');directory.mkdir(parents=True)
            if label=='teacher_information_matched':z,origin=load_new(args,digest,'encodings','information_matched384',ident)
            else:
                pair=load_pair(contract,ident);z=pair['reference'].float();origin=contract['pairs'][ident]['pair'];del pair
            x=z[None].to(device);start=time.perf_counter()
            y=wrapper_forward(node,model,x,(37,48,84)) if model_label=='teacher' else model(x)
            synchronize(device);require(tuple(y.shape)==(1,24,37,48,84) and bool(torch.isfinite(y).all()),'High-resolution upscaler output invalid')
            result=y[0].cpu();atomic_torch_save(directory/'latent.pt',dict(latent=result,sample_id=ident,variant=label,contract_sha256=digest))
            check_frozen(model);require(frozen_versions(model)==before,'Frozen upscaler state changed')
            immutable_json(directory/'done.json',dict(status='completed',kind='prediction',variant=label,sample_id=ident,
                contract_sha256=digest,input_origin=origin,model=model_label,seconds=time.perf_counter()-start,
                checkpoint_sha256=contract['motion_checkpoint']['sha256'] if model_label=='motion' else contract['original_upscaler_fingerprint']['base_weights_sha256'],
                execution_precision_start=mode,execution_precision_end=precision(device,contract),frozen_upscaler_unchanged=True,
                files={'latent.pt':file_record(directory/'latent.pt')}))
            print(json.dumps(dict(stage='predicted',variant=label,sample_id=ident,shape=list(result.shape))),flush=True);del x,y,z,result
        if model_label=='motion':model.adapter.remove()
        node.MODEL_CACHE.clear();del model,node;empty_cache(device)


def summarize_latents(args,contract,digest):
    groups={'target384':{'teacher':{},'motion':{}},'target768':{'teacher':{},'motion':{}},
        'target768_information_control_two_clips':{'teacher_information_matched':{},'teacher_native':{}}};receipts=latent_receipts(args,contract,digest)
    for ident in IDS:
        pair=load_pair(contract,ident);high,rec=load_new(args,digest,'encodings','native768',ident);receipts['native768/'+ident]=rec
        for label in ('teacher','motion'):
            old=old_prediction(contract,label,ident);groups['target384'][label][ident]=latent_metric(old,pair['reference'])
            new,rec=load_new(args,digest,'predictions',label+'_native',ident);receipts[label+'_native/'+ident]=rec
            groups['target768'][label][ident]=latent_metric(new,high)
            if label=='teacher' and ident in PIXEL_IDS:
                groups['target768_information_control_two_clips']['teacher_native'][ident]=latent_metric(new,high)
        if ident in PIXEL_IDS:
            z,rec=load_new(args,digest,'predictions','teacher_information_matched',ident);receipts['teacher_information_matched/'+ident]=rec
            groups['target768_information_control_two_clips']['teacher_information_matched'][ident]=latent_metric(z,high)
        del pair,high
    aggregates={}
    for group,variants in groups.items():
        aggregates[group]={label:dict(count=len(values),sample_ids=list(values),mean={k:statistics.mean(v[k] for v in values.values())
            for k in next(iter(values.values()))}) for label,values in variants.items()}
    require(context(args)==contract,'Frozen inputs changed during latent stage')
    result=dict(status='latent_completed',contract_sha256=digest,groups=groups,aggregates=aggregates,receipts=receipts,
        expected_compute=contract['expected_compute'],no_pixel_decoder_invoked=True,frozen_inputs_and_sources_unchanged=True,
        interpretation='Only compare models inside each target-resolution group; different-grid latent MSE is not a shared resolution-quality score.',
        sample_scope='Post-hoc four-dev paired diagnosis; the information-matched comparison is restricted to two fixed dev clips.')
    immutable_json(args.output/'latent_summary.json',result);print(json.dumps(dict(stage='latent_completed',aggregates=aggregates)),flush=True)
    return result


def raw_reference384(contract,ident):
    rec=contract['caches'][ident]['reference_rgb'];require(record(rec['path'])==rec,'Reference RGB cache changed')
    value=tensor_load(rec['path']);require(value['sample_id']==ident and value['split']=='dev' and
        value['pair_sha256']==contract['pairs'][ident]['pair']['sha256'],'Raw reference RGB cache identity differs')
    rgb=value['reference'];require(tuple(rgb.shape)==(120,384,672,3) and rgb.dtype==torch.uint8,'Reference RGB cache shape/dtype differs')
    return rgb


def native_crop_sheet(source,output,path,ident,label):
    x0,y0,x1,y1=(2*v for v in CROP_BOXES[ident]);w=x1-x0;h=y1-y0
    sheet=Image.new('RGB',(2*w,(h+28)*len(CROP_FRAMES)),'#111111');draw=ImageDraw.Draw(sheet)
    for row,index in enumerate(CROP_FRAMES):
        for col,(name,value) in enumerate((('raw source768',source),(label,output))):
            sheet.paste(Image.fromarray(value[index,y0:y1,x0:x1].numpy()),(col*w,row*(h+28)+28))
            draw.text((col*w+6,row*(h+28)+7),f'{name} | frame {index}',fill='white')
    path.parent.mkdir(parents=True,exist_ok=True);sheet.save(path)
    return dict(crop_native=[x0,y0,x1,y1],frames=list(CROP_FRAMES),no_resize=True,**file_record(path))


def panel_media(outputs,folder):
    folder.mkdir();comparisons(outputs,folder,fps=24.,crf=16,comparison_video=True)
    audit=audit_video(folder/'comparison.mp4',672*len(outputs),412)
    with Image.open(folder/'comparison.png') as im:
        require(im.size==(672*len(outputs),1236) and im.format=='PNG','Panel PNG dimensions differ');im.verify()
    return dict(labels=list(outputs),mp4=audit,png=file_record(folder/'comparison.png'))


@torch.no_grad()
def pixels(args,contract,digest,device):
    require((args.output/'latent_summary.json').exists(),'Complete latent stage before pixels')
    verify_latent_summary(args,contract,digest)
    todo=[ident for ident in PIXEL_IDS if not (args.output/'pixels'/ident/'done.json').exists()]
    if not todo:return finalize_pixels(args,contract,digest)
    vae=load_video_vae(args.base_model,device).eval().requires_grad_(False);perceptual,lpips_id=load_lpips(device)
    require(sha256(inspect.getfile(type(vae)))==contract['decoder_fingerprint']['implementation_sha256'],'Decoder implementation differs')
    for actual,key in ((lpips_id['backbone']['sha256'],'backbone_sha256'),(lpips_id['linear_weights']['sha256'],'linear_sha256'),
        (lpips_id['implementation_sha256'],'implementation_sha256')):require(actual==contract['lpips_fingerprint'][key],'LPIPS identity differs')
    check_frozen(vae,perceptual);before=frozen_versions(vae,perceptual);mode=precision(device,contract)
    for ident in todo:
        directory=args.output/'pixels'/ident;archive_incomplete(directory,args.output,'Incomplete two-clip pixel diagnosis retained');directory.mkdir(parents=True)
        pair=load_pair(contract,ident);source=fresh_native(args,contract,ident);target=pair['source_pixels']
        low=resize_rgb(target,(192,336),'area2');rgb_bicubic=resize_rgb(low,(384,672),'bicubic')
        common={'raw_source384':target,'rgb_bicubic192_to384':rgb_bicubic};native_metrics={};crop_files={};decode_info={}
        rlow=decode_latent(vae,pair['draft'],device,keep_frames=120)
        require(tuple(rlow.shape)==(120,192,336,3),'LR roundtrip geometry differs')
        common['lr_vae192_roundtrip_bicubic']=resize_rgb(rlow,(384,672),'bicubic')
        lr_roundtrip=simple_metrics(rlow,low)
        common['vae384_roundtrip']=raw_reference384(contract,ident)
        decode_info['vae384_roundtrip']=dict(reused_raw_cache=contract['caches'][ident]['reference_rgb'])
        native_raw_bicubic=resize_rgb(target,(768,1344),'bicubic')
        native_metrics['rgb_bicubic384_to768']=simple_metrics(native_raw_bicubic,source)
        crop_files['rgb_bicubic384_to768']=native_crop_sheet(source,native_raw_bicubic,directory/'native_crops/rgb_bicubic384_to768.png',ident,'RGB bicubic384->768')
        del native_raw_bicubic,rlow
        todo_nodes=(('vae768_roundtrip_down','encodings','native768'),
            ('teacher192_to384','old','teacher'),('motion192_to384','old','motion'),
            ('teacher384_to768_down','predictions','teacher_native'),('motion384_to768_down','predictions','motion_native'),
            ('teacher_bicubic192_to384_to768_down','predictions','teacher_information_matched'))
        for name,group,label in todo_nodes:
            if group=='old':z=old_prediction(contract,label,ident);origin=contract['caches'][ident][label]
            else:z,origin=load_new(args,digest,group,label,ident)
            start=time.perf_counter();rgb=decode_latent(vae,z,device,keep_frames=120);synchronize(device)
            decode_info[name]=dict(seconds=time.perf_counter()-start,origin=origin,raw_rgb=pixel_record(rgb))
            if rgb.shape[1:3]==(768,1344):
                native_metrics[name]=simple_metrics(rgb,source)
                crop_files[name]=native_crop_sheet(source,rgb,directory/'native_crops'/f'{name}.png',ident,name)
                common[name]=resize_rgb(rgb,(384,672),'area2')
            else:
                require(tuple(rgb.shape)==(120,384,672,3),'Low-branch decoded geometry differs');common[name]=rgb
            del rgb,z
        require(set(common)==set(COMMON_LABELS),'Incomplete common384 diagnostic nodes')
        # All LPIPS and motion masks operate on this ONE raw384 target, including VAE paths.
        measurements={label:rgb_metrics(common[label],target,perceptual,device,contract['frame_regions']) for label in COMMON_LABELS if label!='raw_source384'}
        transitions={label:rgb_transition_metrics(common[label],target,contract['transition_pair_groups']) for label in COMMON_LABELS if label!='raw_source384'}
        for region in contract['frame_regions']:
            require(len({v['regions'][region]['motion_roi_fraction'] for v in measurements.values()})==1,'Common target ROI differs across nodes')
        panel_a={'raw source384':target,'RGB bicubic LR':common['rgb_bicubic192_to384'],
            'LR VAE roundtrip':common['lr_vae192_roundtrip_bicubic'],'HR384 VAE roundtrip':common['vae384_roundtrip']}
        panel_b={'raw source384':target,'teacher192->384':common['teacher192_to384'],
            'teacher384->768 down':common['teacher384_to768_down'],'motion192->384':common['motion192_to384'],
            'motion384->768 down':common['motion384_to768_down']}
        media={'codec_chain':panel_media(panel_a,directory/'codec_chain'),'resolution_outputs':panel_media(panel_b,directory/'resolution_outputs')}
        # Information-matched output is visible in native crops plus this common-domain still sheet.
        info_dir=directory/'information_control';info_dir.mkdir()
        comparisons({'raw source384':target,'same-information high grid':common['teacher_bicubic192_to384_to768_down'],
            'real384 information high grid':common['teacher384_to768_down']},info_dir,fps=24.,crf=16,comparison_video=False)
        entry=dict(sample_id=ident,contract_sha256=digest,frames=120,common384=measurements,
            common_transition_groups=transitions,native768_auxiliary=native_metrics,lr192_roundtrip_native=lr_roundtrip,
            common_raw_rgb_hashes={k:pixel_record(v) for k,v in common.items()},native_crops=crop_files,
            decode_info=decode_info,media_audit=media,common_target=contract['common_metric_target'],native_target=contract['native_metric_target'])
        immutable_json(directory/'metrics.json',entry)
        check_frozen(vae,perceptual);require(frozen_versions(vae,perceptual)==before,'Frozen VAE/LPIPS state changed')
        files={str(path.relative_to(directory)):file_record(path) for path in sorted(directory.rglob('*')) if path.is_file()}
        immutable_json(directory/'done.json',dict(status='completed',sample_id=ident,contract_sha256=digest,files=files,
            frozen_modules_unchanged=True,execution_precision_start=mode,execution_precision_end=precision(device,contract),
            full_decoded_mp4_files=2,decoder_fingerprint=contract['decoder_fingerprint'],lpips_identity=lpips_id))
        print(json.dumps(dict(stage='pixel_completed',sample_id=ident,new_decodes=7,reused_raw_reference=1,mp4_files=2)),flush=True)
        del pair,source,target,low,common,panel_a,panel_b
    del vae,perceptual;empty_cache(device)
    return finalize_pixels(args,contract,digest)


def finalize_pixels(args,contract,digest):
    check_prepared(args,contract,digest);verify_latent_summary(args,contract,digest)
    samples={};receipts={};media_count=0
    for ident in PIXEL_IDS:
        folder=args.output/'pixels'/ident;rec=read_json(folder/'done.json')
        require(rec['status']=='completed' and rec['sample_id']==ident and rec['contract_sha256']==digest and rec['frozen_modules_unchanged'] and
            rec['execution_precision_start']==rec['execution_precision_end']==contract['execution_precision'],'Pixel receipt identity/freeze differs')
        require(rec['decoder_fingerprint']==contract['decoder_fingerprint'] and rec['full_decoded_mp4_files']==2,
            'Pixel decoder identity/media count differs')
        for name,key in (('backbone','backbone_sha256'),('linear_weights','linear_sha256')):
            require(rec['lpips_identity'][name]['sha256']==contract['lpips_fingerprint'][key],'Pixel LPIPS identity differs')
        require(rec['lpips_identity']['implementation_sha256']==contract['lpips_fingerprint']['implementation_sha256'],
            'Pixel LPIPS implementation differs')
        expected_files={'metrics.json','information_control/comparison.png'}
        for panel in ('codec_chain','resolution_outputs'):
            expected_files.update({panel+'/comparison.png',panel+'/comparison.mp4'})
        native_crops=('rgb_bicubic384_to768','vae768_roundtrip_down','teacher384_to768_down','motion384_to768_down',
            'teacher_bicubic192_to384_to768_down')
        expected_files.update('native_crops/'+label+'.png' for label in native_crops)
        require(set(rec['files'])==expected_files,'Pixel receipt required file set differs')
        for name,expected in rec['files'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts and file_record(folder/name)==expected,'Pixel artifact changed')
        value=read_json(folder/'metrics.json');require(value['sample_id']==ident and value['contract_sha256']==digest,'Pixel metric identity differs')
        require(set(value['common384'])==set(COMMON_LABELS)-{'raw_source384'} and
            set(value['media_audit'])=={'codec_chain','resolution_outputs'} and value['frames']==120,'Pixel metric/media scope differs')
        origins={'vae384_roundtrip':dict(reused_raw_cache=contract['caches'][ident]['reference_rgb'])}
        require(value['decode_info']['vae384_roundtrip']==origins['vae384_roundtrip'],'Reused raw reference identity differs')
        expected_decode_labels=set(origins)
        for name,group,label in (('vae768_roundtrip_down','encodings','native768'),
            ('teacher192_to384','old','teacher'),('motion192_to384','old','motion'),
            ('teacher384_to768_down','predictions','teacher_native'),('motion384_to768_down','predictions','motion_native'),
            ('teacher_bicubic192_to384_to768_down','predictions','teacher_information_matched')):
            origin=contract['caches'][ident][label] if group=='old' else record(args.output/group/label/ident/'done.json')
            require(value['decode_info'][name]['origin']==origin,'Pixel decode input receipt changed')
            expected_decode_labels.add(name)
        require(set(value['decode_info'])==expected_decode_labels,'Pixel decode origin set differs')
        for audit in value['media_audit'].values():
            require(audit['mp4']['passed'] and audit['mp4']['decoded_frames']==120 and audit['mp4']['fps']==24 and audit['mp4']['duration_seconds']==5,'Comparison media timing failed')
            media_count+=1
        samples[ident]=value;receipts[ident]=record(folder/'done.json')
    labels=[x for x in COMMON_LABELS if x!='raw_source384'];means={}
    for region in contract['frame_regions']:
        keys=samples[PIXEL_IDS[0]]['common384'][labels[0]]['regions'][region]
        means[region]={label:{k:statistics.mean(v['common384'][label]['regions'][region][k] for v in samples.values()) for k in keys} for label in labels}
    require(media_count==4,'Expected four fully decoded comparison videos')
    require(context(args)==contract,'Final frozen source/data/assets changed')
    result=dict(status='completed',contract_sha256=digest,latent_summary=record(args.output/'latent_summary.json'),
        prepare_summary=record(args.output/'prepare.json'),pixel_sample_ids=list(PIXEL_IDS),pixel_samples=samples,
        common384_equal_clip_means=means,full_decoded_mp4_files=media_count,receipts=receipts,
        frozen_inputs_and_sources_unchanged=True,no_training=True,no_saved_model_weights=True,
        limitations=contract['limitations'],native_metric_scope='Native768 numbers remain separate; no cross-grid ranking or native LPIPS claim.')
    immutable_json(args.output/'summary.json',result);print(json.dumps(dict(stage='completed',pixel_clips=2,full120_mp4_files=4)),flush=True)
    return result


def self_test():
    """CPU-only pure helpers, no real assets or accelerator model load."""
    torch.set_num_threads(2);torch.manual_seed(768)
    x=torch.randint(0,256,(120,8,12,3),dtype=torch.uint8)
    y=resize_rgb(x,(4,6),'area2')
    expected=x.reshape(120,4,2,6,2,3).float().mean((2,4)).round().byte()
    require(torch.equal(y,expected),'Area spatial alignment differs')
    up=resize_rgb(y,(8,12),'bicubic');require(up.shape==x.shape,'Bicubic geometry differs')
    altered=x.clone();altered[60:]=0
    require(torch.equal(resize_rgb(altered,(4,6),'area2')[:60],y[:60]),'Spatial resize mixes time')
    r=simple_metrics(x,x);require(r['mse']==r['delta_l1']==r['temporal_difference_mse']==0,'Identity metric failed')
    r=simple_metrics(up,x);a=up.float()/255.;b=x.float()/255.
    require(abs(r['temporal_difference_mse']-float(((a[1:]-a[:-1])-(b[1:]-b[:-1])).double().square().mean()))<1e-7,'119-pair temporal coverage failed')
    require(4+2==6 and sum(len(v) for v in PREDICTIONS.values())==10 and len(PIXEL_IDS)==2,'Expected compute changed')
    print(json.dumps(dict(status='passed',scope='CPU spatial operators/metric coverage only; no real-source/model test',
        new_encodes=6,upscaler_forwards=10,pixel_clips=2)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,parents=[_boot])
    p.add_argument('--stage',choices=('check','prepare','latent','pixels','finalize','all','self-test'),default='check')
    for name in ('upstream','weights','pairs','base-model','history','teacher-cache','output'):p.add_argument('--'+name,type=Path)
    p.add_argument('--device',default='npu:0');args=p.parse_args();args.root=args.root.resolve()
    require(args.root==_early.root.resolve(),'Import root differs from arguments')
    if args.stage=='self-test':self_test();return
    home=args.root/'upscaler_v5_motion_20261106'
    args.upstream=args.upstream or args.root/'upstream';args.weights=args.weights or args.root/'assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors'
    args.pairs=args.pairs or args.root/'upscaler_ultraflash/pairs';args.base_model=args.base_model or Path('${H3_BASE_MODEL}')
    args.history=args.history or home/'evaluation';args.teacher_cache=args.teacher_cache or args.root/'upscaler_hybrid/teacher_cache'
    args.output=args.output or home/'diagnostics/resolution_chain_v2';torch.set_num_threads(4)
    contract,digest=contract_for(args)
    if args.stage=='check':
        prepared=check_prepared(args,contract,digest) if (args.output/'prepare.json').exists() else None
        print(json.dumps(dict(status='checked',contract_sha256=digest,prepared=prepared is not None,
            no_select_device_or_model_load=True,expected_compute=contract['expected_compute'])),flush=True);return
    if args.stage in ('prepare','all'):prepare(args,contract,digest)
    if args.stage=='prepare':return
    check_prepared(args,contract,digest)
    if args.stage=='finalize':finalize_pixels(args,contract,digest);return
    device=select_device(args.device);precision(device,contract)
    if args.stage in ('latent','all'):
        encode_new(args,contract,digest,device);infer_new(args,contract,digest,device);summarize_latents(args,contract,digest)
    if args.stage in ('pixels','all'):pixels(args,contract,digest,device)
    require(context(args)==contract,'Post-stage frozen inputs/sources changed')


if __name__=='__main__':main()
