"""Independent full-pretrained VAE validation. Run only under the three-lock launcher.

--admission-only performs CPU file checks and never imports torch or starts a worker.
The 137-latent stress input loops an authorized67-latent clip; it is not new video data.
"""
import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import subprocess
import time
import traceback
from decoder_validation_admission import admit, checked, sha, verify_parent_locks


def run(args,out,write,log):
    admission,plan,prior,rec=admit(args)
    assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
    admission['torch_imported']=False
    admission['torch_npu_imported']=False
    write('admission.json',admission)
    if args.admission_only:
        print(json.dumps(admission),flush=True)
        return
    if not admission['predecessor']['ready']:
        raise RuntimeError('Predecessor is active or not reaped; no accelerator import performed')
    locks=verify_parent_locks(args.physical_device)
    write('device_lock_owner.json',locks)
    smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True).stdout
    (out/'npu_smi_before_model.txt').write_text(smi)
    if f'No running processes found in NPU {args.physical_device}' not in smi:
        raise RuntimeError('Card became busy before model import; no accelerator work started')
    memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    if memory['MemAvailable']<100*(1<<30):raise RuntimeError('Insufficient available host RAM for fixed FP32 KV')

    import numpy as np
    import torch
    import torch_npu
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.npu.set_device('npu:0');device=torch.device('npu:0')
    torch.npu.conv.allow_hf32=False;torch.npu.matmul.allow_hf32=False
    torch.set_float32_matmul_precision('highest')
    sys.path.insert(0,str(Path(args.candidate_stage)/'src'))
    sys.path.insert(0,str(Path(args.prior_stage)))
    from strict_decoder_core import IncrementalVAECore
    from strict_streaming_decoder import StrictStreamingVideoDecoder
    from data_utils import load_video_vae,latent_stats
    import h3_production67_eval_metrics_v1 as metrics
    assert sha(inspect.getfile(metrics))=='d6f0d6c588930dd00b7536840e0d675bd0f001975728c7e3278aaf3db139b7f0'
    spec=importlib.util.spec_from_file_location('pinned_upstream_decoder',Path(args.candidate_stage)/'vendor/solarwm_streaming_original.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    sample=torch.load(rec['path'],map_location='cpu',weights_only=True)
    draft=sample['upscaled'];assert draft.shape==(24,67,68,120) and torch.isfinite(draft).all()
    del sample
    vae=load_video_vae(prior['base_model'],device)
    assert sha(inspect.getfile(type(vae)))==prior['decoder_implementation_sha256']
    assert len(vae.decoder.transformer_blocks)==36 and vae.decoder.transformer_blocks[0].attn.heads==32
    assert all(p.dtype==torch.float32 for p in vae.parameters())
    mean,std=latent_stats(vae,device)
    log('full_pretrained_VAE_loaded',parameters=sum(p.numel() for p in vae.parameters()))
    candidate=IncrementalVAECore(vae,compile=False,compensate=False,cache_device='cpu',token_budget=2600)
    reference=original.StreamingDecoder(vae,compile=False,compensate=False,token_budget=2600)
    errors=[]
    with torch.inference_mode():
        for i in range(9):
            z=draft[None,:,i:i+1,:16,:16].to(device,torch.float32)*std+mean
            actual=candidate.push(z);expected=reference.push(z)
            error=float((actual.float()-expected.float()).abs().max());errors.append(error)
            log('small_spatial_full_VAE_parity',latent=i,max_abs=error)
            assert error==0,errors
    write('full_VAE_small_spatial_parity.json',{'scope':'full pretrained36layer VAE,16x16 latent crop,nine latents','max_abs':max(errors),'by_latent':errors,'reference':'pinned upstream incremental, not official bidirectional overlap'})
    del actual,expected,z,candidate,reference;gc.collect();torch.npu.empty_cache()

    decoder=StrictStreamingVideoDecoder(vae,token_budget=2600)
    rgbpath=out/'first67_visible_RGB.npy'
    rgb=np.lib.format.open_memmap(rgbpath,mode='w+',dtype=np.uint8,shape=(226,1080,1920,3))
    digest=hashlib.sha256();prefix=[];cursor=0;timings=[];audits=[]
    for i in range(137):
        start=time.time()
        part=decoder.push(draft[None,:,i%67:i%67+1],latent_index=i)
        torch.npu.synchronize();elapsed=time.time()-start
        data=part.numpy();payload=data.tobytes();digest.update(payload)
        if i<8:prefix.append(hashlib.sha256(payload).hexdigest())
        if i<67:rgb[cursor:cursor+len(data)]=data
        cursor+=len(data);stats=decoder.cache_stats();timings.append(elapsed);audits.append(stats)
        assert stats['kv_bytes']==plan['host_KV_bytes_expected'] and stats['retained_input_latents']<=7
        assert stats['rope_pattern_count']<=16 and stats['pixel_frames']==cursor
        log('strict_decoder_latent_complete',latent=i,source_latent=i%67,pixel_frames=cursor,seconds=elapsed,cache=stats)
        if i==66:assert cursor==226;rgb.flush()
    assert cursor==464 and decoder.core.slot_passes==190
    rgb.flush();del rgb
    long_report={'status':'full_pretrained_decoder137_stress_complete','latent_count':137,'decoded_frames':464,
        'input':'authorized firstWingIt frozen-upsampled67-latent clip, looped by modulo67',
        'unique_source_latents':67,'host_KV_bytes':audits[-1]['kv_bytes'],
        'max_retained_latents':max(a['retained_input_latents'] for a in audits),
        'slot_forward_count':audits[-1]['slot_forward_count'],'bounded_past_slot_refreshes':audits[-1]['refreshed_past_slots'],
        'all464_RGB_sha256':digest.hexdigest(),'per_latent_seconds':timings,
        'cache_audit_trace':audits,'peak_allocated_NPU_bytes':torch.npu.max_memory_allocated(),
        'full_model_refiner_longstream_validated':False,'optimizer_updates':0}
    write('long_stream.json',long_report)

    decoder.reset();index=0
    for group in [1,4,3]:
        for j in range(group):
            part=decoder.push(draft[None,:,index:index+1],latent_index=index)
            assert hashlib.sha256(part.numpy().tobytes()).hexdigest()==prefix[index]
            index+=1
    assert index==8
    log('full_spatial_reset_chunk_replay_passed',latents=8,RGB_max_abs=0)
    decoder.reset();changed_future=False
    for i in range(8):
        z=draft[None,:,i:i+1].clone()
        if i==7:z*=-5
        part=decoder.push(z,latent_index=i)
        d=hashlib.sha256(part.numpy().tobytes()).hexdigest()
        if i<7:assert d==prefix[i]
        else:changed_future=d!=prefix[i]
    assert changed_future,'Perturbation was ineffective'
    write('full_spatial_gates.json',{'first8_reset_replay_RGB_max_abs':0,'future_prefix_first7_RGB_max_abs':0,'perturbed_eighth_output_changed':True,'scope':'full68x120 latent /1080p pixels; no continuous visual acceptance'})
    del decoder,vae,part,z,mean,std;gc.collect();torch.npu.empty_cache()

    rgb=torch.from_numpy(np.load(rgbpath,allow_pickle=False))
    native=torch.from_numpy(np.load(rec['rgb']['native_hr']['path'],allow_pickle=False))
    decoded_hr=torch.from_numpy(np.load(rec['rgb']['decoded_hr']['path'],allow_pickle=False))
    official=np.load(rec['rgb']['teacher_student']['path'],mmap_mode='r',allow_pickle=False)
    assert rgb.shape==native.shape==decoded_hr.shape==(226,1080,1920,3)
    import lpips
    fp=prior['lpips_fingerprint']
    assert sha(Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth')==fp['backbone_sha256']
    assert sha(Path(lpips.__file__).resolve().parent/'weights/v0.1/alex.pth')==fp['linear_sha256']
    perceptual=lpips.LPIPS(net='alex',version='0.1').to(device).eval().requires_grad_(False)
    assert sha(inspect.getfile(type(perceptual)))==fp['implementation_sha256']
    values=metrics.rgb_metrics(rgb,native,decoded_hr,perceptual,device)
    base=rec['no_refiner_native_metrics'];current=values['regions']['full226'];old=base['regions']['full226']
    ratios={k:current[k]/max(old[k],1e-12) for k in ('pixel_l1','lpips')}
    official_l1=sum(float(np.abs(rgb[i].numpy().astype(np.float32)-official[i].astype(np.float32)).mean()/255) for i in range(226))/226
    result={'status':'strict_decoder_full_model_technical_and_first_clip_metrics_complete_visual_review_pending',
        'sample_id':rec['sample_id'],'input_arm':'frozen_upsampler_no_refiner_decoder_ablation',
        'data_manifest_sha256':admission['data_manifest_sha256'],'candidate_manifest_sha256':admission['candidate_manifest_sha256'],
        'validation_plan_sha256':args.validation_plan_sha,'CPU_report_sha256':admission['CPU_report_sha256'],
        'RGB':{'path':str(rgbpath),'sha256':sha(rgbpath),'shape':list(rgb.shape)},
        'nativeHR_metrics':values,'official_overlap_no_refiner_nativeHR_metrics':base,
        'full226_ratios_vs_official_no_refiner':ratios,'same_input_official_decoder_pixel_l1':official_l1,
        'first_clip_metric_gate_passed':all(r<=1.02 for r in ratios.values()),
        'long_decoder_latents':137,'long_decoder_pixels':464,'raw_shape_first67':[226,1088,1920,3],
        'crop_y':[4,1084],'fps':24,'compensation':False,'precision':'FP32','KV_backing':'CPU',
        'video_encoding_technical_audit_pending':True,'continuous_visual_reviewed':False,
        'strict_decoder_accepted':False,'selected_refiner':None,'optimizer_updates':0,
        'full_model_refiner_longstream_validated':False,'source_sha256':sha(__file__)}
    write('result.json',result);log('strict_decoder_validation_complete',ratios=ratios,strict_decoder_accepted=False)


def main():
    p=argparse.ArgumentParser()
    for n in ['candidate-stage','validation-plan','validation-plan-sha','cpu-report','prior-stage','data-manifest','predecessor-launch','output']:
        p.add_argument('--'+n,required=True)
    p.add_argument('--physical-device',type=int,default=7)
    p.add_argument('--admission-only',action='store_true')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.time()
    def write(name,x):(out/name).write_text(json.dumps(x,indent=2)+'\n')
    def log(event,**kwargs):
        row={'event':event,'elapsed_seconds':time.time()-start,**kwargs}
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:run(a,out,write,log)
    except BaseException as e:
        write('failure.json',{'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),
            'torch_imported':'torch' in sys.modules,'torch_npu_imported':'torch_npu' in sys.modules,
            'optimizer_updates':0})
        raise


if __name__=='__main__':main()
