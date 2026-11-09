"""Decode one full67 prediction and measure against actual native-HR RGB.

The frozen official overlap decoder is used for comparable quality diagnostics.
It is explicitly not a strict-streaming decoder acceptance test.
"""
import argparse, hashlib, inspect, json, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from prepare_pairs import sha

def checked(desc):
    p=Path(desc['path']);assert sha(p)==desc['sha256']
    if 'bytes' in desc:assert p.stat().st_size==desc['bytes']
    return p

def main():
    p=argparse.ArgumentParser()
    for n in ('prior-stage','prior-plan-sha','prediction','prediction-sha','data-manifest','data-manifest-sha','output'):
        p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.time()
    try:
        assert sha(a.prediction)==a.prediction_sha and sha(a.data_manifest)==a.data_manifest_sha
        pred=json.loads(Path(a.prediction).read_text());data=json.loads(Path(a.data_manifest).read_text())
        assert pred['status']=='full67_latent_inference_complete_pixels_pending'
        assert pred['data_manifest_sha256']==a.data_manifest_sha
        rec=next(r for r in data['samples'] if r['sample_id']==pred['sample_id'])
        stage=Path(a.prior_stage);pp=stage/'h3_production67_limiteddev_full226_plan_v1.json'
        assert sha(pp)==a.prior_plan_sha;plan=json.loads(pp.read_text())
        for name,d in plan['files'].items():assert sha(stage/name)==d['sha256']
        for d in plan['runtime_assets'].values():checked(d)
        sys.path.insert(0,str(stage))
        from data_utils import load_video_vae,decode_latent
        import h3_production67_eval_metrics_v1 as metrics
        assert sha(inspect.getfile(metrics))=='d6f0d6c588930dd00b7536840e0d675bd0f001975728c7e3278aaf3db139b7f0'
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True,warn_only=False)
        import torch_npu
        torch.npu.set_device('npu:0');device=torch.device('npu:0')
        torch.npu.conv.allow_hf32=False;torch.npu.matmul.allow_hf32=False;torch.set_float32_matmul_precision('highest')
        z=np.load(checked(pred['output']),allow_pickle=False)
        assert z.shape==(24,67,68,120) and z.dtype==np.float32 and np.isfinite(z).all()
        native=torch.from_numpy(np.load(checked(rec['rgb']['native_hr']),allow_pickle=False))
        decoded_hr=torch.from_numpy(np.load(checked(rec['rgb']['decoded_hr']),allow_pickle=False))
        vae=load_video_vae(plan['base_model'],device)
        assert sha(inspect.getfile(type(vae)))==plan['decoder_implementation_sha256']
        with torch.no_grad():raw=decode_latent(vae,torch.from_numpy(z),device,keep_frames=None)
        assert raw.shape==(226,1088,1920,3) and raw.dtype==torch.uint8
        rgb=raw[:,4:1084].cpu().contiguous();del raw,vae;torch.npu.empty_cache()
        rgbpath=out/'visible_RGB.npy'
        with rgbpath.open('xb') as f:np.save(f,rgb.numpy(),allow_pickle=False)
        import lpips
        fp=plan['lpips_fingerprint']
        assert sha(Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth')==fp['backbone_sha256']
        assert sha(Path(lpips.__file__).resolve().parent/'weights/v0.1/alex.pth')==fp['linear_sha256']
        perceptual=lpips.LPIPS(net='alex',version='0.1').to(device).eval().requires_grad_(False)
        assert sha(inspect.getfile(type(perceptual)))==fp['implementation_sha256']
        values=metrics.rgb_metrics(rgb,native,decoded_hr,perceptual,device)
        base=rec['no_refiner_native_metrics'];full=values['regions']['full226'];ref=base['regions']['full226']
        ratios={k:full[k]/max(ref[k],1e-12) for k in ('pixel_l1','lpips')}
        report={'status':'full226_nativeHR_metrics_complete_visual_review_pending','sample_id':pred['sample_id'],
                'arm':pred.get('arm','strict_streaming_step'+str(pred['training_step'])),
                'training_step':pred['training_step'],'checkpoint_sha256':pred['checkpoint_sha256'],
                'prediction_sha256':a.prediction_sha,'data_manifest_sha256':a.data_manifest_sha,
                'nativeHR_metrics':values,'no_refiner_nativeHR_metrics':base,'full226_ratios_vs_no_refiner':ratios,
                'RGB':{'path':str(rgbpath),'sha256':sha(rgbpath),'shape':list(rgb.shape)},
                'RGB_pixels_sha256':hashlib.sha256(rgb.numpy().tobytes()).hexdigest(),
                'raw_decode_shape':[226,1088,1920,3],'crop_y':[4,1084],'fps':24,
                'seconds':time.time()-start,'actual_decoder_forwards':1,'optimizer_updates':0,
                'decoder':'frozen_official_overlap_provisional_for_quality_comparison',
                'strict_streaming_decoder_validated':False,'continuous_visual_reviewed':False,'visual_quality_passed':False}
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'event':'pixels_complete','sample_id':pred['sample_id'],'step':pred['training_step'],'ratios':ratios}),flush=True)
    except BaseException as e:
        (out/'failure.json').write_text(json.dumps({'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()},indent=2)+'\n');raise

if __name__=='__main__':main()
