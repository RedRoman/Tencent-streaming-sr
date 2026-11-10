"""One exactstage1000 dev clip under a separate three-lock launcher; no waiter."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback
from anchored_evaluation_admission import admit, sha, read, pinned, write
from decoder_validation_admission import verify_parent_locks
from run_evaluation_continuation_v2 import validate_latent, validate_pixels, validate_dense_latent, validate_dense_pixels

def main():
    p=argparse.ArgumentParser()
    for key in ['stage','manifest-sha','training-stage','training','training-launch','data-manifest','assets-manifest',
                'prior-stage','dense-preview','root','sample-id','output']:
        p.add_argument('--'+key,required=True)
    p.add_argument('--physical-device',type=int,required=True)
    p.add_argument('--admission-only',action='store_true')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    try:
        admission,plan,data=admit(a);write(out/'admission.json',admission)
        if a.admission_only:
            print(json.dumps(admission),flush=True);return
        assert admission['ready'], 'Exact complete stage1000 checkpoint required before NPU use'
        owner=verify_parent_locks(a.physical_device);write(out/'device_lock_owner.json',owner)
        smi=subprocess.check_output(['npu-smi','info'],text=True)
        (out/'npu_smi_before_workers.txt').write_text(smi)
        assert f'No running processes found in NPU {a.physical_device}' in smi
        mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
        assert mem['MemAvailable']>=100*(1<<30)
        stage=Path(a.stage);protocol=stage/'config/evaluation_parent_v1.json'
        data_sha=sha(a.data_manifest);cp=admission['checkpoint']['record'];completed=[]
        binding=out/'checkpoint_binding.json';write(binding,admission)
        dense_latent=dense_pixels=None
        if a.sample_id==plan['sample_order'][0]:
            folder=Path(a.dense_preview);r=pinned(folder/'result.json',plan['dense_result_sha256'])
            assert r['status']=='full226_dense_reference_complete_visual_review_pending' and r['optimizer_updates']==0
            assert r['source_manifest_sha256']==plan['native_dense_parent_manifest_sha256']
            lr=pinned(r['latent']['path'],r['latent']['sha256']);pr=pinned(r['pixels']['path'],r['pixels']['sha256'])
            gate=pinned(folder/'latent/upstream_gate.json',plan['dense_gate_sha256'])
            validate_dense_latent(lr,a.sample_id,data_sha,sha(protocol),gate)
            validate_dense_pixels(pr,a.sample_id,data_sha,r['latent']['sha256'])
            dense_latent,dense_pixels=r['latent'],r['pixels']
            write(out/'dense_reuse.json',{'original':r,'upstream_gate_sha256':plan['dense_gate_sha256'],
                  'same_upsampler_noise_condition_geometry_verified':True,'original_receipts_unchanged':True})
        def run(worker,args,name,cpu=False):
            folder=out/name
            print(json.dumps({'event':'start_worker','worker':worker,'output':str(folder)}),flush=True)
            env=None
            if cpu:
                import os
                env=dict(os.environ,ASCEND_VISIBLE_DEVICES='',ASCEND_RT_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='',TORCH_DEVICE_BACKEND_AUTOLOAD='0')
            with (out/(name+'.log')).open('x') as log:
                subprocess.run([sys.executable,str(stage/'src'/worker),*args,'--output',str(folder)],
                               stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
            item={'path':str(folder/'result.json'),'sha256':sha(folder/'result.json')}
            completed.append(item);write(out/'progress.json',{'completed':completed,'new_optimizer_updates':0})
            return item,read(item['path'])
        common=['--root',a.root,'--data-manifest',a.data_manifest,'--data-manifest-sha',data_sha,
                '--protocol',str(protocol),'--sample-id',a.sample_id]
        latent,lr=run('evaluate_anchored_streaming_v1.py',common+['--checkpoint-binding',str(binding),
                '--checkpoint-binding-sha',sha(binding),'--physical-device',str(a.physical_device)],'candidate_latent')
        validate_latent(lr,3500,cp['adapter_sha256'],a.sample_id,data_sha,sha(protocol))
        pixel_common=['--prior-stage',a.prior_stage,'--prior-plan-sha',plan['prior_decoder_plan_sha256'],
                      '--data-manifest',a.data_manifest,'--data-manifest-sha',data_sha]
        pixel,pr=run('evaluate_pixels.py',pixel_common+['--prediction',latent['path'],'--prediction-sha',latent['sha256']],'candidate_pixels')
        assert pr['arm']=='anchored_stage1000_candidate'
        compatibility=dict(pr,arm='strict_streaming_step3500')
        validate_pixels(compatibility,3500,cp['adapter_sha256'],a.sample_id,data_sha,latent['sha256'])
        if dense_latent is None:
            dense_latent,dl=run('evaluate_dense_reference.py',common+['--assets-manifest',a.assets_manifest,
                    '--assets-manifest-sha',plan['assets_manifest_sha256']],'dense_latent')
            gate=read(out/'dense_latent/upstream_gate.json')
            validate_dense_latent(dl,a.sample_id,data_sha,sha(protocol),gate)
            dense_pixels,dp=run('evaluate_pixels.py',pixel_common+['--prediction',dense_latent['path'],
                    '--prediction-sha',dense_latent['sha256']],'dense_pixels')
            validate_dense_pixels(dp,a.sample_id,data_sha,dense_latent['sha256'])
        video,vr=run('encode_anchored_comparison_cpu_v1.py',['--data-manifest',a.data_manifest,
                    '--data-manifest-sha',data_sha,'--candidate-pixels',pixel['path'],'--candidate-pixels-sha',pixel['sha256'],
                    '--dense-pixels',dense_pixels['path'],'--dense-pixels-sha',dense_pixels['sha256'],
                    '--checkpoint-sha',cp['adapter_sha256'],'--sample-id',a.sample_id],'videos',cpu=True)
        assert vr['status']=='full226_three_videos_technical_audit_passed'
        write(out/'result.json',{'status':'anchored_stage1000_full226_video_ready_visual_review_pending',
              'sample_id':a.sample_id,'stage_step':1000,'total_training_step':3500,'checkpoint':admission['checkpoint'],
              'latent':latent,'pixels':pixel,'dense_latent':dense_latent,'dense_pixels':dense_pixels,'video':video,
              'full226_ratios_vs_no_refiner':pr['full226_ratios_vs_no_refiner'],
              'continuous_visual_reviewed':False,'strict_streaming_decoder_validated':False,
              'accepted_refiner':None,'optimizer_updates':0,'training_mutated':False})
    except BaseException as error:
        write(out/'failure.json',{'type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),'optimizer_updates':0})
        raise

if __name__=='__main__':main()
