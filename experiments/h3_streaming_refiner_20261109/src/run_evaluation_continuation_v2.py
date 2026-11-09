"""Continue the frozen cohort after streaming and native-dense previews.

Requires all owned predecessor launchers to have exited and reaped children.
Reuses only SHA-verified completed results. Never adopts a partial directory,
waits for a device, starts training, or selects an accepted refiner. Run inside
the existing three-lock launcher. Streaming/pixel workers remain evaluation_v4;
the separately pinned native-attention dense worker replaces only that worker.
"""
import argparse, hashlib, json, statistics, subprocess, sys, time, traceback
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def receipt(path):return {'path':str(path),'sha256':sha(path)}

def read_receipt(r):
    assert sha(r['path'])==r['sha256'],'Completed result changed'
    return read(r['path'])

def validate_preflight(r,step,checkpoint_sha,sample_id):
    assert r['status']=='two_latent_gates_only_passed'
    assert r['training_step']==step and r['checkpoint_sha256']==checkpoint_sha and r['sample_id']==sample_id
    assert r['optimizer_updates']==0 and not r['full67_inference_complete'] and not r['full226_pixels_complete']
    assert set(r['gates'])=={'cache_parity','future','condition_future','reload'}
    for name,value in r['gates'].items():
        assert 0<=value['max_abs']<=(.03 if name=='cache_parity' else 0)
        assert 0<=value['relative_rms']<=(.005 if name=='cache_parity' else 0)

def validate_latent(r,step,checkpoint_sha,sample_id,data_sha,protocol_sha):
    assert r['status']=='full67_latent_inference_complete_pixels_pending'
    assert (r['training_step'],r['checkpoint_sha256'],r['sample_id'])==(step,checkpoint_sha,sample_id)
    assert r['data_manifest_sha256']==data_sha and r['protocol_sha256']==protocol_sha
    assert r['output']['shape']==[24,67,68,120] and r['pipeline_shape']==[1,24,67,68,120]
    assert sha(r['output']['path'])==r['output']['sha256']
    assert r['trained_gates_passed'] and r['model_forwards_full67']==268 and r['actual_test_optimizer_updates']==0
    assert r['chunk_invariance_first4']['max_abs']==0
    trace=r['cache_bytes_trace'];assert len(trace)==67 and len(set(trace[1:]))==1 and trace[1]>0

def validate_pixels(r,step,checkpoint_sha,sample_id,data_sha,prediction_sha):
    assert r['status']=='full226_nativeHR_metrics_complete_visual_review_pending'
    assert (r['training_step'],r['checkpoint_sha256'],r['sample_id'])==(step,checkpoint_sha,sample_id)
    assert r['arm']==f'strict_streaming_step{step}'
    assert r['data_manifest_sha256']==data_sha and r['prediction_sha256']==prediction_sha
    assert r['RGB']['shape']==[226,1080,1920,3] and sha(r['RGB']['path'])==r['RGB']['sha256']
    assert r['raw_decode_shape']==[226,1088,1920,3] and r['crop_y']==[4,1084] and r['fps']==24
    assert r['nativeHR_metrics']['regions']['full226']['frame_count']==226 and r['optimizer_updates']==0
    assert r['decoder']=='frozen_official_overlap_provisional_for_quality_comparison'
    assert not r['strict_streaming_decoder_validated'] and not r['visual_quality_passed']

def validate_stage_transition(original,native):
    for key in ('frozen_protocol_sha256','dev_manifest_sha256','prior_decoder_plan_sha256'):
        assert original[key]==native[key],f'Frozen input changed: {key}'
    assert set(original['new_sources'])==set(native['new_sources'])
    changed={k for k in original['new_sources'] if original['new_sources'][k]!=native['new_sources'][k]}
    assert changed=={'src/evaluate_dense_reference.py'},changed
    assert native['new_sources']['src/evaluate_dense_reference.py']=='8cef445f84c33c5f5e03e3d80deb576a5609c87c0ed06ced30b153a6f0b7a3b4'

def validate_dense_latent(r,sample_id,data_sha,protocol_sha,gate):
    assert r['status']=='full67_latent_inference_complete_pixels_pending'
    assert r['arm']=='released_dense_refiner' and r['sample_id']==sample_id
    assert r['training_step'] is None and r['checkpoint_sha256'] is None
    assert not r['trained_streaming_adapter_loaded'] and r['same_draft_noise_condition_sigmas']
    assert r['data_manifest_sha256']==data_sha and r['protocol_sha256']==protocol_sha
    assert r['output']['shape']==[24,67,68,120] and r['pipeline_shape']==[1,24,67,68,120]
    assert sha(r['output']['path'])==r['output']['sha256']
    assert r['optimizer_updates']==0 and r['model_forwards']==3
    assert gate['scope']=='full_model_two_latent_dense_vs_upstream_forward' and gate['no_text']
    assert 0<=gate['max_abs']<=.03 and 0<=gate['relative_rms']<=.005

def validate_dense_pixels(r,sample_id,data_sha,prediction_sha):
    assert r['arm']=='released_dense_refiner'
    # Apply identical pixel/payload validation without changing the source result.
    candidate=dict(r);candidate['arm']='strict_streaming_stepNone'
    validate_pixels(candidate,None,None,sample_id,data_sha,prediction_sha)

def main():
    p=argparse.ArgumentParser()
    for n in ('stage','root','prior-stage','training','training-launch','data-manifest','assets-manifest',
              'assets-manifest-sha','prior-batch','prior-launch','preview','preview-launch',
              'dense-stage','dense-preview','dense-launch','failed-continuation-launch','output'):
        p.add_argument('--'+n,required=True)
    p.add_argument('--stream-resident',type=int,default=24)
    p.add_argument('--admission-only',action='store_true')
    a=p.parse_args();stage=Path(a.stage);train=Path(a.training);prior=Path(a.prior_batch);preview=Path(a.preview)
    assert 0<=a.stream_resident<=40
    source_path=stage/'config/evaluation_workers_v4.json';source=read(source_path)
    assert all(sha(stage/n)==v for n,v in source['new_sources'].items())
    dense_stage=Path(a.dense_stage);dense_preview=Path(a.dense_preview)
    native_path=dense_stage/'config/evaluation_workers_v4.json';native=read(native_path)
    validate_stage_transition(source,native)
    assert all(sha(dense_stage/n)==v for n,v in native['new_sources'].items())
    assert sha(dense_stage/'config/evaluation_v1.json')==native['frozen_protocol_sha256']
    protocol=stage/'config/evaluation_v1.json';assert sha(protocol)==source['frozen_protocol_sha256']
    policy=read(protocol);data_sha=source['dev_manifest_sha256'];assert sha(a.data_manifest)==data_sha
    assert sha(a.assets_manifest)==a.assets_manifest_sha
    ts=read(train/'status.json');tl=read(a.training_launch)
    assert ts['optimizer_updates']==200 and ts['status']=='pilot_training_complete_evaluation_pending'
    assert tl['status']=='complete' and tl['exit_code']==0 and tl['child_reaped']
    pl=read(a.prior_launch);vl=read(a.preview_launch)
    assert pl['child_reaped'] and pl['exit_code']==-15,'Only the explicitly interrupted evaluation is accepted'
    assert vl['child_reaped'] and vl['status']=='complete' and vl['exit_code']==0
    dl=read(a.dense_launch);fl=read(a.failed_continuation_launch)
    assert dl['child_reaped'] and dl['status']=='complete' and dl['exit_code']==0
    assert fl['child_reaped'] and fl['status']=='failed' and fl['exit_code']==1
    old=read(prior/'inputs.json');assert old['source_manifest_sha256']==sha(source_path) and old['dev_manifest_sha256']==data_sha
    checkpoint_hashes={s:sha(train/f'adapter_step{s:04d}.pt') for s in policy['checkpoints']}
    assert old['checkpoint_sha256']=={str(s):v for s,v in checkpoint_hashes.items()}
    previous=read(prior/'progress.json')['completed'];assert len(previous)==5
    reused=[]
    for step,r in zip(reversed(policy['checkpoints']),previous):
        validate_preflight(read_receipt(r),step,checkpoint_hashes[step],policy['dev'][0]);reused.append(r)
    vr=read(preview/'result.json');assert vr['status']=='full226_engineering_preview_pixels_complete'
    validate_latent(read_receipt(vr['latent']),200,checkpoint_hashes[200],policy['dev'][0],data_sha,sha(protocol))
    validate_pixels(read_receipt(vr['pixels']),200,checkpoint_hashes[200],policy['dev'][0],data_sha,vr['latent']['sha256'])
    reused.extend((vr['latent'],vr['pixels']))
    dr=read(dense_preview/'result.json')
    assert dr['status']=='full226_dense_reference_complete_visual_review_pending'
    assert dr['source_manifest_sha256']==sha(native_path) and dr['optimizer_updates']==0
    dense_gate=receipt(dense_preview/'latent/upstream_gate.json')
    validate_dense_latent(read_receipt(dr['latent']),policy['dev'][0],data_sha,sha(protocol),read_receipt(dense_gate))
    validate_dense_pixels(read_receipt(dr['pixels']),policy['dev'][0],data_sha,dr['latent']['sha256'])
    reused.extend((dr['latent'],dr['pixels'],dense_gate))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.time();completed=[]
    def write(name,value):
        tmp=out/(name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(out/name)
    write('inputs.json',{'source_manifest_sha256':sha(source_path),'orchestrator_sha256':sha(__file__),
          'checkpoint_sha256':checkpoint_hashes,'protocol_sha256':sha(protocol),'dev_manifest_sha256':data_sha,
          'reused_completed_receipts':reused,'partial_step0_directory_untouched':True,
          'execution_order_change':'reuse complete step200 and first dense WingIt; evaluate remaining fixed cohort',
          'native_dense_manifest_sha256':sha(native_path),'stream_resident_layers':a.stream_resident,
          'residency_changes_only_paging_each_stream_job_repeats_real_gates':True,
          'selection_rule_unchanged':True,'training_updates':200})
    if a.admission_only:
        write('result.json',{'status':'CPU_reuse_admission_passed_no_workers_started',
            'reused_receipts':len(reused),'optimizer_updates':0,'new_accelerator_forwards':0})
        return
    def run(worker,args,folder,worker_stage=None):
        print(json.dumps({'event':'start_worker','worker':worker,'output':str(folder)}),flush=True)
        with (out/(folder.name+'.log')).open('x') as log:
            subprocess.run([sys.executable,str((worker_stage or stage)/'src'/worker),*args,'--output',str(folder)],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        r=receipt(folder/'result.json');completed.append(r)
        write('progress.json',{'completed':completed,'reused':reused,'elapsed_seconds':time.time()-started})
        return r
    common=['--root',a.root,'--data-manifest',a.data_manifest,'--data-manifest-sha',data_sha,
            '--protocol',str(protocol),'--assets-manifest',a.assets_manifest,'--assets-manifest-sha',a.assets_manifest_sha]
    pixel_common=['--prior-stage',a.prior_stage,'--prior-plan-sha',source['prior_decoder_plan_sha256'],
                  '--data-manifest',a.data_manifest,'--data-manifest-sha',data_sha]
    pixel_results=[vr['pixels'],dr['pixels']]
    jobs=[(s,sid) for s in policy['checkpoints'] for sid in policy['dev']
          if (s,sid)!=(200,policy['dev'][0])]+[(None,sid) for sid in policy['dev'][1:]]
    assert len(jobs)==16 and len(set(jobs))==16
    try:
        for step,sid in jobs:
            stem=('dense' if step is None else f'step{step:04d}')+'_'+sid
            extra=[] if step is None else ['--checkpoint',str(train/f'adapter_step{step:04d}.pt'),
                                          '--checkpoint-sha',checkpoint_hashes[step],
                                          '--resident',str(a.stream_resident)]
            latent=run('evaluate_dense_reference.py' if step is None else 'evaluate_streaming.py',
                       common+extra+['--sample-id',sid],out/(stem+'_latent'),
                       dense_stage if step is None else stage)
            pixel_results.append(run('evaluate_pixels.py',pixel_common+['--prediction',latent['path'],
                '--prediction-sha',latent['sha256']],out/(stem+'_pixels')))
        records=[read_receipt(r) for r in pixel_results];assert len(records)==18
        aggregates={}
        for label in [f'strict_streaming_step{s}' for s in policy['checkpoints']]+['released_dense_refiner']:
            rs=[r for r in records if r['arm']==label]
            assert len(rs)==3 and set(r['sample_id'] for r in rs)==set(policy['dev'])
            mean={k:statistics.mean(r['nativeHR_metrics']['regions']['full226'][k] for r in rs)
                  for k in ('pixel_l1','lpips','psnr_db','delta_l1')}
            base={k:statistics.mean(r['no_refiner_nativeHR_metrics']['regions']['full226'][k] for r in rs)
                  for k in ('pixel_l1','lpips')}
            ratios={k:mean[k]/base[k] for k in base}
            aggregates[label]={'equal_clip':mean,'ratios_vs_no_refiner':ratios,
                'metric_gate_passed':ratios['pixel_l1']<=policy['gate']['equal_clip_nativeHR_L1_ratio_vs_no_refiner_max']
                 and ratios['lpips']<=policy['gate']['equal_clip_nativeHR_LPIPS_ratio_vs_no_refiner_max']}
        write('result.json',{'status':'all_full226_metrics_complete_visual_and_decoder_acceptance_pending',
            'aggregates':aggregates,'metric_passing_steps':[s for s in policy['checkpoints'] if aggregates[f'strict_streaming_step{s}']['metric_gate_passed']],
            'pixel_results':pixel_results,'accepted_refiner':None,'selection_rule':policy['selection'],
            'strict_decoder_validated':False,'visual_quality_passed':False,'test_unlocked':False,
            'optimizer_updates_this_evaluation':0,'elapsed_seconds':time.time()-started})
    except BaseException as e:
        write('failure.json',{'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),
                             'completed':completed,'reused':reused});raise

if __name__=='__main__':main()
