"""Continue the frozen protocol after the user-prioritized step200 preview.

Requires both owned predecessor launchers to have exited and reaped children.
Reuses only SHA-verified completed results. Never adopts a partial directory,
waits for a device, starts training, or selects an accepted refiner. Run inside
the existing three-lock launcher. All model/pixel workers remain evaluation_v4.
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

def main():
    p=argparse.ArgumentParser()
    for n in ('stage','root','prior-stage','training','training-launch','data-manifest','assets-manifest',
              'assets-manifest-sha','prior-batch','prior-launch','preview','preview-launch','output'):
        p.add_argument('--'+n,required=True)
    a=p.parse_args();stage=Path(a.stage);train=Path(a.training);prior=Path(a.prior_batch);preview=Path(a.preview)
    source_path=stage/'config/evaluation_workers_v4.json';source=read(source_path)
    assert all(sha(stage/n)==v for n,v in source['new_sources'].items())
    protocol=stage/'config/evaluation_v1.json';assert sha(protocol)==source['frozen_protocol_sha256']
    policy=read(protocol);data_sha=source['dev_manifest_sha256'];assert sha(a.data_manifest)==data_sha
    assert sha(a.assets_manifest)==a.assets_manifest_sha
    ts=read(train/'status.json');tl=read(a.training_launch)
    assert ts['optimizer_updates']==200 and ts['status']=='pilot_training_complete_evaluation_pending'
    assert tl['status']=='complete' and tl['exit_code']==0 and tl['child_reaped']
    pl=read(a.prior_launch);vl=read(a.preview_launch)
    assert pl['child_reaped'] and pl['exit_code']==-15,'Only the explicitly interrupted evaluation is accepted'
    assert vl['child_reaped'] and vl['status']=='complete' and vl['exit_code']==0
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
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.time();completed=[]
    def write(name,value):
        tmp=out/(name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(out/name)
    write('inputs.json',{'source_manifest_sha256':sha(source_path),'orchestrator_sha256':sha(__file__),
          'checkpoint_sha256':checkpoint_hashes,'protocol_sha256':sha(protocol),'dev_manifest_sha256':data_sha,
          'reused_completed_receipts':reused,'partial_step0_directory_untouched':True,
          'execution_order_change':'first dense WingIt reference for requested comparison, then remaining fixed checkpoint/dev cohort',
          'selection_rule_unchanged':True,'training_updates':200})
    def run(worker,args,folder):
        print(json.dumps({'event':'start_worker','worker':worker,'output':str(folder)}),flush=True)
        with (out/(folder.name+'.log')).open('x') as log:
            subprocess.run([sys.executable,str(stage/'src'/worker),*args,'--output',str(folder)],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        r=receipt(folder/'result.json');completed.append(r)
        write('progress.json',{'completed':completed,'reused':reused,'elapsed_seconds':time.time()-started})
        return r
    common=['--root',a.root,'--data-manifest',a.data_manifest,'--data-manifest-sha',data_sha,
            '--protocol',str(protocol),'--assets-manifest',a.assets_manifest,'--assets-manifest-sha',a.assets_manifest_sha]
    pixel_common=['--prior-stage',a.prior_stage,'--prior-plan-sha',source['prior_decoder_plan_sha256'],
                  '--data-manifest',a.data_manifest,'--data-manifest-sha',data_sha]
    pixel_results=[vr['pixels']]
    jobs=[(None,policy['dev'][0])]+[(s,sid) for s in policy['checkpoints'] for sid in policy['dev']
          if (s,sid)!=(200,policy['dev'][0])]+[(None,sid) for sid in policy['dev'][1:]]
    assert len(jobs)==17 and len(set(jobs))==17
    try:
        for step,sid in jobs:
            stem=('dense' if step is None else f'step{step:04d}')+'_'+sid
            extra=[] if step is None else ['--checkpoint',str(train/f'adapter_step{step:04d}.pt'),
                                          '--checkpoint-sha',checkpoint_hashes[step]]
            latent=run('evaluate_dense_reference.py' if step is None else 'evaluate_streaming.py',
                       common+extra+['--sample-id',sid],out/(stem+'_latent'))
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
