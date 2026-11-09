"""Finite serialized evaluation, inside one owned three-lock launcher.

Never waits for training or a device. Refuses incomplete training and existing
output. Stops at the first execution failure and retains all completed results.
"""
import argparse,json,statistics,subprocess,sys,time,traceback
from pathlib import Path
from prepare_pairs import sha

def main():
    p=argparse.ArgumentParser()
    for n in ('root','prior-stage','training','training-launch','data-manifest','assets-manifest','assets-manifest-sha','source-manifest','output'):
        p.add_argument('--'+n,required=True)
    a=p.parse_args();stage=Path(__file__).resolve().parent.parent
    source=json.loads(Path(a.source_manifest).read_text())
    assert all(sha(stage/n)==v for n,v in source['new_sources'].items())
    protocol=stage/'config/evaluation_v1.json';assert sha(protocol)==source['frozen_protocol_sha256']
    policy=json.loads(protocol.read_text());train=Path(a.training)
    status=json.loads((train/'status.json').read_text());launch=json.loads(Path(a.training_launch).read_text())
    assert status['status']=='pilot_training_complete_evaluation_pending' and status['optimizer_updates']==200
    assert launch['status']=='complete' and launch['exit_code']==0 and launch['child_reaped']
    assert not (train/'failure.json').exists()
    assert sha(a.data_manifest)==source['dev_manifest_sha256']
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.time();completed=[]
    checkpoint_hashes={step:sha(train/f'adapter_step{step:04d}.pt') for step in policy['checkpoints']}
    def write(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
    write('inputs.json',{'checkpoint_sha256':checkpoint_hashes,'source_manifest_sha256':sha(a.source_manifest),
                       'dev_manifest_sha256':source['dev_manifest_sha256'],'training_updates':200})
    def run(worker,args,folder):
        cmd=[sys.executable,str(stage/'src'/worker),*args,'--output',str(folder)]
        print(json.dumps({'event':'start_worker','worker':worker,'output':str(folder)}),flush=True)
        with (out/(folder.name+'.log')).open('x') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        receipt={'path':str(folder/'result.json'),'sha256':sha(folder/'result.json')};completed.append(receipt)
        write('progress.json',{'completed':completed,'elapsed_seconds':time.time()-start})
        return receipt
    common=['--root',a.root,'--data-manifest',a.data_manifest,'--data-manifest-sha',source['dev_manifest_sha256'],
            '--protocol',str(protocol),'--assets-manifest',a.assets_manifest,'--assets-manifest-sha',a.assets_manifest_sha]
    pixel_common=['--prior-stage',a.prior_stage,'--prior-plan-sha',source['prior_decoder_plan_sha256'],
                  '--data-manifest',a.data_manifest,'--data-manifest-sha',source['dev_manifest_sha256']]
    pixel_results=[]
    try:
        for step in policy['checkpoints']:
            for sid in policy['dev']:
                stem=f'step{step:04d}_{sid}'
                receipt=run('evaluate_streaming.py',common+['--checkpoint',str(train/f'adapter_step{step:04d}.pt'),
                    '--checkpoint-sha',checkpoint_hashes[step],'--sample-id',sid],out/(stem+'_latent'))
                pix=run('evaluate_pixels.py',pixel_common+['--prediction',receipt['path'],'--prediction-sha',receipt['sha256']],out/(stem+'_pixels'))
                pixel_results.append(pix)
        for sid in policy['dev']:
            stem='dense_'+sid
            receipt=run('evaluate_dense_reference.py',common+['--sample-id',sid],out/(stem+'_latent'))
            pix=run('evaluate_pixels.py',pixel_common+['--prediction',receipt['path'],'--prediction-sha',receipt['sha256']],out/(stem+'_pixels'))
            pixel_results.append(pix)
        records=[json.loads(Path(d['path']).read_text()) for d in pixel_results]
        aggregates={}
        for label in [f'strict_streaming_step{s}' for s in policy['checkpoints']]+['released_dense_refiner']:
            rs=[r for r in records if r['arm']==label];assert len(rs)==3 and set(r['sample_id'] for r in rs)==set(policy['dev'])
            mean={k:statistics.mean(r['nativeHR_metrics']['regions']['full226'][k] for r in rs) for k in ('pixel_l1','lpips','psnr_db','delta_l1')}
            base={k:statistics.mean(r['no_refiner_nativeHR_metrics']['regions']['full226'][k] for r in rs) for k in ('pixel_l1','lpips')}
            ratios={k:mean[k]/base[k] for k in base}
            aggregates[label]={'equal_clip':mean,'ratios_vs_no_refiner':ratios,
                               'metric_gate_passed':ratios['pixel_l1']<=1.02 and ratios['lpips']<=1.02}
        candidates=[s for s in policy['checkpoints'] if aggregates[f'strict_streaming_step{s}']['metric_gate_passed']]
        write('result.json',{'status':'all_full226_metrics_complete_visual_and_decoder_acceptance_pending',
              'aggregates':aggregates,'metric_passing_steps':candidates,'accepted_refiner':None,
              'selection':'defer until real continuous visual review of all metric-passing candidates; then frozen lowest-LPIPS/earliest-tie rule',
              'pixel_results':pixel_results,'strict_decoder_validated':False,'visual_quality_passed':False,'test_unlocked':False,
              'optimizer_updates_this_evaluation':0,'elapsed_seconds':time.time()-start})
    except BaseException as e:
        write('failure.json',{'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'completed':completed});raise

if __name__=='__main__':main()
