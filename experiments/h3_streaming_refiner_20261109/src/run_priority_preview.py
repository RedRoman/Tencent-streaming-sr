"""One full step200 clip for a requested engineering preview; no selection.

Reuses the immutable evaluation workers and frozen metric protocol verbatim.
Invoke inside the existing three-lock launcher. Never waits or retries.
"""
import argparse, hashlib, json, subprocess, sys, time, traceback
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    for n in ('stage','root','prior-stage','training','training-launch','data-manifest','assets-manifest','assets-manifest-sha','output'):
        p.add_argument('--'+n,required=True)
    a=p.parse_args();stage=Path(a.stage);train=Path(a.training)
    source=json.loads((stage/'config/evaluation_workers_v4.json').read_text())
    assert all(sha(stage/n)==v for n,v in source['new_sources'].items())
    protocol=stage/'config/evaluation_v1.json';assert sha(protocol)==source['frozen_protocol_sha256']
    status=json.loads((train/'status.json').read_text());launch=json.loads(Path(a.training_launch).read_text())
    assert status['optimizer_updates']==200 and status['status']=='pilot_training_complete_evaluation_pending'
    assert launch['exit_code']==0 and launch['child_reaped']
    checkpoint=train/'adapter_step0200.pt'
    checkpoint_sha='b1bd7d862dc98e12bebdd6ea58257fbfd069d8d5173f39bebef7290c2a382ab1'
    assert sha(checkpoint)==checkpoint_sha
    assert sha(a.data_manifest)==source['dev_manifest_sha256']
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.time()
    def write(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
    write('inputs.json',{'sample_id':'WingIt_dev_3586_3811','training_step':200,'checkpoint_sha256':checkpoint_sha,
          'protocol_sha256':sha(protocol),'source_manifest_sha256':sha(stage/'config/evaluation_workers_v4.json'),
          'reason':'user requested visualization video; evaluation order only changed; no selection or gate change',
          'accepted_refiner':None,'optimizer_updates':0})
    def run(worker,args,folder):
        print(json.dumps({'event':'start_worker','worker':worker,'output':str(folder)}),flush=True)
        with (out/(folder.name+'.log')).open('x') as log:
            subprocess.run([sys.executable,str(stage/'src'/worker),*args,'--output',str(folder)],stdout=log,stderr=subprocess.STDOUT,check=True)
        receipt={'path':str(folder/'result.json'),'sha256':sha(folder/'result.json')}
        write('progress.json',{'last_completed':receipt,'elapsed_seconds':time.time()-started})
        return receipt
    try:
        latent=run('evaluate_streaming.py',['--root',a.root,'--data-manifest',a.data_manifest,
            '--data-manifest-sha',source['dev_manifest_sha256'],'--checkpoint',str(checkpoint),'--checkpoint-sha',checkpoint_sha,
            '--protocol',str(protocol),'--sample-id','WingIt_dev_3586_3811','--assets-manifest',a.assets_manifest,
            '--assets-manifest-sha',a.assets_manifest_sha],out/'latent')
        pixels=run('evaluate_pixels.py',['--prior-stage',a.prior_stage,'--prior-plan-sha',source['prior_decoder_plan_sha256'],
            '--prediction',latent['path'],'--prediction-sha',latent['sha256'],'--data-manifest',a.data_manifest,
            '--data-manifest-sha',source['dev_manifest_sha256']],out/'pixels')
        write('result.json',{'status':'full226_engineering_preview_pixels_complete','latent':latent,'pixels':pixels,
              'seconds':time.time()-started,'accepted_refiner':None,'continuous_visual_reviewed':False,
              'strict_decoder_validated':False,'remaining_frozen_evaluation_required':True})
    except BaseException as e:
        write('failure.json',{'error':str(e),'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
