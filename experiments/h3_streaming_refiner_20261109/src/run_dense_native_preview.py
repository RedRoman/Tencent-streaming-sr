"""One original dense reference + pixels under a single external lock lifetime."""
import argparse,hashlib,json,subprocess,sys,time,traceback
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    for n in ('stage','root','prior-stage','data-manifest','assets-manifest','assets-manifest-sha','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();stage=Path(a.stage);pin=json.loads((stage/'config/evaluation_workers_v4.json').read_text())
    assert all(sha(stage/n)==v for n,v in pin['new_sources'].items())
    assert sha(stage/'config/evaluation_v1.json')==pin['frozen_protocol_sha256']
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.time()
    def run(name,args,folder):
        with (out/(folder+'.log')).open('x') as log:
            subprocess.run([sys.executable,str(stage/'src'/name),*args,'--output',str(out/folder)],stdout=log,stderr=subprocess.STDOUT,check=True)
        return {'path':str(out/folder/'result.json'),'sha256':sha(out/folder/'result.json')}
    try:
        latent=run('evaluate_dense_reference.py',['--root',a.root,'--data-manifest',a.data_manifest,
            '--data-manifest-sha',pin['dev_manifest_sha256'],'--protocol',str(stage/'config/evaluation_v1.json'),
            '--sample-id','WingIt_dev_3586_3811','--assets-manifest',a.assets_manifest,'--assets-manifest-sha',a.assets_manifest_sha],'latent')
        pixels=run('evaluate_pixels.py',['--prior-stage',a.prior_stage,'--prior-plan-sha',pin['prior_decoder_plan_sha256'],
            '--prediction',latent['path'],'--prediction-sha',latent['sha256'],'--data-manifest',a.data_manifest,
            '--data-manifest-sha',pin['dev_manifest_sha256']],'pixels')
        (out/'result.json').write_text(json.dumps({'status':'full226_dense_reference_complete_visual_review_pending',
            'latent':latent,'pixels':pixels,'source_manifest_sha256':sha(stage/'config/evaluation_workers_v4.json'),
            'optimizer_updates':0,'seconds':time.time()-started,'strict_decoder_validated':False},indent=2)+'\n')
    except BaseException as e:
        (out/'failure.json').write_text(json.dumps({'error':str(e),'traceback':traceback.format_exc()},indent=2)+'\n');raise

if __name__=='__main__':main()
