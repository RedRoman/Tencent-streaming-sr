"""Single bounded CPU postprocessor for the already running preview.

Does not start or wait for an accelerator/device lock. A per-output flock
prevents duplicate encoders; only this task's result files are observed.
"""
import argparse, fcntl, hashlib, json, os, subprocess, sys, time
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    for n in ('preview','encoder','encoder-sha','data-manifest','data-manifest-sha','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();preview=Path(a.preview);out=Path(a.output);started=time.time()
    lock=out.with_suffix('.encoder.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not out.exists() and sha(a.encoder)==a.encoder_sha
        status=out.with_suffix('.postprocessor.json')
        def write(state,**v):
            tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps({'status':state,'pid':os.getpid(),
                 'elapsed_seconds':time.time()-started,'cpu_only':True,**v},indent=2)+'\n');tmp.replace(status)
        write('waiting_for_owned_preview_pixels')
        while not (preview/'result.json').exists():
            if (preview/'failure.json').exists():
                write('stopped_preview_failed');return
            if time.time()-started>7200:
                write('timed_out_without_touching_inference');return
            time.sleep(30)
        result=json.loads((preview/'result.json').read_text());pixels=result['pixels']
        assert result['status']=='full226_engineering_preview_pixels_complete' and sha(pixels['path'])==pixels['sha256']
        write('encoding')
        subprocess.run([sys.executable,a.encoder,'--data-manifest',a.data_manifest,'--data-manifest-sha',a.data_manifest_sha,
            '--sample-id','WingIt_dev_3586_3811','--prediction',pixels['path'],'--prediction-sha',pixels['sha256'],
            '--output',a.output],check=True)
        write('complete',video_audit_sha256=sha(out/'video_audit.json'))

if __name__=='__main__':main()
