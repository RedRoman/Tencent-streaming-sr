"""One-shot local conversion with actual parent wait/reap evidence."""
import argparse,datetime,hashlib,json,os,subprocess
from pathlib import Path

D=Path('/path/to/h3-project/upscaler_v5_motion_20261106/analysis/production67_limiteddev_training_20261107')
STAGE=Path('/path/to/staging/h3_production67_limiteddev_media_20261107_v3')

def record(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}

def write(p,d):
    with p.open('x') as f:
        json.dump(d,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-root',required=True);ap.add_argument('--subset',choices=['priority','native'],required=True);ap.add_argument('--attempt',type=int,required=True);a=ap.parse_args()
    assert record(Path(a.source_root)/'render_receipt.json')['sha256']=='0c871145247fd9b142ce1b3a7d24b09dae28a3bdf3eb478e8b67cceefb3dcead'
    outer=D/('media_execution_20261108_v3/local_H264_v4_outer_attempt'+str(a.attempt));outer.mkdir(exist_ok=False)
    env=dict(os.environ);override={'ASCEND_VISIBLE_DEVICES':'','ASCEND_RT_VISIBLE_DEVICES':'','CUDA_VISIBLE_DEVICES':'','TORCH_DEVICE_BACKEND_AUTOLOAD':'0','PYTHONDONTWRITEBYTECODE':'1'};env.update(override)
    cmd=['/path/to/local-toolchain/opt/python@3.14/bin/python3.14','/path/to/staging/convert_h3_limiteddev_retained_media_v4.py','--adapter-plan','/path/to/staging/h3_limiteddev_H264_subset_plan_v4.json','--adapter-plan-sha256','0893fa1205797235d7c97ab8297e03d1f4f25781eeb8e9756881d7c3b0200ca8','--source-root',a.source_root,'--subset',a.subset,'--attempt',str(a.attempt)]
    start={'schema':'h3-media-v4-local-parent-start-v1','utc':now(),'parent_pid':os.getpid(),'command_argv':cmd,'environment_override':override,'driver':record(__file__),'retry':False}
    with (outer/'convert.log').open('x') as log:
        child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);start.update(child_pid=child.pid,child_pgid=child.pid);write(outer/'started.json',start);rc=child.wait();log.flush();os.fsync(log.fileno())
    ps=subprocess.run(['ps','-axo','pid=,pgid=,stat=,command='],stdout=subprocess.PIPE,text=True,check=True).stdout
    group=[line.strip() for line in ps.splitlines() if len(line.split(None,3))>=3 and line.split(None,3)[1]==str(child.pid)]
    result={**start,'schema':'h3-media-v4-local-parent-exit-v1','finished_utc':now(),'exit_code':rc,'child_reaped':child.returncode is not None,'live_group_members_at_end':group,'log':record(outer/'convert.log'),'status':'completed' if rc==0 and not group else 'failed_no_auto_retry'}
    for n in ['batch_receipt.json','failure.json']:
        p=D/'media_h264_v4'/('batch_attempt_'+str(a.attempt))/n
        if p.exists():result[n]=record(p)
    write(outer/'exit_receipt.json',result);print(json.dumps(result));raise SystemExit(rc if rc else (1 if group else 0))

if __name__=='__main__':main()
