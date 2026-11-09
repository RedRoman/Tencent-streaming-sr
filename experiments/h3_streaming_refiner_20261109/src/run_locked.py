"""One owned child; three flocks held until reaped. Refuses reuse of an attempt."""
import argparse,fcntl,json,os,re,signal,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--device',type=int,required=True);p.add_argument('--attempt',required=True);p.add_argument('--lock-path',action='append',required=True);p.add_argument('command',nargs=argparse.REMAINDER);args=p.parse_args()
assert 0<=args.device<=7 and args.command and len(set(args.lock_path))==3
out=Path(args.attempt);out.mkdir(parents=True,exist_ok=False)
def write(name,data):
    tmp=out/(name+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(out/name)
locks=[];child=None
try:
    for path in args.lock_path:
        f=open(path,'a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True).stdout
    (out/'npu_smi_admission.txt').write_text(smi)
    assert f'No running processes found in NPU {args.device}' in smi,'Card is busy; no process will be started'
    assert not re.search(r'^\|\s*'+str(args.device)+r'\s+\d+\s*\|\s*\d+\s*\|',smi,re.M)
    env=dict(os.environ,ASCEND_VISIBLE_DEVICES=str(args.device),ASCEND_RT_VISIBLE_DEVICES=str(args.device),CUDA_VISIBLE_DEVICES='',TORCH_DEVICE_BACKEND_AUTOLOAD='0')
    record={'parent_pid':os.getpid(),'physical_npu':args.device,'all_three_locks_held':True,'started_unix':time.time(),'command':args.command,'status':'starting'}
    write('launch.json',record)
    child=subprocess.Popen(args.command,env=env,start_new_session=True)
    def relay(sig,frame):
        if child.poll() is None:os.killpg(child.pid,sig)
    signal.signal(signal.SIGTERM,relay);signal.signal(signal.SIGINT,relay)
    record.update(child_pid=child.pid,child_pgid=child.pid,status='running');write('launch.json',record)
    code=child.wait();record.update(status='complete' if code==0 else 'failed',exit_code=code,ended_unix=time.time(),child_reaped=True);write('launch.json',record)
    sys.exit(code)
except BaseException as e:
    if not isinstance(e,SystemExit):write('launcher_failure.json',{'type':type(e).__name__,'error':str(e),'time':time.time()})
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM);child.wait()
    raise
finally:
    for f in locks:f.close()
