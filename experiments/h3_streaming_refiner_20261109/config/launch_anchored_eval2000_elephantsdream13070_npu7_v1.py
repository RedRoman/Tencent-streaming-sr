import os,json,hashlib,subprocess
from pathlib import Path
r=Path('<AUTHORIZED_CACHE>/h3-streaming-refiner-20261109')
p=r/'config/anchored_step2000_launch_template_elephantsdream13070_npu7_v1.json'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='162eb8a9f76cc397de784792c16ddf430ead9e136c0996ebc0a663cb9f9ae5ac'
t=json.loads(p.read_text())
assert hashlib.sha256(Path(t['launcher']).read_bytes()).hexdigest()=='0dd684ffaf2b172426566dcfdee74b204052c6ad3d53622b9dc6fb7aea7bb7c6'
old=json.loads((r/'launches/anchored_step1000_WingIt_dev_300_525_v1/launch.json').read_text())
assert old['status']=='complete' and old['exit_code']==0 and old['child_reaped']
assert not any(Path('/proc',str(old[k])).exists() for k in ['parent_pid','child_pid'])
prior=json.loads((r/'evaluations/anchored_step1000_v1/WingIt_dev_300_525/result.json').read_text())
assert prior['status']=='anchored_stage1000_full226_video_ready_visual_review_pending'
assert not Path(t['attempt']).exists() and not Path(t['output']).exists()
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
    try:cmd=(proc/'cmdline').read_bytes()
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    assert not (b'run_anchored_step2000_evaluation_v1.py' in cmd and b'ElephantsDream_dev_13070_13295' in cmd), 'Another exact clip dispatcher already exists'
mem={x.split(':')[0]:int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')};assert mem['MemAvailable']>200*1024*1024
smi=subprocess.check_output(['npu-smi','info'],text=True);assert 'No running processes found in NPU 7' in smi
argv=['python',t['launcher'],'--device','7','--attempt',t['attempt'],'--lock-path','<AUTHORIZED_CACHE>/device_locks/npu7.lock','--lock-path','/tmp/yunfeng_npu_7.lock','--lock-path','/tmp/codex-h3-vae-npu-7.lock',*t['worker_arguments']]
log=open(r/'logs/anchored_eval2000_elephantsdream13070_v1.log','x');os.dup2(log.fileno(),1);os.dup2(log.fileno(),2);os.execvpe(argv[0],argv,os.environ)
