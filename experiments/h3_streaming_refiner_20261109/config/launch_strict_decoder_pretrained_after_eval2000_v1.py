import os,json,hashlib,subprocess
from pathlib import Path
r=Path('<AUTHORIZED_CACHE>/h3-streaming-refiner-20261109')
p=r/'config/strict_decoder_pretrained_launch_after_eval2000_v1.json'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='11e0797fac37e2118dc3a663203cf83a69505b1130174108401bf095a9d5b9ea'
t=json.loads(p.read_text())
manifest_path=Path(t['validation_stage'])/'manifest.json'
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==t['validation_stage_manifest_sha256']
for rel,rec in json.loads(manifest_path.read_text())['files'].items():
    f=Path(t['validation_stage'])/rel
    assert f.stat().st_size==rec['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==rec['sha256']
original=json.loads((r/'launches/evaluation_continuation_v2/launch.json').read_text())
assert original['status'] in ['complete','failed'] and original['child_reaped']
assert not any(Path('/proc',str(original[k])).exists() for k in ['parent_pid','child_pid'])

assert hashlib.sha256(Path(t['launcher']).read_bytes()).hexdigest()=='0dd684ffaf2b172426566dcfdee74b204052c6ad3d53622b9dc6fb7aea7bb7c6'
old=json.loads((r/'launches/anchored_step2000_ElephantsDream_dev_13070_13295_v1/launch.json').read_text())
assert old['status']=='complete' and old['exit_code']==0 and old['child_reaped']
assert not any(Path('/proc',str(old[k])).exists() for k in ['parent_pid','child_pid'])
prior=json.loads((r/'evaluations/anchored_step2000_v1/ElephantsDream_dev_13070_13295/result.json').read_text())
assert prior['status']=='anchored_stage2000_full226_video_ready_visual_review_pending'
assert not Path(t['attempt']).exists() and not Path(t['output']).exists()
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
    try:cmd=(proc/'cmdline').read_bytes()
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    assert not (b'validate_strict_decoder.py' in cmd and b'--admission-only' not in cmd), 'Another exact clip dispatcher already exists'
mem={x.split(':')[0]:int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')};assert mem['MemAvailable']>200*1024*1024
smi=subprocess.check_output(['npu-smi','info'],text=True);assert 'No running processes found in NPU 7' in smi
argv=['python',t['launcher'],'--device','7','--attempt',t['attempt'],'--lock-path','<AUTHORIZED_CACHE>/device_locks/npu7.lock','--lock-path','/tmp/yunfeng_npu_7.lock','--lock-path','/tmp/codex-h3-vae-npu-7.lock',*t['worker_arguments']]
log=open(r/'logs/strict_decoder_pretrained_v1.log','x');os.dup2(log.fileno(),1);os.dup2(log.fileno(),2);os.execvpe(argv[0],argv,os.environ)
