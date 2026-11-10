"""Standard-library admission for the exact new-stage1000 video evaluation."""
import hashlib
import json
import os
from pathlib import Path

TRAINING_SHA = '1ac8ae9c82a052386cfff984fa1adbad07c4740383770e7ec31afd2144c08a0a'
DEV_SHA = '9ce8eebb66ad62e243dfa0eafef2961bdf72539d44ed514f03b8dc99d4f31f23'
UPSAMPLER_SHA = '7b9f43c17462fb2d0554645bc7889dc7abdebc052ee9f18f6780624de912e0b5'

def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def pinned(path, digest):
    assert sha(path) == digest, 'Pinned file changed: '+Path(path).name
    return read(path)

def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)

def checkpoint_receipt(training):
    """Never scan for latest or substitute an earlier checkpoint."""
    training = Path(training)
    path = training/'checkpoint_stage1000.json'
    if not path.exists():
        return None
    row = read(path)
    assert row['stage_step'] == 1000 and row['total_optimizer_updates'] == 3500
    paths = [training/'adapter_stage1000.pt', training/'private_resume_stage1000.pt']
    for p, hk, bk in zip(paths, ['adapter_sha256','private_optimizer_sha256'],
                         ['adapter_bytes','private_state_bytes']):
        assert p.stat().st_size == row[bk] and sha(p) == row[hk]
    return {'receipt_path':str(path),'receipt_sha256':sha(path),'record':row,
            'adapter_path':str(paths[0]),'private_resume_path':str(paths[1])}

def guard_training_device(launch, device):
    assert 0 <= device <= 7
    live = any(Path('/proc',str(launch[k])).exists() for k in ('parent_pid','child_pid'))
    if device == launch['physical_npu']:
        assert not live and launch.get('child_reaped') is True
    return {'training_process_present':live,'same_device':device == launch['physical_npu']}

def admit(a):
    stage = Path(a.stage)
    manifest = pinned(stage/'manifest.json',a.manifest_sha)
    for name,digest in manifest['files'].items():
        assert sha(stage/name) == digest, name
    control = read(stage/'evidence/anchored_eval_CPU_control_v1.json')
    assert control['status']=='passed' and not control['torch_imported'] and not control['accelerator_imported']
    for name,digest in control['tested_sources'].items():
        assert sha(stage/name)==digest
    plan = read(stage/'config/anchored_step1000_evaluation_v1.json')
    assert plan['stage_step'] == 1000 and plan['total_step'] == 3500
    original = pinned(stage/'config/evaluation_parent_v1.json',plan['parent_protocol_sha256'])
    trainstage = Path(a.training_stage)
    frozen = pinned(trainstage/'manifest.json',TRAINING_SHA)
    for name,digest in frozen['files'].items():
        assert sha(trainstage/name) == digest
    for name in ('anchored_refiner_v1.py','paged_runtime.py','causal_attention.py'):
        assert sha(stage/'src'/name) == frozen['files']['src/'+name]
    data = pinned(a.data_manifest,DEV_SHA)
    assert data['protocol_sha256'] == plan['parent_protocol_sha256']
    assert data['accepted_upsampler_sha256'] == original['shared_input']['upsampler_sha256'] == UPSAMPLER_SHA
    assert not data['test_unlocked'] and a.sample_id in original['dev'] == plan['sample_order']
    rec = next(v for v in data['samples'] if v['sample_id'] == a.sample_id)
    assert sha(rec['path']) == rec['sha256']
    assert sha(data['condition']['path']) == data['condition']['sha256'] == plan['condition_sha256']
    assets = pinned(a.assets_manifest,plan['assets_manifest_sha256'])
    assert assets['status'] == 'complete'
    for name,v in assets['files'].items():
        s = (Path(a.root)/name).stat()
        assert (s.st_size,s.st_mtime_ns,s.st_ino) == (v['bytes'],v['mtime_ns'],v['inode'])
    pp = Path(a.prior_stage)/'h3_production67_limiteddev_full226_plan_v1.json'
    decoder = pinned(pp,plan['prior_decoder_plan_sha256'])
    for name,v in decoder['files'].items():
        assert sha(Path(a.prior_stage)/name) == v['sha256']
    launch = read(a.training_launch)
    command = launch['command']
    assert sha(command[1]) == frozen['files']['src/train_anchored_refiner_v1.py']
    assert command[command.index('--manifest-sha')+1] == TRAINING_SHA
    assert Path(command[command.index('--output')+1]).resolve() == Path(a.training).resolve()
    device = guard_training_device(launch,a.physical_device)
    cp = checkpoint_receipt(a.training)
    return {'status':'stdlib_admission','ready':cp is not None,'checkpoint':cp,
            'stage_manifest_sha256':a.manifest_sha,'training_stage_manifest_sha256':TRAINING_SHA,
            'sample_id':a.sample_id,'device_check':device,'optimizer_updates':0,
            'training_mutated':False}, plan, data

def verify_worker_lock_ancestor(device):
    """Workers run under a dispatcher whose parent holds the three locks."""
    dispatcher = os.getppid()
    owner = int(Path('/proc',str(dispatcher),'stat').read_text().rsplit(')',1)[1].split()[1])
    cmd = Path('/proc',str(owner),'cmdline').read_bytes().split(b'\0')
    assert Path(cmd[1].decode()).name == 'run_locked.py'
    assert sha(cmd[1].decode()) == '0dd684ffaf2b172426566dcfdee74b204052c6ad3d53622b9dc6fb7aea7bb7c6'
    identities=[]
    for line in Path('/proc/locks').read_text().splitlines():
        p=line.split()
        if len(p)>5 and p[1]=='FLOCK' and p[3]=='WRITE' and p[4]==str(owner):
            major,minor,inode=p[5].split(':');identities.append((int(major,16),int(minor,16),int(inode)))
    for path in [f'<AUTHORIZED_CACHE>/device_locks/npu{device}.lock',f'/tmp/yunfeng_npu_{device}.lock',f'/tmp/codex-h3-vae-npu-{device}.lock']:
        s=Path(path).stat();assert (os.major(s.st_dev),os.minor(s.st_dev),s.st_ino) in identities
    assert all(os.environ.get(k)==str(device) for k in ('ASCEND_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES'))
    return {'dispatcher_pid':dispatcher,'lock_owner_pid':owner,'three_locks_verified':True,'physical_device':device}
