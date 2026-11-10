"""Standard-library admission for the exact new-stage2000 video evaluation."""
import hashlib
import json
import os
from pathlib import Path

ENDPOINT_ADAPTER_SHA = '5e95b0826db51003968597b8ffbc6159f42d0c784ad8c0a10843734fae3dc485'
ENDPOINT_PRIVATE_SHA = '5541609e21faa4ad4cd97b09a1455fd9a81ba08236465b0359f8cc8077b93549'
ENDPOINT_RECEIPT_SHA = 'b6f398f549de1a4dbb0c606dccf9236a41441d4ad37cc7aa873309ca6e7c042e'
REFERENCE_STAGE_SHA = '1a75fb5347bffca8ec269e207494b9c00b78939d122347c23d1db03b49bb987b'
REFERENCE_ADAPTER_SHA = 'f16119620e6bcbce0c6da7ce00560dc1733e9d0b3c0d6d4107c38974d43c50a6'
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
    path = training/'checkpoint_stage2000.json'
    if not path.exists():
        return None
    row = pinned(path,ENDPOINT_RECEIPT_SHA)
    assert row['adapter_sha256']==ENDPOINT_ADAPTER_SHA and row['private_optimizer_sha256']==ENDPOINT_PRIVATE_SHA
    assert row['stage_step'] == 2000 and row['total_optimizer_updates'] == 4500
    paths = [training/'adapter_stage2000.pt', training/'private_resume_stage2000.pt']
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
    control = read(stage/'evidence/anchored_eval2000_CPU_control_v1.json')
    assert control['status']=='passed' and not control['torch_imported'] and not control['accelerator_imported']
    for name,digest in control['tested_sources'].items():
        assert sha(stage/name)==digest
    plan = read(stage/'config/anchored_step2000_evaluation_v1.json')
    assert plan['stage_step'] == 2000 and plan['total_step'] == 4500
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
    assert launch.get('status')=='complete' and launch.get('exit_code')==0 and launch.get('child_reaped')
    assert all(not Path('/proc',str(launch[k])).exists() for k in ('parent_pid','child_pid'))
    terminal=read(Path(a.training)/'complete.json')
    assert terminal['stage_optimizer_updates']==2000 and terminal['total_optimizer_updates']==4500
    device = guard_training_device(launch,a.physical_device)
    cp = checkpoint_receipt(a.training)
    reference = reference_admission(a,plan['parent_protocol_sha256'])
    return {'status':'stdlib_admission','ready':cp is not None,'checkpoint':cp,
            'stage_manifest_sha256':a.manifest_sha,'training_stage_manifest_sha256':TRAINING_SHA,
            'sample_id':a.sample_id,'device_check':device,'optimizer_updates':0,
            'training_mutated':False,'reference':reference}, plan, data

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


def validate_reference_metadata(result,sample_id):
    assert result['status']=='anchored_stage1000_full226_video_ready_visual_review_pending'
    assert result['sample_id']==sample_id and result['stage_step']==1000 and result['total_training_step']==3500
    assert result['checkpoint']['record']['adapter_sha256']==REFERENCE_ADAPTER_SHA
    assert result['optimizer_updates']==0 and not result['training_mutated']

def reference_admission(a,protocol_sha):
    from run_evaluation_continuation_v2 import validate_dense_latent, validate_dense_pixels
    folder=Path(a.reference_evaluation)
    result=pinned(folder/'result.json',a.reference_result_sha)
    validate_reference_metadata(result,a.sample_id)
    launch=read(a.reference_launch)
    assert launch['status']=='complete' and launch['exit_code']==0 and launch['child_reaped']
    assert all(not Path('/proc',str(launch[k])).exists() for k in ('parent_pid','child_pid'))
    command=launch['command']
    assert Path(command[command.index('--output')+1]).resolve()==folder.resolve()
    assert command[command.index('--sample-id')+1]==a.sample_id
    assert command[command.index('--manifest-sha')+1]==REFERENCE_STAGE_SHA
    oldstage=Path(command[command.index('--stage')+1])
    original=pinned(oldstage/'manifest.json',REFERENCE_STAGE_SHA)
    for name,digest in original['files'].items(): assert sha(oldstage/name)==digest
    old_admission=read(folder/'admission.json')
    assert old_admission['stage_manifest_sha256']==REFERENCE_STAGE_SHA
    lr=pinned(result['dense_latent']['path'],result['dense_latent']['sha256'])
    pr=pinned(result['dense_pixels']['path'],result['dense_pixels']['sha256'])
    gate_path=Path(result['dense_latent']['path']).parent/'upstream_gate.json'
    gate=pinned(gate_path,a.reference_gate_sha)
    validate_dense_latent(lr,a.sample_id,DEV_SHA,protocol_sha,gate)
    validate_dense_pixels(pr,a.sample_id,DEV_SHA,result['dense_latent']['sha256'])
    video=pinned(result['video']['path'],result['video']['sha256'])
    assert video['status']=='full226_three_videos_technical_audit_passed' and video['stage_step']==1000
    assert video['sample_id']==a.sample_id and video['frames']==226 and video['fps']==24
    return {'source_result_path':str(folder/'result.json'),'source_result_sha256':a.reference_result_sha,
            'source_launch_sha256':sha(a.reference_launch),'original_source_manifest_sha256':REFERENCE_STAGE_SHA,
            'dense_latent':result['dense_latent'],'dense_pixels':result['dense_pixels'],
            'dense_gate_sha256':a.reference_gate_sha,'same_upsampler_noise_condition_geometry_verified':True,
            'original_receipts_unchanged':True,'reference_optimizer_updates':0}
