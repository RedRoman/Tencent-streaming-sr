"""CPU-only integrity and predecessor checks for the independent decoder validator."""
import hashlib
import json
import os
from pathlib import Path

DEV_SHA = '9ce8eebb66ad62e243dfa0eafef2961bdf72539d44ed514f03b8dc99d4f31f23'
PRIOR_SHA = 'd031e4383c689bc03615a954bdb9b37fe7970b8425b24775a93b7f7262edb737'
CANDIDATE_SHA = '7e373940d85620cb95dbd0be3fdedd11514cbecfec3809e906e4a1e6573553ec'
PLAN_SHA = 'd96555e80fac797782e60feafbdb7e6894f1614e9b5f3e9a18b0418bdd63636e'
CPU_SHA = 'e042b0fb54538aa0329a9583f910ebac0c52b70a2ff629fd3ed5e6aaf4f65504'
UPSAMPLER_SHA = '7b9f43c17462fb2d0554645bc7889dc7abdebc052ee9f18f6780624de912e0b5'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def read_pinned(path,digest):
    if sha(path)!=digest:raise ValueError('Pinned file SHA mismatch: '+Path(path).name)
    return json.loads(Path(path).read_text())


def checked(desc):
    path=Path(desc['path'])
    if 'bytes' in desc and path.stat().st_size!=desc['bytes']:raise ValueError('Payload size mismatch')
    if sha(path)!=desc['sha256']:raise ValueError('Payload SHA mismatch: '+path.name)
    return path


def predecessor_state(path):
    row=json.loads(Path(path).read_text())
    if any(not isinstance(row.get(k),int) or row[k]<=1 for k in ('parent_pid','child_pid')):
        raise ValueError('Predecessor process identities missing')
    terminal=row.get('status') in ('complete','failed') and row.get('child_reaped') is True
    # A still-live PID is conservatively treated as occupied, even if it was reused.
    absent=all(not Path('/proc',str(row.get(k,-1))).exists() for k in ('parent_pid','child_pid'))
    return {'terminal':terminal,'processes_absent':absent,'ready':terminal and absent}


def verify_parent_locks(physical_device):
    """Fail before importing accelerator libraries unless the parent owns all three flocks."""
    if not 0<=physical_device<=7:raise ValueError('Invalid physical device')
    parent=os.getppid()
    paths=[Path('/cache/PROJECT/device_locks')/f'npu{physical_device}.lock',
           Path(f'/tmp/PROJECT_npu_{physical_device}.lock'),
           Path(f'/tmp/codex-h3-vae-npu-{physical_device}.lock')]
    owner=[]
    for line in Path('/proc/locks').read_text().splitlines():
        parts=line.split()
        if len(parts)>=6 and parts[1]=='FLOCK' and parts[3]=='WRITE' and parts[4]==str(parent):
            major,minor,inode=parts[5].split(':')
            owner.append((int(major,16),int(minor,16),int(inode)))
    for path in paths:
        stat=path.stat();identity=(os.major(stat.st_dev),os.minor(stat.st_dev),stat.st_ino)
        if identity not in owner:raise RuntimeError('Parent does not hold all required device locks')
    if any(os.environ.get(k)!=str(physical_device) for k in ('ASCEND_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES')):
        raise RuntimeError('Physical device visibility does not match the lock owner')
    return {'parent_pid':parent,'physical_device':physical_device,'three_FLOCK_inodes_verified':True}


def admit(args):
    stage=Path(args.candidate_stage)
    manifest=read_pinned(stage/'manifest.json',CANDIDATE_SHA)
    for name,d in manifest['files'].items():checked({'path':str(stage/name),**d})
    if args.validation_plan_sha!=PLAN_SHA:raise ValueError('Predeclared validation plan changed')
    plan=read_pinned(args.validation_plan,PLAN_SHA)
    cpu=read_pinned(args.cpu_report,CPU_SHA)
    if cpu['status']!='passed' or plan['cpu_evidence_sha256']!=CPU_SHA:raise ValueError('CPU prerequisite failed')
    if plan['compensation'] or plan['compile'] or plan['precision']!='FP32':raise ValueError('Decoder convention changed')
    for name,digest in plan['runtime_source_sha256'].items():
        if sha(stage/'src'/name)!=digest or cpu['implementation_sha256'][name]!=digest:
            raise ValueError('CPU-tested source changed')
    prior_path=Path(args.prior_stage)/'h3_production67_limiteddev_full226_plan_v1.json'
    prior=read_pinned(prior_path,PRIOR_SHA)
    for name,d in prior['files'].items():checked({'path':str(Path(args.prior_stage)/name),**d})
    for d in prior['runtime_assets'].values():checked(d)
    data=read_pinned(args.data_manifest,DEV_SHA)
    if data['accepted_upsampler_sha256']!=UPSAMPLER_SHA:raise ValueError('Frozen upsampler changed')
    rec=next(r for r in data['samples'] if r['sample_id']==plan['sample_id'])
    checked(rec)
    for name in ('native_hr','decoded_hr','teacher_student'):checked(rec['rgb'][name])
    predecessor=predecessor_state(args.predecessor_launch)
    evidence={'status':'CPU_integrity_admission_complete','candidate_manifest_sha256':CANDIDATE_SHA,
        'CPU_report_sha256':CPU_SHA,'validation_plan_sha256':args.validation_plan_sha,
        'prior_plan_sha256':PRIOR_SHA,'data_manifest_sha256':DEV_SHA,
        'sample_id':rec['sample_id'],'predecessor':predecessor,'runtime_started':False,
        'optimizer_updates':0,'accelerator_forwards':0}
    return evidence,plan,prior,rec
