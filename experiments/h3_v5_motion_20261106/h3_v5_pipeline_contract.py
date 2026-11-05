"""Immutable launch contract and equal-budget verification for v5 continuation."""
import hashlib
import json
import os
import sys
from pathlib import Path
from evaluate_h3_cmn import module_source
from datetime import datetime, timezone
ROOT=Path(os.environ['EXP_ROOT'])
OUT=ROOT/'upscaler_v5_motion_20261106'
SOURCE=Path(os.environ['EXP_CODE_DIR'])
def source_path(name):
    return module_source(name) if name.endswith('.py') else SOURCE/name
NAMES=('train_h3_v5_motion.py','h3_v5_motion.py','test_h3_v5_motion.py',
       'evaluate_h3_v5_motion.py','test_h3_v5_eval.py','run_h3_v5_motion_pipeline.sh',
       'h3_v5_pipeline_contract.py','run_h3_v5_motion_preflight.sh',
       'audit_h3_conv3d_precision.py','audit_h3_conv3d_causality.py','locked_run.py','env.sh')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,data):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(p)
def verify():
    c=read(OUT/'pipeline_contract.json')
    for name,digest in c['sources'].items():
        assert sha(source_path(name))==digest,name
    assert sha(OUT/'preflight/summary.json')==c['preflight_summary_sha256']
    for name,digest in c['inherited_sources'].items():
        assert sha(source_path(name))==digest,name
    return c
mode=sys.argv[1]
if mode=='prepare':
    assert not (OUT/'pipeline_contract.json').exists(),'Pipeline already started; inspect state instead of repeating'
    for arm in ('control','motion'):
        assert not (OUT/f'{arm}_steps2000').exists()
        assert not (OUT/f'{arm}.log').exists()
    pre=read(OUT/'preflight/summary.json')
    assert pre['status']=='preflight_completed' and pre['completed_updates']==8
    assert pre['frozen_modules_unchanged'] and pre['source_contract_unchanged'] and pre['execution_precision_unchanged']
    assert pre['initial_causality']['passed'] and pre['causality']['passed'] and not pre['test_opened']
    assert pre['contract']['v5_dependency_parity']['passed']
    assert all(pre[k]>0 for k in ['pixel_head_grad_l2','motion_head_grad_l2','motion_roi_head_grad_l2','residual_warp_head_grad_l2'])
    for name,digest in pre['contract']['source_sha256'].items():assert sha(source_path(name))==digest,name
    c=dict(status='frozen',created_utc=datetime.now(timezone.utc).isoformat(),
           sources={n:sha(source_path(n)) for n in NAMES},
           inherited_sources=pre['contract']['source_sha256'],
           preflight_summary_sha256=sha(OUT/'preflight/summary.json'),
           initial_sha256=pre['contract']['initial_sha256'],updates_per_arm=2000,
           selection='Same fixed sparse16dev rule including step0; full120dev gate before test',
           test_reuse='Existing reused16test plus3legacy, never blind; no test checkpoint selection',
           model_scope='Only fixed-latent to upscaled-latent causal module; frozen decoder noncausal')
    write(OUT/'pipeline_contract.json',c);verify()
    write(OUT/'pipeline_status.json',dict(status='running',stage='prepared'))
    print('Prepared frozen once-only pipeline contract')
elif mode=='verify':
    verify();print('Pipeline sources and preflight unchanged')
elif mode=='paired':
    c=verify();summaries={a:read(OUT/f'{a}_steps2000/summary.json') for a in ('control','motion')}
    long_audit_records={}
    for a,s in summaries.items():
        assert s['status']=='completed' and s['completed_updates']==2000 and not s['test_opened']
        assert s['frozen_modules_unchanged'] and s['source_contract_unchanged'] and s['execution_precision_unchanged']
        assert s['initial_causality']['passed'] and s['causality']['passed']
        assert s['contract']['initial_sha256']==c['initial_sha256']
        audit=read(OUT/f'causal_{a}_hf32_off.json')
        assert audit['all_passed'] and audit['sources_unchanged'] and audit['execution_precision']['conv_allow_hf32'] is False
        best=OUT/f'{a}_steps2000/best.pt'
        assert audit['checkpoint']['sha256']==s['best_checkpoint_sha256']==sha(best)
        assert audit['mode']=='actual_345m_checkpoint' and audit['status']=='passed'
        assert audit['frames']==163 and audit['check_count']==16 and len(audit['checks'])==16
        assert audit['input_spatial']==[2,2] and audit['architecture']['parameter_count']==345280216
        assert audit['requested_conv_hf32'] is False and audit['execution_precision']['matmul_allow_hf32'] is False
        assert audit['sources']==audit['sources_after']
        for name,check in audit['checks'].items():
            assert check['passed'],(a,name)
            if 'max_abs' in check:assert check['finite'] and check['max_abs']<=check['max_abs_limit'],(a,name)
        for record in [*audit['sources'].values(),audit['precision_driver']]:
            path=Path(record['path'])
            assert path.stat().st_size==record['bytes'] and sha(path)==record['sha256']
        assert audit['released_weights']['sha256']==s['contract']['original_upscaler_fingerprint']['base_weights_sha256']
        long_audit_records[a]=dict(audit_sha256=sha(OUT/f'causal_{a}_hf32_off.json'),
            checkpoint_sha256=audit['checkpoint']['sha256'],frames=163,checks=16,conv_allow_hf32=False)
    configs={a:dict(s['config']) for a,s in summaries.items()}
    for config in configs.values():
        for key in ['arm','output','loss_definition']:config.pop(key)
    assert configs['control']==configs['motion'],'Different non-objective configuration'
    for key in ['decoder_fingerprint','lpips_fingerprint','source_sha256','execution_precision','initial_sha256','teacher_cache_ready_sha256']:
        assert summaries['control']['contract'][key]==summaries['motion']['contract'][key],key
    logs={a:[json.loads(v) for v in (OUT/f'{a}_steps2000/steps.jsonl').read_text().splitlines()] for a in summaries}
    for a,rows in logs.items():assert [r['step'] for r in rows]==list(range(1,2001)),a
    for x,y in zip(logs['control'],logs['motion']):
        for key in ['step','sample_id','lr','spec']:assert x[key]==y[key],(x['step'],key)
    x=summaries['control']['initial_dev'];y=summaries['motion']['initial_dev']
    assert x.keys()==y.keys()
    differences={k:abs(x['mean'][k]-y['mean'][k]) for k in x['mean']}
    assert max(differences.values())<1e-6,differences
    assert x['samples'].keys()==y['samples'].keys()
    for ident,a in x['samples'].items():
        b=y['samples'][ident]
        assert a.keys()==b.keys()
        for key,value in a.items():
            if isinstance(value,(int,float)):assert abs(value-b[key])<1e-6,(ident,key)
            else:assert value==b[key],(ident,key)
    write(OUT/'paired_training_audit.json',dict(passed=True,updates_per_arm=2000,
        same_sample_region_lpips_lr_schedule=True,initial_dev_mean_abs_differences=differences,
        bound_long_causality_audits=long_audit_records,
        scope='Both objectives share model, data, optimizer, schedule and fixed selection; no test accessed'))
    print('Equal-budget training audit passed')
elif mode=='progress':
    write(OUT/'pipeline_status.json',dict(status='running',stage=sys.argv[2],
        updated_utc=datetime.now(timezone.utc).isoformat()))
elif mode=='status':
    stage,code=sys.argv[2],int(sys.argv[3])
    status='completed' if code==0 and stage=='completed' else 'failed'
    evaluation=OUT/'evaluation/summary.json'
    if status=='completed' and evaluation.exists() and read(evaluation)['status']=='dev_gate_failed':
        status='dev_gate_failed'
    write(OUT/'pipeline_status.json',dict(status=status,
        stage=stage,exit_code=code,updated_utc=datetime.now(timezone.utc).isoformat()))
else:raise ValueError(mode)
