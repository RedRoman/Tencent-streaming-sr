"""Pure-stdlib terminal metadata binding; never opens model or array payloads.

This is a new limited-dev candidate, not a completion assertion. Public bind()
requires the original immutable policy plus explicit actual terminal outer SHAs.
Only original attempt1, fresh 2000/2000 and the fixed three devs are supported.
"""
from __future__ import annotations
import argparse, ast, copy, datetime, hashlib, json, math, os, re
from pathlib import Path, PurePosixPath

POLICY_SHA = '9f047f243634ffdc4db569a23ffb98c5a107198387c3424a603e23427f74b055'
LIMIT = 64 << 20
ARMS = ('control', 'teacher')
METRICS = ('hr_pixel_l1','hr_lpips','teacher_pixel_l1','teacher_lpips','teacher_motion_l1','teacher_delta_l1')
SCHEMA = 'h3-production67-limiteddev-terminal-metadata-binding-v1'

def require(ok, message):
    if not ok: raise ValueError(message)

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def canonical(v): return sha_bytes(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def sha_value(s): require(isinstance(s,str) and re.fullmatch('[0-9a-f]{64}',s), 'Invalid SHA256')
def pairs(items):
    out={}
    for k,v in items:
        require(k not in out,'Duplicate JSON key'); out[k]=v
    return out

def parse(b):
    return json.loads(b,object_pairs_hook=pairs,parse_constant=lambda s:require(False,'Nonfinite JSON'))

def descriptor(d, path=None):
    require(isinstance(d,dict) and all(k in d for k in ('path','bytes','sha256')), 'Missing descriptor')
    p=PurePosixPath(d['path'])
    require(p.is_absolute() and str(p)==d['path'] and '..' not in p.parts, 'Noncanonical descriptor path')
    require(type(d['bytes']) is int and d['bytes']>0, 'Invalid descriptor size'); sha_value(d['sha256'])
    if path is not None: require(d['path']==str(path),'Descriptor belongs to another path/arm')

def false_claims(v):
    """Reject any recorded test access or unsupported quality/formal assertion."""
    if isinstance(v,dict):
        for k,x in v.items():
            if k in ('test_access','formal16dev_claim','formal_training_allowed','full226_dev_quality_gate_passed'):
                require(x is False,'Forbidden test/formal/full226 claim: '+k)
            if k in ('split','partition'): require(x!='test','Test partition is locked')
            false_claims(x)
    elif isinstance(v,list):
        for x in v: false_claims(x)

def finite(v):
    if isinstance(v,float): require(math.isfinite(v),'Nonfinite metadata number')
    elif isinstance(v,dict):
        for x in v.values(): finite(x)
    elif isinstance(v,list):
        for x in v: finite(x)

def stable_stat(st):
    # Access-time changes caused by our read are not mutations of the evidence.
    return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)

class Files:
    """Bounded regular-file reader restricted to explicit metadata paths."""
    def __init__(self): self.snapshot={}
    def raw(self,path):
        p=Path(path)
        require(p.is_absolute() and p.resolve()==p and p.is_file() and not p.is_symlink(),'Exact regular metadata file required')
        require(p.suffix in ('.json','.jsonl','.py','.sh','.txt'),'Payload access forbidden')
        st=p.stat(); require(0<st.st_size<=LIMIT,'Metadata exceeds bound')
        b=p.read_bytes(); require(len(b)==st.st_size and stable_stat(p.stat())==stable_stat(st),'Metadata changed while reading')
        d={'path':str(p),'bytes':len(b),'sha256':sha_bytes(b)}
        if str(p) in self.snapshot: require(self.snapshot[str(p)]==d,'Metadata changed between references')
        self.snapshot[str(p)]=d; return b
    def json(self,path,expected=None):
        b=self.raw(path)
        if expected is not None:
            descriptor(expected,path); require(self.snapshot[str(path)]=={k:expected[k] for k in ('path','bytes','sha256')},'Metadata descriptor differs')
        return parse(b)
    def identity(self,path): return self.snapshot[str(path)]
    def unchanged(self):
        prior=copy.deepcopy(self.snapshot)
        for p in prior: self.raw(p)
        require(self.snapshot==prior,'Metadata snapshot changed')

def frozen_selection(recipe_bytes, recipe_sha):
    require(sha_bytes(recipe_bytes)==recipe_sha,'Frozen teacher recipe changed')
    tree=ast.parse(recipe_bytes); names={'selection_fields','hr_guard'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes}==names and len(nodes)==2,'Frozen selection functions absent')
    # Compile only the two SHA-pinned pure functions. Never import the Torch module.
    ns={'copy':copy,'math':math}; exec(compile(ast.Module(body=nodes,type_ignores=[]),'<frozen selection AST>','exec'),ns)
    return ns['selection_fields'],ns['hr_guard']

def assert_initial(v,p):
    require(v['original_checkpoint_sha256']==p['initializer_checkpoint_sha256'] and
        v['original_checkpoint_file_verified'] is True and v['selected_step']==1800 and
        v['tensor_count']==322 and v['parameter_count']==345280216 and
        v['strict_names_shapes_dtype_values_passed'] is True and v['parameter_identity_preserved'] is True and
        v['optimizer_or_rng_restored'] is False and v['aligned_initialization_called'] is False and
        v['effective_input']==[1,24,67,44,80] and v['explicit_target']==[67,68,120], 'Fresh-v5 initial-load receipt differs')

def outer(reader,root,stage,expected_sha,policy,protocol):
    path=root+'/launches/'+stage+'_attempt1.json'; o=reader.json(path); sha_value(expected_sha)
    require(reader.identity(path)['sha256']==expected_sha,'Actual terminal outer SHA differs')
    count=0 if stage=='step0' else 2000
    require(o['stage']==stage and o['status']=='completed' and o['exit_code']==0 and
        o['optimizer_updates']==count and o['plan_sha256']==policy['execution_plan']['sha256'], 'Outer not successful terminal')
    require(o['physical_npu']==policy['physical_devices'][stage] and o['all_three_locks_held'] is True,'Wrong historical device/lock receipt')
    require(o['device_idle']['no_running_processes'] is True,'Startup idle proof missing')
    require(o['CANN_import_preflight']['staged_sources']==policy['expected_staged_sources'],'Recorded runtime source origins/SHA differ')
    c=o['cleanup']; require(c['reaped'] is True and c['returncode']==0 and c['live_group_members_at_end']==[] and
        c['pid']==o['child_pid'] and c['pgid']==o['child_pgid'] and o['child_pid']==o['child_pgid'],'Child wait/reap proof differs')
    require(datetime.datetime.fromisoformat(o['finished_utc'])>datetime.datetime.fromisoformat(o['started_utc']),'Completion time invalid')
    rp=root+'/'+stage+'/report.json'; r=reader.json(rp,o['report'])
    require(r['stage']==stage and r['plan_sha256']==policy['execution_plan']['sha256'] and
        r['protocol_sha256']==canonical(protocol) and r['physical_npu']==o['physical_npu'] and
        r['scope']=='limiteddev_train2_dev3_test0' and r['optimizer_updates_this_attempt']==count and
        r['precision']==protocol['precision'] and r['actual_mode']==protocol['mode'],'Worker terminal identity differs')
    require(r['status']==('completed_shared_limiteddev_step0' if stage=='step0' else 'completed_limiteddev2000_awaiting_full226_guard'),'Worker not complete')
    assert_initial(r['initial_load'],protocol); false_claims(r)
    require(all(r[k] is False for k in ('test_access','formal16dev_claim','full226_dev_quality_gate_passed')),'Worker scope fields missing/changed')
    mode=reader.json(root+'/'+stage+'/mode_configured_before_NPU.json')
    require(mode=={'mode':protocol['mode'],'NPU_initialized':False,'stage':stage},'Strict mode pre-init receipt differs')
    return o,r

def observation(v,baseline,p,step,fields):
    require(v['schema']=='h3-production67-sparse-dev-observation-v1' and v['step']==step and
        v['registry_sha256']==p['registry_sha256'] and v['selection_plan_sha256']==p['selection_plan_sha256'] and
        v['selection_evaluator_sha256']==p['selection_evaluator_sha256'] and v['source_sha256']==baseline['source_sha256'] and
        v['precision']==p['precision'] and v['mode']==p['mode'] and v['coverage']==baseline['coverage'] and
        v['optimizer_updates']==0,'Sparse observation identity/coverage differs')
    require([x['sample_id'] for x in v['samples']]==p['dev_order'],'Dev order differs / test locked')
    rows=[]
    for row,base in zip(v['samples'],baseline['samples']):
        require(set(row)==set(base) and {k:x for k,x in row.items() if k!='metrics'}=={k:x for k,x in base.items() if k!='metrics'},'Dev sample/array/region identity changed')
        m=row['metrics'];require(set(m)==set(METRICS) and all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in m.values()),'Invalid dev base metrics');rows.append(m)
    proof=v['readonly_proof'];require(proof['scope']=='read_only_selection' and all(proof[k] is True for k in ('preserves_existing_student_gradients','RNG_restored_exact','before_after_modules_exact')),'Read-only observation proof missing')
    sha_value(proof['module_snapshot_sha256'])
    for key in ('student','decoder','LPIPS'):sha_value(proof['tensor_value_fingerprints'][key])
    for key in ('decoder','LPIPS'):require(proof['tensor_value_fingerprints'][key]==baseline['readonly_proof']['tensor_value_fingerprints'][key],'Frozen evaluator value fingerprint differs')
    require(v['test_access'] is False and v['full226_dev_quality_gate_passed'] is False,'Observation scope fields missing/changed')
    false_claims(v);return fields({k:sum(float(row[k]) for row in rows)/len(rows) for k in METRICS})

def selected_descriptor(d,arm,step,root,p):
    require(set(d)=={'path','bytes','sha256','step','arm','protocol_sha256','initializer_checkpoint_sha256'},'Selected model descriptor schema differs')
    descriptor(d,root+'/'+arm+f'/updates/best_step_{step:04d}.pt')
    require(d['arm']==arm and d['step']==step and d['protocol_sha256']==canonical(p) and
        d['initializer_checkpoint_sha256']==p['initializer_checkpoint_sha256'],'Selected model lineage differs')
    # Deliberately do not stat/hash/open .pt. This descriptor is not payload acceptance.

def arm_terminal(reader,root,arm,report,shared,p,schedule,fields,guard,policy):
    path=root+'/'+arm+'/updates'; state=reader.json(path+'/attempt.json')
    require(state==report['result'],'Final attempt/result differs')
    require(state['schema']=='h3-production67-limiteddev-paired2000-attempt-v1' and state['status']=='completed2000_awaiting_separate_full226_dev_gate' and
        state['arm']==arm and state['protocol_sha256']==canonical(p) and state['schedule_sha256']==p['schedule_sha256'] and
        state['registry_sha256']==p['registry_sha256'] and state['mode']==p['mode'] and state['precision']==p['precision'],'Final attempt identity differs')
    require(state['completed_updates']==state['optimizer_updates_this_attempt']==state['last_confirmed_optimizer_update']==2000 and
        state['optimizer_step_started'] is False and state.get('partial_optimizer_update_possible',False) is False and
        state['resume'] is None and state['first_full_spatial_backward_admission']=='passed_actual_same_graph','Incomplete/uncertain/resumed attempt')
    require('exception' not in state and 'exception_type' not in state,'Failure retained in claimed success');false_claims(state)
    adm=state['admission'];require(adm['scope']=='explicit_user_authorized_production67_limiteddev_2train_3dev' and
        adm['protocol_sha256']==canonical(p) and adm['physical_NPU_locks_held'] is True and adm['old_failed_gates_preserved'] is True and
        adm['first_step_admission']=='full_spatial_loss_backward_then_finite_nonzero_clip_then_optimizer' and
        adm['payload_audit']==shared['payload_audit'] and report['payload_audit']==shared['payload_audit'] and
        adm['common_step0_dev']==shared['common_step0'],'Admission/common baseline mixed')
    sha_value(adm['launch_receipt_sha256']) # Running launch receipt changes on completion; no false equality to terminal SHA.
    require(reader.json(path+'/common_step0_reference.json')==shared['common_step0'],'Different shared step0 reference')
    blob=reader.raw(path+'/steps.jsonl');require(blob.endswith(b'\n'),'Incomplete durable last line')
    lines=blob.splitlines();require(len(lines)==2000 and all(lines),'Durable step count differs')
    steps=[parse(x) for x in lines];require(steps==state['steps'],'Durable steps differ from final attempt')
    for i,(row,wanted) in enumerate(zip(steps,schedule),1):
        require(sorted(row)==policy['durable_row_fields'],'Durable loss/gradient/receipt field set changed')
        require(all(type(v) in (int,float) and math.isfinite(v) for k,v in row.items() if k not in ('sample_id','spec','formal_training_preflight_passed')),'Non-numeric/nonfinite durable metric')
        require({k:row[k] for k in ('step','sample_id','spec','lr')}==wanted and row['step']==i,'Sample/spec/LPIPS/LR schedule differs')
        require(row['optimizer_updates']==1 and row['formal_training_preflight_passed'] is False and
            row['gradient_norm_before_clip']>0 and row['seconds']>=0 and row['peak_allocated']>0 and row['peak_reserved']>=row['peak_allocated'],'Durable update evidence invalid')
        finite(row)
    cps=state['checkpoints'];require(len(cps)==21 and [x['completed_updates'] for x in cps]==list(range(0,2001,100)),'Saved checkpoint receipt series incomplete')
    for d in cps:
        n=d['completed_updates'];descriptor(d,path+f'/checkpoint_step_{n:04d}.pt')
        require(set(d)=={'path','bytes','sha256','completed_updates','arm','protocol_sha256','schedule_sha256','registry_sha256'} and
            d['arm']==arm and d['protocol_sha256']==canonical(p) and d['schedule_sha256']==p['schedule_sha256'] and d['registry_sha256']==p['registry_sha256'],'Checkpoint receipt lineage differs')
        require(reader.json(path+f'/checkpoint_step_{n:04d}.receipt.json')==d,'Checkpoint receipt bytes/descriptor differs')
    baseline=reader.json(shared['common_step0']['path'],shared['common_step0'])
    s=state['selection'];require(set(s)=={'baseline','history','best_step','best_score','best_checkpoint'} and s['baseline']==baseline['metrics'],'Selection baseline differs')
    history=s['history'];require([x['step'] for x in history]==list(range(0,2001,200)),'Selection evaluations incomplete')
    best=None; obsrefs=[]
    for h in history:
        n=h['step'];require(set(h)=={'step','metrics','guard_pass'},'Selection history schema differs')
        if n==0:obs=baseline['observation']
        else:
            op=path+f'/dev_observations/step{n:04d}.json';obs=reader.json(op);obsrefs.append(reader.identity(op))
        m=observation(obs,baseline['observation'],p,n,fields)
        require(h['metrics']==m and h['guard_pass'] is bool(guard(m,baseline['metrics'])),'Selection metric/guard not frozen recipe or raw observation')
        if best is None or (h['guard_pass'] and m['selection_score']<best['metrics']['selection_score']):best=h
    require(best['guard_pass'] is True and s['best_step']==best['step'] and s['best_score']==best['metrics']['selection_score'],'Forged best / tie / reselected negative result')
    selected_descriptor(s['best_checkpoint'],arm,s['best_step'],root,p)
    return {'arm':arm,'completed_updates':2000,'durable_steps':reader.identity(path+'/steps.jsonl'),
        'attempt':reader.identity(path+'/attempt.json'),'selected_model_descriptor':s['best_checkpoint'],
        'selected_model_expected_schema':'h3-production67-limiteddev-selected-weights-v1','selected_metrics':best['metrics'],
        'selected_step':best['step'],'sparse_history':history,'observations':obsrefs,
        'checkpoint_receipts_verified':21,'model_or_optimizer_payloads_read':0,'selected_model_payload_verified':False}

def _verify(reader,policy,outer_shas,fields,guard):
    """Internal dependency-injected validator; public bind always pins policy bytes."""
    root=policy['training_root'];require(set(outer_shas)==set(ARMS),'Exactly two arm outer SHAs required')
    immutable=policy['immutable']
    p=reader.json(immutable['protocol']['path'],immutable['protocol']);schedule=reader.json(immutable['schedule']['path'],immutable['schedule'])
    selection=reader.json(immutable['selection_plan']['path'],immutable['selection_plan'])
    require(canonical(p)==policy['protocol_sha256'] and canonical(schedule)==p['schedule_sha256'] and canonical(selection)==p['selection_plan_sha256'],'Immutable protocol/schedule identity differs')
    require(p['schema']=='h3-production67-limiteddev-paired2000-protocol-v1' and p['updates_per_arm']==2000 and
        p['save_every']==100 and p['selection_every']==200 and p['seed']==20261107 and len(p['train_order'])==2 and len(p['dev_order'])==3 and
        p['train_count_required']==2 and p['dev_count_required']==3 and p['test_count_required']==0 and p['arms']==list(ARMS),'Fixed limited protocol differs')
    require(len(schedule)==2000 and [x['step'] for x in schedule]==list(range(1,2001)) and all(x['sample_id'] in p['train_order'] for x in schedule),'Invalid fixed schedule')
    _,shared=outer(reader,root,'step0',immutable['step0_outer']['sha256'],policy,p)
    require(reader.identity(root+'/step0/report.json')==immutable['step0_report'],'Shared actual step0 changed')
    for key in ('common_step0','protocol','schedule','selection_plan','payload_audit'):require(shared[key]==immutable[key],'Shared immutable descriptor differs')
    base=reader.json(immutable['common_step0']['path'],immutable['common_step0']);audit=reader.json(immutable['payload_audit']['path'],immutable['payload_audit'])
    require(base['schema']=='h3-production67-limiteddev-common-step0-v1' and base['dev_order']==p['dev_order'] and base['payload_audit']==shared['payload_audit'] and
        base['fresh_v5_identity']['all_initial_values_match_frozen_v5'] is True and base['fresh_v5_identity']['actual_tensor_count']==322 and
        base['fresh_v5_identity']['initial_tensor_schema_sha256']==p['initial_tensor_schema_sha256'],'Baseline identity invalid')
    require(observation(base['observation'],base['observation'],p,0,fields)==base['metrics'],'Baseline aggregation differs')
    require(audit['status']=='actual_five_child_payloads_verified_limiteddev_only' and audit['registry_sha256']==p['registry_sha256'] and
        audit['cohort_science_sha256']==p['cohort_science_sha256'] and audit['counts']=={'train':2,'dev':3,'test':0},'Prior payload audit identity differs')
    arms=[]
    for arm in ARMS:
        _,r=outer(reader,root,arm,outer_shas[arm],policy,p)
        item=arm_terminal(reader,root,arm,r,shared,p,schedule,fields,guard,policy)
        item['outer']=reader.identity(root+'/launches/'+arm+'_attempt1.json');item['worker_report']=reader.identity(root+'/'+arm+'/report.json');arms.append(item)
    for obj in (p,base,audit):false_claims(obj)
    reader.unchanged()
    return {'schema':SCHEMA,'status':'terminal_metadata_verified_pending_selected_payload_and_full226_eval',
        'scope':'limiteddev_train2_dev3_test0','protocol_sha256':canonical(p),'execution_plan':policy['execution_plan'],
        'registry_sha256':p['registry_sha256'],'cohort_science_sha256':p['cohort_science_sha256'],'generation_mode_semantics':p['generation_mode_semantics'],
        'precision':p['precision'],'mode':p['mode'],'train_order':p['train_order'],'dev_order':p['dev_order'],
        'actual_updates_reported_and_metadata_cross_checked':4000,'paired_schedule_sha256':p['schedule_sha256'],
        'shared_step0':shared['common_step0'],'prior_payload_audit':shared['payload_audit'],'arms':arms,
        'dev_input_identities':[{k:v for k,v in r.items() if k!='metrics'} for r in base['observation']['samples']],
        'metadata_snapshot':copy.deepcopy(reader.snapshot),'metadata_before_after_equal':True,
        'selected_model_payloads_verified':False,'current_array_payloads_reverified':False,'test_access':False,
        'formal16dev_claim':False,'full226_dev_quality_gate_passed':False,'visual_quality_passed':False,
        'next_required':'Verify selected model-only file bytes/SHA/schema/arm/step/protocol/322 tensors, revalidate actual dev payloads, then separate full226 evaluation; no reselection or test access.'}

def bind(policy_path,control_outer_sha256,teacher_outer_sha256):
    r=Files();raw=r.raw(policy_path);require(sha_bytes(raw)==POLICY_SHA,'Exact reviewed terminal policy required');policy=parse(raw)
    plan=r.json(policy['execution_plan']['path'],policy['execution_plan'])
    require(plan['output']==policy['training_root'] and plan['counts']=={'train':2,'dev':3,'test':0},'Execution plan differs')
    for d in plan['files'].values():
        descriptor(d);r.raw(d['path']);require(r.identity(d['path'])==d,'Frozen training source differs')
    recipe=policy['teacher_recipe'];fields,guard=frozen_selection(r.raw(recipe['path']),recipe['sha256'])
    result=_verify(r,policy,{'control':control_outer_sha256,'teacher':teacher_outer_sha256},fields,guard)
    result['binding_source_sha256']=sha_bytes(Path(__file__).read_bytes());result['policy_sha256']=POLICY_SHA
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('policy','control-outer-sha256','teacher-outer-sha256','output'):ap.add_argument('--'+name,required=True)
    a=ap.parse_args();out=Path(a.output)
    require(out.is_absolute() and not out.exists() and out.parent.resolve()==out.parent,'New absolute output required')
    require(not str(out).startswith('/path/to/h3-project/upscaler_production67_limiteddev_20261107/'),'Do not write training directory')
    value=bind(a.policy,a.control_outer_sha256,a.teacher_outer_sha256)
    with out.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({'path':str(out),'bytes':out.stat().st_size,'sha256':sha_bytes(out.read_bytes())}))
