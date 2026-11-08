"""Bounded new deterministic production67 restore core; no model factory or launch.
The caller injects exact reviewed loaded modules and a root-bound real-data contract.
This module does not admit a device, claim trainingREADY, or run a formal2000 loop.
"""
from __future__ import annotations
import copy,hashlib,inspect,json,os,random,time
from pathlib import Path
import torch
import h3_production67_train_step_v1 as core

SCHEMA='h3-production67-deterministic-restore-engineering-v1'
MODE={'deterministic_algorithms':True,'warn_only':False}
STAGES={'continuous':(3,8,5),'branch_start':(3,7,4),'branch_finish':(6,8,2)}
RESTORE_SOURCE_SCHEMA='h3-production67-paired8-engineering-v3'
SEED=20261107
SOURCE_SHA='32bd9154fdbae7b9e5cb8ac16029014da3dae9227d276a3e90d5f37ea7877997'


def require(ok,message):
 if not ok:raise ValueError(message)


def canonical_sha(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()


def atomic_json(path,value):
 path=Path(path);tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
 with tmp.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)


def cpu_tree(x):
 if torch.is_tensor(x):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu_tree(v) for v in x]
 if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
 return copy.deepcopy(x)


def exact_tree(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and tuple(a.shape)==tuple(b.shape) and torch.equal(a.detach().cpu(),b.detach().cpu())
 if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(exact_tree(a[k],b[k]) for k in a)
 if type(a)!=type(b):return False
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(exact_tree(x,y) for x,y in zip(a,b))
 return a==b


def capture_rng(device):
 device=torch.device(device)
 return {'torch_cpu':torch.get_rng_state().clone(),'device':torch.npu.get_rng_state(device).cpu().clone() if device.type=='npu' else None,'python':random.getstate()}


def restore_rng(value,device):
 device=torch.device(device);torch.set_rng_state(value['torch_cpu']);random.setstate(value['python'])
 if device.type=='npu':require(value['device'] is not None,'Missing selected-device RNG');torch.npu.set_rng_state(value['device'],device)
 else:require(value['device'] is None,'CPU checkpoint carries accelerator RNG')
 require(exact_tree(capture_rng(device),value),'Live RNG restoration is not exact')


def optimizer_counters(optimizer,parameters,updates):
 parameters=tuple(parameters);require(type(updates) is int and updates>=0,'Bad update counter')
 require({id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in parameters},'Optimizer/complete parameter identity differs')
 if updates==0:require(len(optimizer.state)==0,'Fresh AdamW must be empty');return
 require(len(optimizer.state)==len(parameters),'Missing optimizer states')
 for p in parameters:
  s=optimizer.state[p];require(set(s)=={'step','exp_avg','exp_avg_sq'},'Unexpected AdamW state')
  require(torch.is_tensor(s['step']) and s['step'].numel()==1 and float(s['step'])==updates,'Per-parameter AdamW step differs')
  for name in ('exp_avg','exp_avg_sq'):
   v=s[name];require(v.shape==p.shape and v.dtype==torch.float32 and v.device==p.device and bool(torch.isfinite(v).all()),'Bad optimizer moment shape/dtype/device/finite')


def deterministic_mode():
 got={'deterministic_algorithms':bool(torch.are_deterministic_algorithms_enabled()),'warn_only':bool(torch.is_deterministic_algorithms_warn_only_enabled())}
 require(got==MODE,'Strict deterministic mode must already be configured before NPU initialization');return got


def stage_bounds(stage):
 require(stage in STAGES,'Unknown bounded deterministic stage');return STAGES[stage]


def checkpoint_lineage(stage,original_protocol_sha,original_checkpoint):
 require(stage in STAGES and isinstance(original_protocol_sha,str) and len(original_protocol_sha)==64,'Explicit original protocol required')
 require(original_checkpoint['arm']=='control' and original_checkpoint['completed_updates']==3 and original_checkpoint['protocol_sha256']==original_protocol_sha,'Original control checkpoint3 lineage required')
 return {'kind':'new_deterministic_restore_protocol','stage':stage,'original_control_checkpoint3':copy.deepcopy(original_checkpoint),'original_update_protocol_sha256':original_protocol_sha,'execution_is_original_run':False}


def save_checkpoint(path,*,model,optimizer,device,updates,arm,protocol_sha,schedule_sha,stage,original_protocol_sha,original_checkpoint):
 path=Path(path);require(not path.exists(),'Never overwrite preflight checkpoints');optimizer_counters(optimizer,model.parameters(),updates);mode=deterministic_mode()
 require(arm=='control' and protocol_sha!=original_protocol_sha and updates in ({6,8} if stage=='continuous' else {6} if stage=='branch_start' else {8}) and stage in STAGES,'Wrong bounded save lineage')
 obj={'schema':SCHEMA,'arm':arm,'completed_updates':updates,'protocol_sha256':protocol_sha,'original_protocol_sha256':original_protocol_sha,'schedule_sha256':schedule_sha,'model':cpu_tree(model.state_dict()),'optimizer':cpu_tree(optimizer.state_dict()),'rng':capture_rng(device),'formal_training_allowed':False,'mode':mode,'lineage':checkpoint_lineage(stage,original_protocol_sha,original_checkpoint)}
 tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
 with tmp.open('xb') as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);receipt={'path':str(path),'bytes':path.stat().st_size,'sha256':file_sha(path),'completed_updates':updates,'arm':arm,'protocol_sha256':protocol_sha,'original_protocol_sha256':original_protocol_sha,'schedule_sha256':schedule_sha,'stage':stage}
 atomic_json(path.with_suffix('.receipt.json'),receipt);return receipt


def restore_checkpoint(receipt,*,model,optimizer,device,arm,protocol_sha,original_protocol_sha,schedule_sha,stage,original_checkpoint):
 expected_updates,_,_=stage_bounds(stage);deterministic_mode();require(arm=='control' and protocol_sha!=original_protocol_sha,'New deterministic protocol must differ from old protocol')
 p=Path(receipt['path']);require(p.is_file() and not p.is_symlink() and p.stat().st_size==receipt['bytes'] and file_sha(p)==receipt['sha256'],'Explicit checkpoint identity differs; do not deserialize')
 expected_schema=SCHEMA if stage=='branch_finish' else RESTORE_SOURCE_SCHEMA
 source_protocol=protocol_sha if stage=='branch_finish' else original_protocol_sha
 if stage!='branch_finish':require(receipt==original_checkpoint,'First two stages must restore the exact original control3')
 saved=torch.load(p,map_location='cpu',weights_only=False)
 require(saved['schema']==expected_schema and saved['arm']==arm==receipt['arm'] and saved['protocol_sha256']==source_protocol==receipt['protocol_sha256'] and saved['schedule_sha256']==schedule_sha==receipt['schedule_sha256'] and saved['formal_training_allowed'] is False,'Checkpoint lineage differs')
 updates=saved['completed_updates'];require(type(updates) is int and updates==expected_updates and updates==receipt['completed_updates'],'Stage restores only its explicit control3 or branch6')
 if stage=='branch_finish':
  require(saved['mode']==MODE and saved['original_protocol_sha256']==receipt['original_protocol_sha256']==original_protocol_sha and receipt['stage']=='branch_start','Only new deterministic branch checkpoint6 may be resumed')
  require(saved['lineage']==checkpoint_lineage('branch_start',original_protocol_sha,original_checkpoint),'New branch checkpoint original lineage differs')
 expected=model.state_dict();require(set(saved['model'])==set(expected) and all(v.shape==expected[k].shape and v.dtype==expected[k].dtype==torch.float32 and bool(torch.isfinite(v).all()) for k,v in saved['model'].items()),'Checkpoint model schema invalid')
 ids={n:id(v) for n,v in model.named_parameters()};model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
 require(ids=={n:id(v) for n,v in model.named_parameters()},'Resume replaced student parameters');optimizer_counters(optimizer,model.parameters(),updates)
 require(exact_tree(model.state_dict(),saved['model']) and exact_tree(optimizer.state_dict(),saved['optimizer']),'Live restored model/optimizer state not exact')
 restore_rng(saved['rng'],device);require(file_sha(p)==receipt['sha256'],'Checkpoint changed during restore');deterministic_mode()
 return updates,{'restored_updates':updates,'model_bitwise_exact':True,'optimizer_bitwise_exact':True,'CPU_and_selected_device_RNG_exact':True,'checkpoint':copy.deepcopy(receipt),'source_schema':expected_schema,'source_protocol_sha256':source_protocol,'new_execution_protocol_sha256':protocol_sha,'lineage':'new_deterministic_branch6' if stage=='branch_finish' else 'original_control3_external_not_owned_by_this_experiment'}


def build_paired_schedule(sample_ids,recipe):
 require(core.file_sha(inspect.getfile(core))==SOURCE_SHA,'Reviewed train-step changed')
 sampler=core.Production67ScheduleSampler()
 s=recipe.paired_schedule(sample_ids,sampler,steps=8,seed=SEED,lr=1e-5,min_lr=1e-6,warmup=100,full_every=4)
 require([r['step'] for r in s]==list(range(1,9)),'Schedule continuity differs')
 for i,row in enumerate(s):
  require(row['spec']['region']['spatial']['spatial_mode']==('full_spatial' if i%4==0 else 'stable_tile_interior'),'Original step0 full-spatial schedule must never be replaced by tiles')
 return s


def check_frozen(provider,lpips_fn):
 provider.validate_frozen();require(all(not m.training for m in lpips_fn.modules()),'LPIPS must be eval')
 for p in list(lpips_fn.parameters())+list(lpips_fn.buffers()):
  require(not p.requires_grad and p.grad is None and (not p.is_floating_point() or p.dtype==torch.float32),'LPIPS not frozen FP32')


def precision(device):
 device=torch.device(device);require(device.type=='npu','Production preflight requires actual NPU')
 d={'dtype':'float32','conv_hf32':bool(torch.npu.conv.allow_hf32),'matmul_hf32':bool(torch.npu.matmul.allow_hf32),'float32_matmul_precision':torch.get_float32_matmul_precision()}
 require(d=={'dtype':'float32','conv_hf32':False,'matmul_hf32':False,'float32_matmul_precision':'highest'},'Require new bound HF32-off precision');return d


def load_bound_example(cache_root,record,data_contract):
 loader=core.strict_loader();root=Path(cache_root).resolve()
 def subroot(name):
  rel=Path(record[name]);require(not rel.is_absolute() and '..' not in rel.parts,'Relative bounded cache roots required')
  p=(root/rel).resolve();require(p.is_relative_to(root) and p.is_dir(),'Pair/teacher root escapes cache');return p
 pair=loader.load_pair(subroot('pair_root'),record['pair_record_path'],record['pair_record_sha256'],data_contract)
 teacher=loader.load_teacher(subroot('teacher_root'),record['teacher_record_path'],record['teacher_record_sha256'],pair,data_contract)
 require(pair['record']['sample_id']==record['sample_id'] and pair['record']['split']==record['split']=='train','Actual pair source identity differs')
 values=loader.torch_arrays(pair)
 example={'draft':values['draft'].unsqueeze(0),'reference':values['reference_hr'].unsqueeze(0),'teacher':loader.torch_teacher(teacher).unsqueeze(0),'flow':values['flow'],'valid':values['flow_valid'],'pair_bundle':pair,'teacher_bundle':teacher,'contract':copy.deepcopy(data_contract),'scope':'candidate_checked_cache_not_training_READY'}
 core.validate_example_tensors(example);return example


def run_arm(*,student,provider,lpips_fn,recipe,device,cache_root,data_contract,rows,schedule,output,arm,protocol_sha,original_protocol_sha,stage,admission,initial_tensor_schema,resume_receipt=None):
 """Bounded new deterministic protocol: 5 + 4 + 2 actual updates.

 The unchanged numerical loop uses the original loss/schedule/AdamW recipe.
 Strict mode is set by the reviewed worker before NPU initialization and is
 verified here, never switched here. Original failed gates remain failed.
 Branch_start stops normally at7 after saving6; it is not crash recovery.
 """
 device=torch.device(device);prec=precision(device);mode=deterministic_mode();expected_start,stop_after,max_updates=stage_bounds(stage)
 require(arm in ('control','teacher') and isinstance(protocol_sha,str) and len(protocol_sha)==64,'Wrong arm/protocol')
 require(arm=='control' and resume_receipt is not None and isinstance(original_protocol_sha,str) and len(original_protocol_sha)==64 and protocol_sha!=original_protocol_sha,'Only explicit new deterministic control protocol supported')
 require(admission['scope']=='real_production67_deterministic_restore_engineering' and admission['formal_training_allowed'] is False and admission['native_HR_pair_accepted'] is True and admission['physical_NPU_locks_held'] is True,'Data/launcher admission missing')
 require(admission['first_step_admission']=='full_spatial_loss_backward_then_finite_nonzero_clip_then_optimizer','First-step same-graph admission must be explicit')
 for key in ('data_acceptance_sha256','launch_receipt_sha256'):require(type(admission[key]) is str and len(admission[key])==64,'Missing externally checked admission SHA')
 require(provider.allow_full_spatial_grad is True and provider.checkpoint_decode is True,'Explicit full-spatial checkpoint opt-in required, not automatic tile substitution')
 require(student.source_checkpoint_sha256==core.V5_SHA and student.model_config['kind']=='h3_causal_conv3d_explicit_target_v1' and getattr(student,'_production67_block_checkpointing',False) is True,'Exact reviewed v5/checkpointed explicit wrapper required')
 student.eval();require(all(p.requires_grad and p.dtype==torch.float32 and p.device==device for p in student.parameters()),'Student trainable FP32 on one actual device')
 require(len(tuple(student.parameters()))==322 and sum(p.numel() for p in student.parameters())==345280216,'Wrong actual student size')
 # Each arm is built fresh from selectedv5 even before a controlled resume.
 st=student.original_state_dict();require(set(st)==set(initial_tensor_schema),'Initial v5 schema differs')
 for name,t in st.items():
  c=t.detach().cpu().contiguous();require(list(c.shape)==initial_tensor_schema[name]['shape'] and hashlib.sha256(memoryview(c.numpy()).cast('B')).hexdigest()==initial_tensor_schema[name]['tensor_sha256'],'Fresh v5 value differs: '+name)
 check_frozen(provider,lpips_fn);loader=core.strict_loader();loader.validate_contract(data_contract)
 require(rows and all(r['split']=='train' for r in rows) and len({r['sample_id'] for r in rows})==len(rows),'Only unique fixed train source probes allowed')
 by_id={r['sample_id']:r for r in rows};expected=build_paired_schedule(list(by_id),recipe);require(schedule==expected,'Paired8 sample/spec/LPIPS/LR schedule differs');sched_sha=canonical_sha(schedule)
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 require(admission['external_common_prefix_checkpoint']==resume_receipt and resume_receipt['completed_updates']==expected_start and resume_receipt['arm']=='control','Exact explicitly admitted bounded prefix required')
 original_checkpoint=admission['original_control_checkpoint3'];lineage=checkpoint_lineage(stage,original_protocol_sha,original_checkpoint)
 require(not Path(resume_receipt['path']).resolve().is_relative_to(output.resolve()),'Do not relabel original checkpoint as probe-owned')
 attempts=list(output.glob('attempt*.json'));attempt=len(attempts)+1;status_path=output/f'attempt{attempt}.json';require(not status_path.exists(),'Attempt collision')
 torch.manual_seed(SEED);random.seed(SEED);torch.npu.manual_seed(SEED)
 optimizer=core.fresh_adamw(student.parameters(),lr=1e-5);optimizer_counters(optimizer,student.parameters(),0)
 start=0;restored=None
 if resume_receipt is not None:start,restored=restore_checkpoint(resume_receipt,model=student.model,optimizer=optimizer,device=device,arm=arm,protocol_sha=protocol_sha,original_protocol_sha=original_protocol_sha,schedule_sha=sched_sha,stage=stage,original_checkpoint=original_checkpoint)
 state={'schema':SCHEMA,'status':'running','lineage':lineage,'stage':stage,'original_protocol_sha256':original_protocol_sha,'mode':mode,'max_new_updates':max_updates,'arm':arm,'protocol_sha256':protocol_sha,'schedule_sha256':sched_sha,'schedule':schedule,'precision':prec,'completed_updates':start,'optimizer_updates_this_attempt':0,'first_full_spatial_backward_admission':'original_control_step1_preserved_not_repeated_here','optimizer_step_started':False,'resume':restored,'admission':copy.deepcopy(admission),'formal_training_allowed':False,'original_default_HF32_gate_passed':False,'checkpoints':[],'steps':[]};atomic_json(status_path,state)
 before_frozen=[(id(t),t._version) for m in (provider.vae,lpips_fn) for t in list(m.parameters())+list(m.buffers())]
 try:
  with (output/f'steps_attempt{attempt}.jsonl').open('x') as log:
   for row in schedule[start:stop_after]:
    deterministic_mode();require(precision(device)==prec,'Precision changed before update')
    record=by_id[row['sample_id']]
    example=load_bound_example(cache_root,record,data_contract)
    example=core.move_example(example,device=device)
    for g in optimizer.param_groups:g['lr']=row['lr']
    torch.npu.synchronize(device);torch.npu.reset_peak_memory_stats(device);beg=time.monotonic()
    loss,terms,pixels=core.loss_for_example(student,example,row['spec']['region'],lpips_packed_positions=row['spec']['lpips_packed_positions'],temporal_scope='nonblend_only',decode_region=provider.decode_region,lpips_fn=lpips_fn,recipe=recipe,arm=arm)
    metrics={'loss':float(loss.detach()),**{k:float(v.detach()) for k,v in {**terms,**pixels}.items()}}
    # The reviewed function performs backward, finite/nonzero norm+clip,
    # then optimizer.step, in that order. A failure before optimizer.step
    # has zero new updates. Optimizer exceptions are marked uncertain, not
    # falsely labeled zero or automatically retried.
    optimizer.zero_grad(set_to_none=True);loss.backward()
    params=tuple(student.parameters());require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params),'Missing/nonfinite gradient before optimizer')
    norm=torch.nn.utils.clip_grad_norm_(params,1.,foreach=False,error_if_nonfinite=True);require(float(norm)>0,'Zero gradient before optimizer')
    state['first_full_spatial_backward_admission']='passed_actual_same_graph' if row['step']==1 else state['first_full_spatial_backward_admission']
    state['optimizer_step_started']=True;state['optimizer_step_pending_update']=row['step'];atomic_json(status_path,state)
    optimizer.step();torch.npu.synchronize(device);state['optimizer_step_started']=False
    state['completed_updates']=row['step'];state['optimizer_updates_this_attempt']+=1;state['last_confirmed_optimizer_update']=row['step'];atomic_json(status_path,state)
    update={'gradient_norm_before_clip':float(norm),'optimizer_updates':1,'formal_training_preflight_passed':False}
    optimizer_counters(optimizer,student.parameters(),row['step']);check_frozen(provider,lpips_fn);require(precision(device)==prec,'Precision changed during updates');deterministic_mode()
    rec={**copy.deepcopy(row),**metrics,**update,'seconds':time.monotonic()-beg,'peak_allocated':torch.npu.max_memory_allocated(device),'peak_reserved':torch.npu.max_memory_reserved(device)}
    log.write(json.dumps(rec,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno());state['steps'].append(rec)
    del loss,terms,pixels,example
    if row['step'] in (3,6,8):state['checkpoints'].append(save_checkpoint(output/f'checkpoint_step_{row["step"]:03d}.pt',model=student.model,optimizer=optimizer,device=device,updates=row['step'],arm=arm,protocol_sha=protocol_sha,schedule_sha=sched_sha,stage=stage,original_protocol_sha=original_protocol_sha,original_checkpoint=original_checkpoint))
    atomic_json(status_path,state)
  after_frozen=[(id(t),t._version) for m in (provider.vae,lpips_fn) for t in list(m.parameters())+list(m.buffers())];require(before_frozen==after_frozen,'Frozen decoder/LPIPS state version changed')
  deterministic_mode();require(precision(device)==prec,'Precision changed at completion')
  require(state['completed_updates']==stop_after and state['optimizer_updates_this_attempt']==max_updates and [x['step']for x in state['steps']]==list(range(expected_start+1,stop_after+1)),'Bounded stage actual updates differ')
  expected_saves={'continuous':[6,8],'branch_start':[6],'branch_finish':[8]}[stage]
  require([x['completed_updates'] for x in state['checkpoints']]==expected_saves,'Bounded save cadence changed')
  state['status']='controlled_stop_after_unsaved_step7' if stage=='branch_start' else 'completed_bounded_deterministic_updates_not_formal_training'
  state['termination_kind']='normal_return_not_signal_or_crash';state['unsaved_tail_updates']=[7] if stage=='branch_start' else []
  atomic_json(status_path,state);return state
 except BaseException as e:
  state['partial_optimizer_update_possible']=state.get('optimizer_step_started',False)
  state.update(status='failed_no_automatic_retry',exception_type=type(e).__name__,exception=str(e));atomic_json(status_path,state);raise
