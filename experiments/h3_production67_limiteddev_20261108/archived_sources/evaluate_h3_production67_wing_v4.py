"""Fixed v5/control8/teacher8 on one native Wing dev pair. No training or test.
Separate infer and pixels stages. Parent launcher must own device/three locks.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,ast,hashlib,importlib.util,inspect,json,os,sys,time,math,re,subprocess
ROOT='/path/to/h3-project'
STAGE='/path/to/staging/h3_production67_wing_eval_20261107_v4'
OUTPUT=ROOT+'/upscaler_production67_wing_engineering_20261107/evaluation_v4'
SOURCE_ID='WingIt_dev_3586_3811'
ADAPTER='0147d14f0aa31802eccb9fd7bcb4b6751991f2d70e2e7ab244e6aa079c0296f4'
RGBSEQ='9597a1492860fa87f2e61e796ee5fdb31094ad9a1554c4598421da70d8af1ba9'
V5='c7aaaa396d877a84c3fed2a5536189b7ed5e61226a82ef1a9e6be1e831f7f914'
MODELS=('v5','control','teacher_student')
STAGES=('infer','pixels')
FRAMES=226

def require(v,m):
 if not v:raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def record(p):
 p=Path(p);require(p.is_file() and not p.is_symlink(),'Regular file required');return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def module(name,p):
 require(name not in sys.modules,'Unexpected imported module');s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def verify_plan(path,digest):
 require(sha(path)==digest,'Evaluation source plan SHA differs');p=read(path)
 require(p['schema']=='h3-production67-singleWing-evaluation-plan-v4' and p['stage_directory']==STAGE and p['output']==OUTPUT and p['stages']==list(STAGES),'Wrong explicit evaluation scope')
 require(p['precision']=={'dtype':'float32','conv_hf32':False,'matmul_hf32':False,'float32_matmul_precision':'highest'},'Wrong explicit precision')
 for n,r in p['files'].items():require(r['path']==STAGE+'/'+n and record(r['path'])==r,'Staged evaluation source differs '+n)
 require(p['files'][Path(__file__).name]['path']==str(Path(__file__).resolve()),'Exact staged worker required');return p

def bind_data(data_root):
 root=Path(data_root).resolve();rp=root/'npu_report.json';d=read(rp);contract=read(root/'data_contract.json')
 require(d['status']=='completed' and d['stage']=='npu' and d['actual_encodes']==2 and d['actual_teacher_forwards']==1 and d['actual_decoder_forwards']==0 and d['optimizer_updates']==0 and d['training_ready'] is False,'Actual Wing data preparation incomplete')
 require(d['source_adapter']['sha256']==ADAPTER and d['native_RGB_sequence_sha256']==RGBSEQ and d['source_before_after_SHA_equal'] is True,'Wrong accepted native Wing source')
 require(d['precision']['tensor_dtype']=='float32' and d['precision']['conv_allow_hf32'] is False and d['precision']['matmul_allow_hf32'] is False and canonical(d['precision'])==contract['generation_binding']['execution_precision_sha256'],'Wing generation precision differs')
 pair=record(root/'pair/pair.json');teacher=record(root/'teacher/teacher.json');require(pair==d['pair'] and teacher==d['teacher'],'Data report identities differ')
 pr=read(pair['path']);tr=read(teacher['path']);require(all(x['sample_id']==SOURCE_ID and x['source_group']=='wing_it' and x['split']=='dev' for x in (pr,tr)),'Only fixed Wing dev accepted')
 require((root/'data_contract.json').read_bytes()==(root/'pair/data_contract.json').read_bytes(),'Pair/root contract differs')
 return {'root':str(root),'report':record(rp),'contract':record(root/'data_contract.json'),'pair':pair,'teacher':teacher,'source_adapter':d['source_adapter'],'native_RGB_sequence_sha256':RGBSEQ}

def deterministic_restore_admission(plan,training_root,comparison,comparison_sha256):
 a=plan['engineering_admission'];tr=Path(training_root)
 require(a['schema']=='h3-deterministic11-pass-singleWing-readonly-admission-v1' and str(tr)==a['training_root'] and a['old_failure_remains_failed'] is True and a['old_common_prefix_failure_remains_failed'] is True and a['formal_training_allowed'] is False and a['test_unlocked'] is False and a['scope']=='singleWing_engineering_metric_readout_only','Wrong limited engineering admission')
 require(sha(comparison)==comparison_sha256,'Actual deterministic restore report SHA differs')
 q=read(comparison);checks=q['comparisons']
 require(q['status']=='completed_distinct_deterministic_restore_comparison' and q['deterministic_restore_comparison_passed'] is True and set(checks)=={'step_metrics','model','optimizer','rng'} and all(v['passed'] is True for v in checks.values()) and q['branch_before_stop_vs_continuous']['passed'] is True and q['unsaved7_vs_replayed7']['passed'] is True,'Actual deterministic comparisons, prefix and unsaved7 replay must all pass')
 require(q['source']==a['deterministic_checker'] and q['source_plan']==a['deterministic_source_plan'] and q['original_checker']==a['deterministic_old_checker'] and q['original_comparator']==a['deterministic_old_comparator'],'Wrong reviewed deterministic checker/source lineage')
 for k in ('source','source_plan','original_checker','original_comparator'):require(record(q[k]['path'])==q[k],'Deterministic evidence changed '+k)
 require(q['original_checker']['sha256']==a['original_checker']['sha256'] and q['original_comparator']['sha256']=='f28046f0551815d8bfd306d96dd7bd9711a546159e4bf956649bc9a376c7ac0f','Original checker/comparator changed')
 dp=read(q['source_plan']['path']);mode={'deterministic_algorithms':True,'warn_only':False};counts={'continuous':5,'branch_start':4,'branch_finish':2};indices={'continuous':[4,5,6,7,8],'branch_start':[4,5,6,7],'branch_finish':[7,8]}
 require(dp['mode']==mode and dp['precision']==plan['precision'] and dp['original_control']==a['deterministic_original_control'] and dp['restore_tolerances']==a['tolerances'] and dp['deterministic_protocol_sha256']==a['new_deterministic_protocol_sha256'],'Deterministic plan recipe/lineage changed')
 require(q['new_protocol_sha256']==a['new_deterministic_protocol_sha256']!=a['original_protocol_sha256'] and q['original_update_protocol_sha256']==a['original_protocol_sha256'] and q['original_control_checkpoint3']==a['deterministic_original_control']['checkpoint3'],'New and original protocol identity differ')
 require(q['new_optimizer_updates']==11 and q['stage_update_counts']==counts and q['compared_logical_steps']==[4,5,6,7,8] and q['actual_branch_stop_step']==7 and q['actual_branch_save_and_restore_step']==6 and q['unchanged_tolerances']==a['tolerances'],'Actual deterministic 5+4+2 counts/steps/tolerances differ')
 require(q['original_restore_acceptance_remains_failed'] is True and q['old_common_prefix_acceptance_remains_failed'] is True and q['old_failure_modified'] is False and q['formal_training_allowed'] is False and q['read_only_checker_NPU_or_optimizer_actions']==0,'New result cannot upgrade old failures or formal admission')
 require(q['bound_prior_evidence']==dp['evidence'] and dp['evidence']['old_common_prefix_failure']==a['old_common_prefix_failure'],'Prior failed/common evidence differs')
 for rc in q['bound_prior_evidence'].values():require(record(rc['path'])==rc,'Bound prior evidence changed')
 require(set(q['stage_reports'])==set(counts),'Three real deterministic stages required')
 stage_states={}
 for name,n in counts.items():
  rc=q['stage_reports'][name];require(rc['path']==str(Path(dp['output'])/name/'report.json') and record(rc['path'])==rc,'Actual deterministic stage report changed')
  r=read(rc['path']);state=r['result'];stage_states[name]=state
  require(r['status']=='completed_deterministic_engineering_stage' and r['stage']==name and r['plan_sha256']==q['source_plan']['sha256'] and r['protocol_sha256']==q['new_protocol_sha256'] and r['original_update_protocol_sha256']==a['original_protocol_sha256'] and r['actual_mode']==mode and r['precision']==plan['precision'],'Actual deterministic stage identity/mode differs')
  require(r['optimizer_updates_this_attempt']==state['optimizer_updates_this_attempt']==n and state['completed_updates']==indices[name][-1] and [x['step']for x in state['steps']]==indices[name],'Actual deterministic update rows/count differ')
  require(state['optimizer_step_started'] is False and state.get('partial_optimizer_update_possible',False) is False and state['termination_kind']=='normal_return_not_signal_or_crash' and state['unsaved_tail_updates']==([7]if name=='branch_start'else[]),'Actual deterministic completion or unsaved tail differs')
  require(state['resume']['restored_updates']==(6 if name=='branch_finish'else 3) and all(state['resume'][k] is True for k in ('model_bitwise_exact','optimizer_bitwise_exact','CPU_and_selected_device_RNG_exact')),'Actual deterministic live restoration must be exact')
  require(r['formal_training_allowed'] is False and r['old_restore_acceptance_remains_failed'] is True and r['old_common_prefix_acceptance_remains_failed'] is True,'Actual stage upgraded old failure or formal scope')
  previous=None if name=='continuous'else q['stage_reports']['continuous'if name=='branch_start'else'branch_start']['sha256']
  require(r['predecessor_report_sha256']==previous,'Stage predecessor changed')
  expected_source=a['deterministic_original_control']['checkpoint3']if name!='branch_finish'else stage_states['branch_start']['checkpoints'][0]
  require(r['source_checkpoint']==state['resume']['checkpoint']==expected_source,'Actual stage resume lineage changed')
  require([x['completed_updates']for x in state['checkpoints']]==({'continuous':[6,8],'branch_start':[6],'branch_finish':[8]}[name]),'Actual save6/8 cadence changed')
  require(state['status']==('controlled_stop_after_unsaved_step7'if name=='branch_start'else'completed_bounded_deterministic_updates_not_formal_training'),'Actual stage completion status differs')
 for k in ('old_failure_receipt','old_failure_log','old_final_diagnostic','original_plan','original_checker','old_common_prefix_failure'):require(record(a[k]['path'])==a[k],'Bound original evidence changed '+k)
 require(read(a['old_failure_receipt']['path'])['exit_code']==1 and not (Path(a['old_failure_receipt']['path']).parent/'restore_acceptance.json').exists(),'Original failed acceptance must remain failed/absent')
 d=read(a['old_final_diagnostic']['path']);require(d['status']=='diagnostic_only_original_acceptance_remains_failed' and d['formal_training_allowed'] is False and d['paired8_and_restore_engineering_passed'] is False and d['original_failure_receipt']==a['old_failure_receipt'] and d['original_failure_log']==a['old_failure_log'] and d['unchanged_comparator_results']['optimizer']['passed'] is False,'Original optimizer failure must remain explicit')
 old=read(a['old_common_prefix_failure']['path']);require(old['status']=='completed_common_prefix_comparison' and old['common_prefix_comparison_passed'] is False and old['comparisons']['optimizer']['passed'] is False and old['original_restore_acceptance_remains_failed'] is True and old['formal_training_allowed'] is False,'Old common-prefix optimizer failure must remain failed')
 return a,q

def paired8_proof(a):
 # Reuse reviewed JSON-only stage/schedule checks; no repeated large checkpoint torch.load.
 name='wing_eval_original_paired_checker';C=sys.modules.get(name) or module(name,a['original_checker']['path'])
 require(C.TOL==a['tolerances'],'Old tolerances changed');tr=Path(a['training_root']);reports={};checkpoints={};schedule=None
 for label,arm in (('control','control'),('teacher_student','teacher')):
  require(record(tr/arm/'report.json')==a['reports'][arm],'Actual fixed arm report changed')
  r,s,_=C.stage(tr,arm,a['original_plan']['sha256'],a['original_data_binding_sha256'],a['original_protocol_sha256'])
  if schedule is None:schedule=s['schedule']
  require(len(schedule)==8 and [x['step']for x in schedule]==list(range(1,9)) and s['schedule']==schedule and s['schedule_sha256']==canonical(schedule)==a['original_schedule_sha256'],'Paired sample/spec/LPIPS/LR schedule differs')
  require([x['spec']['region']['spatial']['spatial_mode']for x in schedule]==['full_spatial'if i%4==0 else 'stable_tile_interior'for i in range(8)],'Original full-spatial steps1/5 changed')
  require(s['arm']==arm and s['resume'] is None and [x['completed_updates']for x in s['checkpoints']]==[3,6,8],'Original arm must be fresh8 with saves3/6/8')
  C.check_steps(s['steps'],schedule,list(range(1,9)))
  for rc in s['checkpoints']:require(rc==read(Path(rc['path']).with_suffix('.receipt.json')),'Original save sidecar changed')
  cp=s['checkpoints'][-1];require(cp==a['final_checkpoints'][arm] and record(cp['path'])=={k:cp[k]for k in ('path','bytes','sha256')},'Exact original step8 SHA differs')
  require(cp['arm']==arm and cp['completed_updates']==8 and cp['protocol_sha256']==a['original_protocol_sha256'] and cp['schedule_sha256']==a['original_schedule_sha256'],'Checkpoint8 lineage differs')
  checkpoints[label]=cp;reports[label]=record(tr/arm/'report.json')
 return reports,checkpoints,{'source':a['original_checker'],'same_full_schedule_sha256':a['original_schedule_sha256'],'continuous_steps_each':[1,2,3,4,5,6,7,8],'fresh_v5_initialization_both_verified':True,'same_sample_spec_LPIPS_LR_each_step':True,'completed_optimizer_updates_each':8,'proof_scope':'JSON durable stage/step/admission records plus exact final checkpoint SHA; no duplicate checkpoint optimizer reload; not the failed global restore acceptance.'}

def make_binding(data_root,training_root,comparison,comparison_sha256,plan):
 data=bind_data(data_root);a,q=deterministic_restore_admission(plan,training_root,comparison,comparison_sha256);reports,checkpoints,proof=paired8_proof(a)
 return {'schema':'h3-production67-singleWing-evaluation-binding-v4','sample_id':SOURCE_ID,'split':'dev','formal16dev':False,'checkpoint_selection':False,'formal_training_allowed':False,'test_unlocked':False,'data':data,'deterministic_restore_comparison':record(comparison),'new_deterministic_protocol_sha256':a['new_deterministic_protocol_sha256'],'old_common_prefix_failure':a['old_common_prefix_failure'],'old_common_prefix_acceptance_remains_failed':True,'original_restore_failure':{k:a[k]for k in ('old_failure_receipt','old_failure_log','old_final_diagnostic')},'original_restore_acceptance_remains_failed':True,'training_reports':reports,'selected_step8_checkpoints':checkpoints,'training_protocol_sha256':a['original_protocol_sha256'],'paired8_proof':proof,'engineering_admission':{'scope':a['scope'],'new_deterministic_four_checks_prefix_unsaved7_passed':True,'new_deterministic_optimizer_updates':11,'old_common_prefix_gate_upgraded':False,'original_failed_restore_gate_upgraded':False,'formal2000_allowed':False}}

def verify_binding(path,digest,plan):
 require(sha(path)==digest,'Evaluation binding SHA differs');b=read(path);require(b['schema']=='h3-production67-singleWing-evaluation-binding-v4' and b['sample_id']==SOURCE_ID and b['split']=='dev' and b['formal16dev'] is False and b['checkpoint_selection'] is False and b['formal_training_allowed'] is False and b['test_unlocked'] is False,'Wrong Wing evaluation binding')
 tr=Path(b['training_reports']['control']['path']).parent.parent;rc=b['deterministic_restore_comparison']
 require(record(rc['path'])==rc and make_binding(b['data']['root'],tr,rc['path'],rc['sha256'],plan)==b,'Actual source/checkpoint/deterministic/old-failures changed');return b

def launch_admission(args):
 end=time.monotonic()+10
 while True:
  d=read(args.launch_receipt)
  if d.get('child_pid')==os.getpid():break
  require(d.get('status')=='starting_child' and time.monotonic()<end,'Missing owned launch handoff');time.sleep(.01)
 require(d['status']=='running' and d['stage']==args.stage and d['physical_npu']==args.physical_device and d['plan_sha256']==args.plan_sha256 and d['all_three_locks_held'] is True and d['device_idle']['no_running_processes'] is True,'No actual locked idle device admission')
 require(d['child_pgid']==os.getpgrp() and d['parent_pid']==os.getppid(),'Wrong owned PGID')

def precision(torch,device):
 require(device.type=='npu' and torch.npu.conv.allow_hf32 is False and torch.npu.matmul.allow_hf32 is False and torch.get_float32_matmul_precision()=='highest','FP32 HF32-off changed')
 return {'dtype':'float32','conv_hf32':False,'matmul_hf32':False,'float32_matmul_precision':'highest'}
def frozen_versions(m):return [(id(t),t._version) for t in list(m.parameters())+list(m.buffers())]
def frozen(m,device):require(not m.training and all(not p.requires_grad and p.grad is None and p.dtype==__import__('torch').float32 and p.device==device for p in m.parameters()),'Frozen model/precision/device differs')
def checked_pair(b):
 import h3_production67_pair_loader_v1 as L
 c=read(b['data']['contract']['path']);root=Path(b['data']['root']);pair=L.load_pair(root/'pair','pair.json',b['data']['pair']['sha256'],c);teacher=L.load_teacher(root/'teacher','teacher.json',b['data']['teacher']['sha256'],pair,c)
 return L,c,pair,teacher

def infer(args,plan,b,torch,device):
 import numpy as np
 import h3_production67_train_step_v1 as core
 L,c,pair,t=checked_pair(b);arrays=L.torch_arrays(pair);teacher=L.torch_teacher(t).unsqueeze(0).to(device);reference=arrays['reference_hr'].unsqueeze(0).to(device);draft=arrays['draft'].unsqueeze(0).to(device);flow=arrays['flow'].to(device);valid=arrays['flow_valid'].to(device)
 recipe=core.frozen_recipe(STAGE);folder=Path(OUTPUT)/'infer';folder.mkdir(exist_ok=False);receipts={}
 factory=module('wing_original_v5_factory',plan['files']['check_h3_production67_explicit_hf32off_npu_v1.py']['path'])
 student,loader,schema,source_paths,initial,checkpoint=factory.build(plan,torch)
 for label in MODELS:
  student.to('cpu');loaded={'kind':'original_v5_step1800','sha256':V5}
  if label!='v5':
   rc=b['selected_step8_checkpoints'][label];require(record(rc['path'])=={k:rc[k] for k in ('path','bytes','sha256')},'Selected checkpoint changed before trusted load')
   saved=torch.load(rc['path'],map_location='cpu',weights_only=False,mmap=True)
   require(saved['schema']=='h3-production67-paired8-engineering-v3' and saved['completed_updates']==8 and saved['arm']==('teacher' if label=='teacher_student' else 'control') and saved['protocol_sha256']==b['training_protocol_sha256'] and saved['formal_training_allowed'] is False,'Wrong checkpoint8 lineage')
   state=saved['model'];require(set(state)==set(schema) and all(v.dtype==torch.float32 and list(v.shape)==schema[k]['shape'] and bool(torch.isfinite(v).all()) for k,v in state.items()),'Checkpoint8 model schema/finite differs')
   student.load_state_dict(state,strict=True,assign=False);require(all(torch.equal(student.original_state_dict()[k],v) for k,v in state.items()),'Loaded step8 weights not bitwise exact');del state,saved;loaded={'kind':'fixed_engineering_step8_model_only_no_optimizer_restore','checkpoint':rc}
  student.to(device).eval().requires_grad_(False);frozen(student,device);before=frozen_versions(student);beg=time.monotonic()
  with torch.no_grad():pred=core.explicit_forward67(student,draft);terms=core.latent_terms_all(pred,reference,teacher,flow,valid,recipe)
  torch.npu.synchronize(device);require(before==frozen_versions(student),'Frozen evaluated model changed');frozen(student,device);precision(torch,device)
  array=pred[0].float().cpu().contiguous().numpy();fp=folder/(label+'.npy')
  with fp.open('xb') as f:np.save(f,array,allow_pickle=False)
  r={'status':'completed_fixed_prediction','sample_id':SOURCE_ID,'variant':label,'plan_sha256':args.plan_sha256,'binding_sha256':args.binding_sha256,'initial_v5_load':initial,'evaluated_weights':loaded,'input_pair_sha256':b['data']['pair']['sha256'],'teacher_record_sha256':b['data']['teacher']['sha256'],'latent':record(fp),'shape':list(array.shape),'dtype':'float32','latent_metrics':{k:float(v) for k,v in terms.items()},'raw_legacy_eWarp_preserved_separately':True,'all_67_tokens_and_66_valid_warp_pairs':True,'precision':precision(torch,device),'frozen_unchanged':True,'seconds':time.monotonic()-beg,'new_optimizer_updates':0}
  rp=folder/(label+'.json');write(rp,r);receipts[label]=record(rp);print(json.dumps({'stage':'inferred','variant':label}),flush=True)
  del pred,terms,array;student.to('cpu');torch.npu.empty_cache()
 write(folder/'report.json',{'status':'completed','stage':'infer','plan_sha256':args.plan_sha256,'binding_sha256':args.binding_sha256,'predictions':receipts,'actual_new_upscaler_forwards':3,'new_decoder_forwards':0,'optimizer_updates':0,'formal16dev_gate_passed':False})

def verify_infer(args):
 folder=Path(OUTPUT)/'infer';require(not (folder/'failure.json').exists(),'Inference had a postcheck failure');done=read(folder/'complete.json');require(done['status']=='completed_after_all_checks' and record(folder/'report.json')==done['report'] and done['plan_sha256']==args.plan_sha256 and done['binding_sha256']==args.binding_sha256,'No completed inference identity chain');r=read(folder/'report.json');require(r['status']=='completed' and r['plan_sha256']==args.plan_sha256 and r['binding_sha256']==args.binding_sha256 and set(r['predictions'])==set(MODELS),'Incomplete fixed inference')
 for label,rc in r['predictions'].items():
  require(record(rc['path'])==rc,'Prediction receipt changed');d=read(rc['path']);require(d['variant']==label and record(d['latent']['path'])==d['latent'] and d['shape']==[24,67,68,120] and d['dtype']=='float32' and d['frozen_unchanged'],'Wrong fixed prediction')
 return r

def pixels(args,plan,b,torch,device):
 import numpy as np
 import h3_production67_eval_metrics_v1 as metrics
 from data_utils import load_video_vae,decode_latent
 evidence=verify_infer(args);L,c,pair,t=checked_pair(b);arrays=L.torch_arrays(pair,include_native_hr_rgb=True);teacher_z=L.torch_teacher(t)
 folder=Path(OUTPUT)/'pixels';folder.mkdir(exist_ok=False)
 for r in plan['runtime_assets'].values():require(record(r['path'])==r,'Decoder/LPIPS dependency changed')
 vae=load_video_vae(plan['base_model'],device).eval().requires_grad_(False);require(sha(inspect.getfile(type(vae)))==plan['decoder_implementation_sha256'],'Wrong decoder implementation')
 import lpips
 require(sha(Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth')==plan['lpips_fingerprint']['backbone_sha256'] and sha(Path(lpips.__file__).resolve().parent/'weights/v0.1/alex.pth')==plan['lpips_fingerprint']['linear_sha256'],'Local LPIPS weights differ; network fallback forbidden')
 perceptual=lpips.LPIPS(net='alex',version='0.1').to(device).eval().requires_grad_(False);require(sha(inspect.getfile(type(perceptual)))==plan['lpips_fingerprint']['implementation_sha256'],'Wrong LPIPS implementation')
 frozen(vae,device);frozen(perceptual,device);before=[frozen_versions(m) for m in (vae,perceptual)];decoded={};targets={};decode_proofs={}
 def decode(z,label,origin):
  with torch.no_grad():raw=decode_latent(vae,z,device,keep_frames=None)
  require(tuple(raw.shape)==(226,1088,1920,3) and raw.dtype==torch.uint8,'Raw decoder output must be226x1088, no time trim')
  rgb=raw[:,4:1084].contiguous();require(tuple(rgb.shape)==(226,1080,1920,3),'Visible crop differs');decode_proofs[label]={'origin':origin,'raw_shape':list(raw.shape),'visible_shape':list(rgb.shape),'spatial_crop':[4,1084,0,1920],'time_crop':None,'visible_RGB_sha256':hashlib.sha256(memoryview(rgb.numpy()).cast('B')).hexdigest()};return rgb
 decoded['teacher']=decode(teacher_z,'teacher',b['data']['teacher']);decoded['decoded_hr']=decode(arrays['reference_hr'],'decoded_hr',b['data']['pair']);decoded['native_hr']=arrays['native_hr_rgb'];targets={t:{} for t in metrics.TARGETS}
 for label in MODELS:
  rec=read(evidence['predictions'][label]['path']);z=torch.from_numpy(np.load(rec['latent']['path'],allow_pickle=False).copy());rgb=decode(z,label,rec['latent'])
  for domain in metrics.TARGETS:targets[domain][label]=metrics.rgb_metrics(rgb,decoded[domain],decoded['decoded_hr'],perceptual,device)
  write(folder/(label+'_metrics.json'),{'variant':label,'sample_id':SOURCE_ID,'targets':{d:targets[d][label] for d in metrics.TARGETS},'decode':decode_proofs[label]});del rgb,z;torch.npu.empty_cache();print(json.dumps({'stage':'measured_full226','variant':label}),flush=True)
 for region in metrics.frame_regions226():require(len({targets[t][m]['regions'][region]['motion_mask_sha256'] for t in metrics.TARGETS for m in MODELS})==1,'Shared decodedHR mask changed by model/target')
 for group in metrics.transition_pairs226():require(len({targets[t][m]['transitions'][group]['motion_mask_sha256'] for t in metrics.TARGETS for m in MODELS})==1,'Transition mask changed by model/target')
 require(before==[frozen_versions(m) for m in (vae,perceptual)],'Frozen VAE/LPIPS changed');frozen(vae,device);frozen(perceptual,device);precision(torch,device)
 result={'status':'completed_singleWing_engineering_evaluation','sample_id':SOURCE_ID,'split':'dev','frames':226,'fps':[24,1],'duration_seconds':226/24,'plan_sha256':args.plan_sha256,'binding_sha256':args.binding_sha256,'targets':targets,'frame_regions':metrics.frame_regions226(),'transition_pair_groups':metrics.transition_pairs226(),'primary_target':'teacher','secondary_independent_target':'native_hr','retained_original_guard_target':'decoded_hr','ROI_definition':'Same detached decodedHR abs adjacent RGB channel-mean change>=.05 for all targets/models','metric_domain':'All226 assembled lossless decoder uint8/255 and separately verified native source uint8/255; LPIPS mapped[-1,1]. No lossy media metric.','latent_metrics':{m:read(evidence['predictions'][m]['path'])['latent_metrics'] for m in MODELS},'engineering_conditions':metrics.engineering_conditions(targets),'regressions':metrics.regressions(targets),'decodes':decode_proofs,'actual_decoder_forwards':5,'actual_optimizer_updates':0,'formal16dev_gate_passed':False,'test_unlocked':False,'checkpoint_reselection':False,'training_steps_compared':{'v5':1800,'control':8,'teacher_student':8},'engineering_admission':b['engineering_admission'],'original_restore_acceptance_remains_failed':True,'formal_training_allowed':False,'native_source_not_private_production':True,'source_adapter_sha256':ADAPTER,'native_RGB_sequence_sha256':RGBSEQ,'training_direct_supervision':{'nonblend_frames':166,'excluded_blend_frames':60},'limits':['One authorized Wing film-level dev window; no statistical/generalization or formal16dev claim.','NativeHR synthetic LR degradation area1280x704 and reflect4 compute1088 are public engineering choices, not private production parity.','Primary teacher is frozen bidirectional upscaler+decoder, not nativeHR or diffusionrefiner.','Decoder noncausal; HF32off choice does not erase historical defaultHF32 causality failure.','No video encoding/continuous-playback visual acceptance is performed by this metric-only entry.']}
 write(folder/'summary.json',result);write(folder/'report.json',{'status':'completed','stage':'pixels','plan_sha256':args.plan_sha256,'binding_sha256':args.binding_sha256,'summary':record(folder/'summary.json'),'actual_decoder_forwards':5,'optimizer_updates':0})

def run(args):
 plan=verify_plan(args.plan,args.plan_sha256);b=verify_binding(args.binding,args.binding_sha256,plan);p=str(args.physical_device)
 require(type(args.physical_device) is int and 0<=args.physical_device<=7 and os.environ.get('ASCEND_RT_VISIBLE_DEVICES')==p and os.environ.get('ASCEND_VISIBLE_DEVICES')==p and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0','Explicit physical routing differs');launch_admission(args)
 sys.path[:]=[STAGE]+[x for x in sys.path if x and x!=STAGE];guard=module('wing_eval_CANN_guard',plan['files']['h3_production67_cann_imports_v1.py']['path']);guard.staged_model_origins(plan,STAGE);guard.vendor_check(import_tbe=True)
 import torch
 torch.set_num_threads(4);torch.set_num_interop_threads(1);require(torch.__version__=='2.7.1+cpu','Torch version differs')
 import torch_npu
 require(torch_npu.__version__=='2.7.1.post2','NPU version differs');device=torch.device('npu:0');torch.npu.set_device(device);torch.npu.conv.allow_hf32=False;torch.npu.matmul.allow_hf32=False;torch.set_float32_matmul_precision('highest');precision(torch,device)
 try:
  {'infer':infer,'pixels':pixels}[args.stage](args,plan,b,torch,device)
  verify_plan(args.plan,args.plan_sha256);require(verify_binding(args.binding,args.binding_sha256,plan)==b,'Evaluation binding changed afterstage')
  for r in plan['runtime_assets'].values():require(record(r['path'])==r,'Runtime dependency changed afterstage')
  folder=Path(OUTPUT)/args.stage;write(folder/'complete.json',{'status':'completed_after_all_checks','stage':args.stage,'physical_npu':args.physical_device,'plan_sha256':args.plan_sha256,'binding_sha256':args.binding_sha256,'report':record(folder/'report.json'),'optimizer_updates':0})
 except BaseException as e:
  folder=Path(OUTPUT)/args.stage;folder.mkdir(exist_ok=True);write(folder/'failure.json',{'status':'failed_no_auto_retry','stage':args.stage,'type':type(e).__name__,'error':str(e),'optimizer_updates':0});raise

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=STAGES);p.add_argument('--physical-device',type=int,required=True)
 for key in ('plan','plan-sha256','binding','binding-sha256','launch-receipt'):p.add_argument('--'+key,required=True)
 run(p.parse_args())
