"""Explicit bounded real production67 paired8 worker. No formal2000 stage."""
from __future__ import annotations
import argparse,hashlib,importlib.util,inspect,json,os,sys,time
from pathlib import Path
ROOT='/path/to/h3-project'
STAGE='/path/to/staging/h3_production67_paired8_npu_20261106_v2'
OUTPUT=ROOT+'/upscaler_v5_motion_20261106/analysis/production67_paired8_engineering_v1'
STAGES=('control','teacher','restore_start','restore_finish')
LEAF_REPORT=ROOT+'/upscaler_v5_motion_20261106/analysis/production67_decoder_leaf_npu_v1/decoder_leaf/report.json'
V5='c7aaaa396d877a84c3fed2a5536189b7ed5e61226a82ef1a9e6be1e831f7f914'


def require(ok,msg):
 if not ok:raise ValueError(msg)


def sha(p):
 p=Path(p);require(p.is_file() and not p.is_symlink(),'Require regular exact source');h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()


def identity(p):
 p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}


def module(n,p):
 require(n not in sys.modules,'Already imported '+n);s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m


def verify_plan(path,digest):
 require(sha(path)==digest,'Source plan SHA differs');p=json.loads(Path(path).read_text())
 require(p['schema']=='h3-production67-paired8-NPU-plan-v1' and p['stage_directory']==STAGE and p['output']==OUTPUT and p['stages']==list(STAGES),'Wrong fixed new stage')
 require(p['precision']=={'dtype':'float32','conv_hf32':False,'matmul_hf32':False,'float32_matmul_precision':'highest'},'New HF32off choice differs')
 require(p['training']=={'steps':8,'seed':20261107,'save_steps':[3,6,8],'full_every':4,'lr':1e-5,'min_lr':1e-6,'warmup':100,'formal_training_allowed':False,'first_step_admission':'full_spatial_loss_backward_then_finite_nonzero_clip_then_optimizer'},'Fixed paired8 recipe changed')
 for n,r in p['files'].items():require(r['path']==STAGE+'/'+n and identity(r['path'])==r,'Staged source changed '+n)
 require(p['files'][Path(__file__).name]['path']==str(Path(__file__).resolve()),'Worker exact staged path required');return p


def binding(path,digest):
 require(sha(path)==digest,'Explicit actual-data binding SHA differs');b=json.loads(Path(path).read_text())
 require(b['schema']=='h3-production67-paired8-real-data-binding-v1' and b['formal_training_allowed'] is False and b['scope']=='single_source_engineering_preflight_not_formal_dataset','Wrong actual-data binding scope')
 require(Path(b['cache_root']).is_absolute() and Path(b['cache_root']).is_dir(),'Explicit data root missing')
 for k in ('contract','acceptance'):
  require(identity(b[k]['path'])==b[k],'Actual data gate input changed '+k)
 c=json.loads(Path(b['contract']['path']).read_text());r=json.loads(Path(b['acceptance']['path']).read_text())
 require(b['accepted_status'] in ('passed','completed') and r['status']==b['accepted_status'],'Actual data acceptance did not complete')
 require(r.get('stage')=='npu' and r.get('actual_encodes')==2 and r.get('actual_teacher_forwards')==1 and r.get('actual_decoder_forwards')==0 and r.get('optimizer_updates')==0 and r.get('training_ready') is False,'Not the completed first native engineering pair generation')
 require(len(b['samples'])==1 and b['samples'][0]['split']=='train','Exactly one Charge train engineering sample required')
 row=b['samples'][0];base=Path(b['cache_root']).resolve()
 for kind in ('pair','teacher'):
  path=(base/row[kind+'_root']/row[kind+'_record_path']).resolve()
  require(path.is_relative_to(base) and identity(path)==r[kind] and sha(path)==row[kind+'_record_sha256'],'Actual accepted '+kind+' record differs')
 # Strict pairloader independently checks actual nativeHR/flow/teacher payloads.
 # This explicit root binding is engineering admission, not full train/dev READY.
 return b,c


def predecessor(plan,leaf_sha):
 leaf=identity(LEAF_REPORT);require(leaf['sha256']==leaf_sha,'Root-accepted actual VAE leaf receipt SHA differs');d=json.loads(Path(LEAF_REPORT).read_text())
 require(d['status']=='passed' and d['actual_VAE_leaf_gradient_passed'] is True and d['optimizer_updates']==0 and d['precision']['conv_allow_hf32'] is False and d['precision']['matmul_allow_hf32'] is False,'Actual frozenVAE HF32off tile leaf not passed')
 for r in plan['precision_predecessors'].values():require(identity(r['path'])==r,'Precision evidence changed')
 return leaf


def await_launch(path,stage,device,plan_sha):
 end=time.monotonic()+10
 while True:
  d=json.loads(Path(path).read_text())
  if d.get('child_pid')==os.getpid():break
  require(d.get('status')=='starting_child' and time.monotonic()<end,'Own child handoff absent');time.sleep(.01)
 require(d['status']=='running' and d['stage']==stage and d['physical_npu']==device and d['plan_sha256']==plan_sha and d['all_three_locks_held'] is True and d['device_idle']['no_running_processes'] is True,'No current owned admission')
 require(d['child_pgid']==os.getpgrp() and d['parent_pid']==os.getppid(),'Wrong process ownership');return d


def run(args):
 plan=verify_plan(args.plan,args.plan_sha256);b,c=binding(args.data_binding,args.data_binding_sha256);pred=predecessor(plan,args.leaf_report_sha256)
 require(args.stage in STAGES and type(args.physical_device) is int and 0<=args.physical_device<=7,'Wrong physical device/stage');p=str(args.physical_device)
 require(os.environ.get('ASCEND_RT_VISIBLE_DEVICES')==p and os.environ.get('ASCEND_VISIBLE_DEVICES')==p and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0','Routing differs')
 launch=await_launch(args.launch_receipt,args.stage,args.physical_device,args.plan_sha256)
 out=Path(OUTPUT)/args.stage;out.mkdir(exist_ok=False)
 def write(n,v):
  with (out/n).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
 write('started.json',{'stage':args.stage,'plan_sha256':args.plan_sha256,'data_binding_sha256':args.data_binding_sha256,'leaf_report':pred,'optimizer_updates':0})
 try:
  sys.path[:]=[STAGE]+[p for p in sys.path if p and p!=STAGE]
  guard=module('paired8_CANN_guard',plan['files']['h3_production67_cann_imports_v1.py']['path']);guard.staged_model_origins(plan,STAGE);vendor=guard.vendor_check(import_tbe=True)
  import torch
  torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.float32);require(torch.__version__=='2.7.1+cpu','Torch version differs')
  import h3_production67_paired8_core_v3 as training
  from h3_production67_pixel_decoder_v1 import Production67PixelDecoder
  from data_utils import load_video_vae
  # Reuse actual-tested strict CPU v5 factory; unrelated diagnostic run() is never called.
  factory=module('paired8_strict_initial_factory',plan['files']['check_h3_production67_explicit_hf32off_npu_v1.py']['path'])
  student,loader,schema,source_paths,initial,checkpoint=factory.build(plan,torch)
  import torch_npu
  require(torch_npu.__version__=='2.7.1.post2','NPU version differs');device=torch.device('npu:0');torch.npu.set_device(device)
  torch.npu.conv.allow_hf32=False;torch.npu.matmul.allow_hf32=False;torch.set_float32_matmul_precision('highest')
  student.to(device);student.eval();loader.enable_checked_block_checkpointing(student,source_paths)
  # Verify native decoder and LPIPS artifacts before load; existing cache required,
  # no automatic network downloads are allowed in this entry.
  for r in plan['runtime_assets'].values():require(identity(r['path'])==r,'Decoder/LPIPS asset changed')
  vae=load_video_vae(plan['base_model'],device)
  require(sha(inspect.getfile(type(vae)))==plan['decoder_implementation_sha256'],'Decoder implementation changed')
  import lpips
  require(sha(Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth')==plan['lpips_fingerprint']['backbone_sha256'],'LPIPS local backbone missing/different')
  require(sha(Path(lpips.__file__).resolve().parent/'weights/v0.1/alex.pth')==plan['lpips_fingerprint']['linear_sha256'],'LPIPS local linear weight differs')
  perceptual=lpips.LPIPS(net='alex',version='0.1').to(device).eval().requires_grad_(False)
  require(sha(inspect.getfile(type(perceptual)))==plan['lpips_fingerprint']['implementation_sha256'],'LPIPS implementation differs')
  provider=Production67PixelDecoder(vae,device,temporal_scope='nonblend_only',checkpoint_decode=True,allow_full_spatial_grad=True)
  recipe=training.core.frozen_recipe(STAGE);schedule=training.build_paired_schedule([r['sample_id'] for r in b['samples']],recipe)
  protocol_sha=training.canonical_sha({'source_plan_sha256':args.plan_sha256,'data_binding_sha256':args.data_binding_sha256,'schedule':schedule,'initial_checkpoint_sha256':V5,'precision':plan['precision'],'configuration':plan['training']})
  resume=None;arm='teacher' if args.stage=='teacher' else 'control';updates_out=out/'updates'
  if args.stage=='restore_finish':
   updates_out=Path(OUTPUT)/'restore_start'/'updates';rp=updates_out/'checkpoint_step_003.receipt.json';resume=json.loads(rp.read_text())
   require(args.resume_checkpoint_sha256 and resume['sha256']==args.resume_checkpoint_sha256,'Root must explicitly bind actual resume checkpoint3')
  else:require(args.resume_checkpoint_sha256 is None,'Unexpected resume argument')
  admission={'scope':'real_production67_paired8_engineering','formal_training_allowed':False,'native_HR_pair_accepted':True,'physical_NPU_locks_held':True,'first_step_admission':plan['training']['first_step_admission'],'data_acceptance_sha256':b['acceptance']['sha256'],'launch_receipt_sha256':sha(args.launch_receipt)}
  result=training.run_arm(student=student,provider=provider,lpips_fn=perceptual,recipe=recipe,device=device,cache_root=b['cache_root'],data_contract=c,rows=b['samples'],schedule=schedule,output=updates_out,arm=arm,protocol_sha=protocol_sha,admission=admission,initial_tensor_schema=schema,resume_receipt=resume,stop_after=4 if args.stage=='restore_start' else None)
  require(result['status'] in ('completed_eight_engineering_updates_not_formal_training','controlled_stop_after_unsaved_step4'),'Actual bounded updates incomplete')
  verify_plan(args.plan,args.plan_sha256);require(binding(args.data_binding,args.data_binding_sha256)==(b,c),'Actual data binding changed')
  for r in plan['runtime_assets'].values():require(identity(r['path'])==r,'Runtime asset changed after steps')
  report={'status':'passed_engineering_stage','stage':args.stage,'plan_sha256':args.plan_sha256,'data_binding_sha256':args.data_binding_sha256,'physical_npu':args.physical_device,'protocol_sha256':protocol_sha,'result':result,'initial_load':initial,'CANN_vendor':vendor,'optimizer_updates_this_attempt':result['optimizer_updates_this_attempt'],'formal_training_allowed':False,'paired8_plus_restore_gate_passed':False,'limits':'Real engineering updates only. This single-source preflight does not establish full train/dev/test geometry or native-quality gates. Matched arms and controlled restoration must be separately accepted; original default-HF32 failure remains.'}
  write('report.json',report);print(json.dumps({'status':report['status'],'report':identity(out/'report.json')}),flush=True)
 except BaseException as e:
  write('failure.json',{'status':'failed_no_automatic_retry','stage':args.stage,'plan_sha256':args.plan_sha256,'exception_type':type(e).__name__,'exception':str(e),'formal_training_allowed':False,'actual_updates':'Read preserved updates/attempt*.json; never infer zero after optimizer began'});raise


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=STAGES)
 for name in ('plan','plan-sha256','data-binding','data-binding-sha256','leaf-report-sha256','launch-receipt'):p.add_argument('--'+name,required=True)
 p.add_argument('--physical-device',type=int,required=True);p.add_argument('--resume-checkpoint-sha256');run(p.parse_args())
if __name__=='__main__':main()
