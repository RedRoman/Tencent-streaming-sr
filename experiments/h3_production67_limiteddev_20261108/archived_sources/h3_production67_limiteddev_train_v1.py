"""Explicit user-authorized train2/dev3 paired2000; no test or formal16dev claim."""
from __future__ import annotations
import argparse,hashlib,importlib.util,inspect,json,os,sys
from pathlib import Path
ROOT='/path/to/h3-project'
STAGE='/path/to/staging/h3_production67_limiteddev_20261107_v1'
OUTPUT=ROOT+'/upscaler_production67_limiteddev_20261107'
STAGES=('step0','control','teacher')
MODE={'deterministic_algorithms':True,'warn_only':False}
PRECISION={'dtype':'float32','conv_hf32':False,'matmul_hf32':False,'float32_matmul_precision':'highest'}
def require(v,msg):
 if not v:raise ValueError(msg)
def sha(p):
 p=Path(p);require(p.is_file() and not p.is_symlink(),'Regular exact file required');h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def identity(p):
 p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def module(n,p):
 require(n not in sys.modules,'Module already imported '+n);s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def verify_plan(path,digest):
 require(sha(path)==digest,'Plan changed');p=json.loads(Path(path).read_text())
 require(p['schema']=='h3-production67-limiteddev-execution-plan-v1' and p['stage_directory']==STAGE and p['output']==OUTPUT and p['stages']==list(STAGES),'Wrong limited experiment')
 require(p['precision']==PRECISION and p['mode']==MODE and p['physical_devices']==[4,7] and p['counts']=={'train':2,'dev':3,'test':0} and p['updates_per_arm']==2000 and p['test_access'] is False and p['formal16dev_claim'] is False,'Limited scope changed')
 for n,r in p['files'].items():require(r['path']==STAGE+'/'+n and identity(r['path'])==r,'Staged source changed '+n)
 for r in p['reused_evidence'].values():require(identity(r['path'])==r,'Existing actual evidence changed')
 require(p['files'][Path(__file__).name]['path']==str(Path(__file__).resolve()),'Exact staged worker required');return p

def predecessor(plan,stage,predecessor_sha):
 require(stage in STAGES,'Wrong stage')
 if stage=='step0':require(predecessor_sha is None,'First stage has no predecessor');return None
 path=Path(OUTPUT)/'step0'/'report.json';require(predecessor_sha is not None and sha(path)==predecessor_sha,'Actual shared step0 report SHA required')
 r=json.loads(path.read_text());outer=json.loads((Path(OUTPUT)/'launches/step0_attempt1.json').read_text())
 require(outer['status']=='completed' and outer['exit_code']==0 and outer['report']==identity(path),'Shared step0 parent not completed/reaped')
 require(r['status']=='completed_shared_limiteddev_step0' and r['optimizer_updates_this_attempt']==0 and r['plan_sha256']==sha(Path(STAGE)/'h3_production67_limiteddev_plan_v1.json'),'Wrong step0 scope')
 for name in ('payload_audit','common_step0','protocol','schedule','selection_plan'):require(identity(r[name]['path'])==r[name],'Shared artifact changed '+name)
 return r

def configure_mode(torch):
 torch.use_deterministic_algorithms(True,warn_only=False)
 got={'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'warn_only':torch.is_deterministic_algorithms_warn_only_enabled()};require(got==MODE,'Strict deterministic mode unavailable');return got

def run(args):
 plan=verify_plan(args.plan,args.plan_sha256);prior=predecessor(plan,args.stage,args.predecessor_report_sha256)
 require(args.physical_device in plan['physical_devices'],'Card not in explicit plan');d=str(args.physical_device)
 require(os.environ.get('ASCEND_RT_VISIBLE_DEVICES')==d and os.environ.get('ASCEND_VISIBLE_DEVICES')==d and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0','Device routing differs')
 parent=module('limited_frozen_owned_admission',plan['files']['check_h3_production67_paired8_npu_v1.py']['path']);parent.await_launch(args.launch_receipt,args.stage,args.physical_device,args.plan_sha256)
 out=Path(OUTPUT)/args.stage;out.mkdir(exist_ok=False);write(out/'started.json',{'stage':args.stage,'plan_sha256':args.plan_sha256,'scope':'limiteddev_train2_dev3_test0','optimizer_updates':0})
 try:
  sys.path[:]=[STAGE]+[p for p in sys.path if p and p!=STAGE]
  guard=module('limited_CANN_guard',plan['files']['h3_production67_cann_imports_v1.py']['path']);guard.staged_model_origins(plan,STAGE);vendor=guard.vendor_check(import_tbe=True)
  import h3_production67_limiteddev_data_v1 as data
  dataset=data.read_router(ROOT,plan['router'])
  if args.stage=='step0':
   audit=data.audit_payloads(dataset);write(out/'payload_audit.json',audit);audit_desc=identity(out/'payload_audit.json')
  else:audit_desc=prior['payload_audit'];data.validate_audit(dataset,json.loads(Path(audit_desc['path']).read_text()))
  for r in plan['runtime_evidence_files']:require(identity(r['path'])==r,'Installed runtime source changed')
  require(os.environ.get('ASCEND_HOME_PATH')==plan['selected_cann_home'] and os.environ.get('ASCEND_OPP_PATH')==plan['selected_cann_home']+'/opp','CANN selection changed')
  import torch
  require(torch.__version__=='2.7.1+cpu','Torch differs');torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.float32);mode=configure_mode(torch)
  import torch_npu
  require(torch_npu.__version__=='2.7.1.post2' and not torch.npu.is_initialized(),'Mode must precede NPU init');write(out/'mode_configured_before_NPU.json',{'mode':mode,'NPU_initialized':False,'stage':args.stage})
  import h3_production67_limiteddev_core_v1 as training
  import h3_production67_limiteddev_sparse_v1 as sparse
  from h3_production67_pixel_decoder_v1 import Production67PixelDecoder
  from data_utils import load_video_vae
  factory=module('limited_strict_initial_factory',plan['files']['check_h3_production67_explicit_hf32off_npu_v1.py']['path'])
  student,loader,schema,source_paths,initial,_=factory.build(plan,torch)
  device=torch.device('npu:0');torch.npu.set_device(device);torch.npu.conv.allow_hf32=False;torch.npu.matmul.allow_hf32=False;torch.set_float32_matmul_precision('highest')
  student.to(device).eval();loader.enable_checked_block_checkpointing(student,source_paths)
  for r in plan['runtime_assets'].values():require(identity(r['path'])==r,'Frozen decoder/LPIPS asset changed')
  vae=load_video_vae(plan['base_model'],device);require(sha(inspect.getfile(type(vae)))==plan['decoder_implementation_sha256'],'Decoder implementation changed')
  import lpips
  require(sha(Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth')==plan['lpips_fingerprint']['backbone_sha256'],'LPIPS local backbone changed');require(sha(Path(lpips.__file__).resolve().parent/'weights/v0.1/alex.pth')==plan['lpips_fingerprint']['linear_sha256'],'LPIPS local linear changed')
  perceptual=lpips.LPIPS(net='alex',version='0.1').to(device).eval().requires_grad_(False);require(sha(inspect.getfile(type(perceptual)))==plan['lpips_fingerprint']['implementation_sha256'],'LPIPS implementation changed')
  provider=Production67PixelDecoder(vae,device,temporal_scope='nonblend_only',checkpoint_decode=True,allow_full_spatial_grad=True)
  recipe=training.core.frozen_recipe(STAGE);schedule=training.build_paired_schedule([r['sample_id'] for r in data.rows_for(dataset,'train')],recipe);selection=training.build_selection_plan([r['sample_id'] for r in data.rows_for(dataset,'dev')])
  protocol=training.candidate_protocol(dataset,schedule,selection,evaluator_source_sha256=sha(inspect.getfile(sparse)),initial_tensor_schema_sha256=canonical(schema))
  if args.stage=='step0':
   write(out/'protocol.json',protocol);write(out/'schedule.json',schedule);write(out/'selection_plan.json',selection)
   baseline=sparse.make_common_step0(student=student,provider=provider,lpips_fn=perceptual,dataset=dataset,selection_plan=selection,tensor_manifest_path=plan['files']['h3_causal_v5_tensor_validation.json']['path'],payload_audit=audit_desc)
   write(out/'common_step0.json',baseline);result={'completed_updates':0};status='completed_shared_limiteddev_step0';count=0
  else:
   require(protocol==json.loads(Path(prior['protocol']['path']).read_text()) and schedule==json.loads(Path(prior['schedule']['path']).read_text()) and selection==json.loads(Path(prior['selection_plan']['path']).read_text()),'Materialized shared protocol/schedule differs')
   admission={'scope':'explicit_user_authorized_production67_limiteddev_2train_3dev','protocol_sha256':canonical(protocol),'physical_NPU_locks_held':True,'first_step_admission':'full_spatial_loss_backward_then_finite_nonzero_clip_then_optimizer','payload_audit':audit_desc,'common_step0_dev':prior['common_step0'],'launch_receipt_sha256':sha(args.launch_receipt),'limiteddev_user_request':'directly_start_training_current_accepted_train2_dev3','old_failed_gates_preserved':True,'formal_training_allowed':False,'test_access':False}
   result=training.run_arm(student=student,provider=provider,lpips_fn=perceptual,recipe=recipe,device=device,dataset=dataset,schedule=schedule,selection_plan=selection,evaluate_dev=sparse.evaluate_dev,output=out/'updates',arm=args.stage,protocol=protocol,admission=admission,initial_tensor_schema=schema)
   require(result['completed_updates']==2000 and result['optimizer_updates_this_attempt']==2000,'Incomplete arm');status='completed_limiteddev2000_awaiting_full226_guard';count=2000
  require(training.precision(device)==PRECISION and training.deterministic_mode()==MODE,'Mode/precision drift');verify_plan(args.plan,args.plan_sha256);data.validate_audit(dataset,json.loads(Path(audit_desc['path']).read_text()))
  for r in plan['runtime_assets'].values():require(identity(r['path'])==r,'Runtime asset changed')
  report={'status':status,'stage':args.stage,'plan_sha256':args.plan_sha256,'protocol_sha256':canonical(protocol),'physical_npu':args.physical_device,'precision':PRECISION,'actual_mode':mode,'scope':'limiteddev_train2_dev3_test0','result':result,'initial_load':initial,'CANN_vendor':vendor,'optimizer_updates_this_attempt':count,'formal16dev_claim':False,'full226_dev_quality_gate_passed':False,'test_access':False,'payload_audit':audit_desc}
  if args.stage=='step0':
   for key,name in (('common_step0','common_step0.json'),('protocol','protocol.json'),('schedule','schedule.json'),('selection_plan','selection_plan.json')):report[key]=identity(out/name)
  write(out/'report.json',report);print(json.dumps({'status':status,'report':identity(out/'report.json')}),flush=True)
 except BaseException as e:
  write(out/'failure.json',{'status':'failed_no_automatic_retry','stage':args.stage,'exception_type':type(e).__name__,'exception':str(e),'actual_updates':'Read updates/attempt.json; optimizer exception may be partial','formal16dev_claim':False,'test_access':False});raise

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=STAGES)
 for n in ('plan','plan-sha256','launch-receipt'):p.add_argument('--'+n,required=True)
 p.add_argument('--physical-device',type=int,required=True);p.add_argument('--predecessor-report-sha256');run(p.parse_args())
if __name__=='__main__':main()
