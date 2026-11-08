"""Independent HF32-off numeric diagnostic after a preserved default-HF32 failure.
No encoder/decoder/optimizer/training/cache output. Source latents are frozen.
"""
from __future__ import annotations
import argparse,gc,hashlib,importlib.util,json,os,resource,sys,time,types
from pathlib import Path

ROOT='/path/to/h3-project'
SOURCE_RUN=ROOT+'/upscaler_v5_motion_20261106/diagnostics/surrogate67_704to1080_v3'
STAGE='/path/to/staging/h3_production67_explicit_hf32off_npu_20261106_v1'
OUTPUT=ROOT+'/upscaler_v5_motion_20261106/analysis/production67_explicit_hf32off_diagnostic_v1'
STAGES=('causal_hf32off',)
CONTRACT_SHA='41f3e018aa90760945659e42abf13adb91374452f884723fe3078096a4825879'
CPU_PLAN_SHA='9bc7054be55f2b3e78c2932484b54627ed90d4678259dd1a52fd8caa14f51a66'
CPU_REPORT='/path/to/staging/h3_production67_v5_full_cpu_20261106_v1/results_attempt1/report.json'
CPU_REPORT_SHA='df258b5529e89ccb704525dec919cf76b598db324925fb331a7e65b79c335800'
A_PLAN_SHA='5f11c7822deaa76917c6f8354aeeb3fc1fba7ee1c167697c83b31fa99c18652a'
A_REPORT=ROOT+'/upscaler_v5_motion_20261106/analysis/production67_explicit_parity_npu_v2/parity/report.json'
ATOL=3e-5
RTOL=3e-5
DEFAULT_FAILURE='/path/to/h3-project/upscaler_v5_motion_20261106/analysis/production67_explicit_checks_BC_v2/causal/failure.json'
DEFAULT_FAILURE_SHA='9202eed516025f4aea795b4c7d54978ceea2a88f9a8f5159f996c830daa96e28'
DEFAULT_PLAN_SHA='767d1e689ed1dab6bd360eb442a483bd2c009c09818993cbe002436c076f5525'
A_REPORT_SHA='f99005bb4656ed782c1783db9e5c3748d37d70224ad171f070932cc01f5f3a0b'
CUTS=(1,33,66)
CHUNKS=([1]*67,[7]*9+[4],[13,17,37])


def require(ok,message):
 if not ok:raise ValueError(message)


def sha(p):
 p=Path(p);require(p.is_file() and not p.is_symlink(),'Expected regular non-symlink: '+str(p));h=hashlib.sha256()
 with p.open('rb') as f:
  for v in iter(lambda:f.read(8<<20),b''):h.update(v)
 return h.hexdigest()


def identity(p):
 p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}


def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def module(name,path):
 require(name not in sys.modules,'Unexpected imported module: '+name)
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m


def verify_plan(path,digest):
 require(sha(path)==digest,'Protocol SHA differs');p=json.loads(Path(path).read_text())
 require(p['schema']=='h3_production67_explicit_hf32off_npu_plan_v1' and p['stage_directory']==STAGE and p['output']==OUTPUT,'Wrong new protocol')
 require(p['stages']==list(STAGES) and p['samples']==['VIDGEN_0001'],'Wrong fixed stages/samples')
 require(p['tolerances']=={'atol':ATOL,'rtol':RTOL},'Changed predeclared tolerances')
 require(p['causal_cuts']==list(CUTS) and p['stream_chunks']==[list(v) for v in CHUNKS],'Changed causal schedule')
 require(p['input_shape']==[1,24,67,44,80] and p['target_shape']==[1,24,67,68,120],'Wrong geometry')
 require(p['precision']['conv_hf32'] is False and p['precision']['matmul_hf32'] is False and p['precision']['storage_dtype']=='float32','HF32-off diagnostic precision differs')
 require(p['backward_allowed'] is False and p['formal_training_allowed'] is False and p['original_default_precision_gate_passed'] is False,'Wrong diagnostic-only scope')
 require(p['default_failure']=={'path':DEFAULT_FAILURE,'sha256':DEFAULT_FAILURE_SHA},'Original failure pin differs')
 require(p['optimizer_updates']==0 and p['new_cache_files']==0 and p['decoder_loaded'] is False,'Wrong scope')
 for name,r in p['files'].items():
  require(r['path']==STAGE+'/'+name and sha(r['path'])==r['sha256'] and Path(r['path']).stat().st_size==r['bytes'],'Source pin differs: '+name)
 require(p['files'][Path(__file__).name]['path']==str(Path(__file__).resolve()),'Execute only exact staged worker')
 return p


def input_manifest(plan):
 records={}
 for sid,rows in plan['inputs'].items():
  for role,r in rows.items():
   require(identity(r['path'])==r,'Pinned source input differs: '+sid+'/'+role);records[sid+'/'+role]=r
  inp=json.loads(Path(rows['input_receipt']['path']).read_text());pred=json.loads(Path(rows['v5_receipt']['path']).read_text())
  require(inp['status']==pred['status']=='completed' and inp['sample_id']==pred['sample_id']==sid,'Wrong source receipts')
  require(inp['contract_sha256']==pred['contract_sha256']==CONTRACT_SHA and inp['shape']==[24,67,44,80] and pred['shape']==[24,67,68,120],'Receipt geometry/source differs')
  require(inp['files']['latent.pt']=={k:rows['input_latent'][k] for k in ('sha256','bytes')} and pred['files']['latent.pt']=={k:rows['v5_latent'][k] for k in ('sha256','bytes')},'Payload receipts differ')
  require(pred['input_origin']==rows['input_receipt'],'Prediction uses different source input')
 return records


def predecessor(stage,plan_sha,cpu_sha,parity_sha):
 require(cpu_sha==CPU_REPORT_SHA and sha(CPU_REPORT)==cpu_sha,'Reviewed actual345M CPU SHA differs');r=json.loads(Path(CPU_REPORT).read_text())
 require(r['status']=='passed_actual345M_CPU_load_only' and r['execution_plan_sha256']==CPU_PLAN_SHA and r['actual_parameter_count']==345280216 and r['actual_tensor_count']==322,'Real345M CPU load not accepted')
 require(parity_sha==A_REPORT_SHA,'Wrong independently accepted A-v2 report')
 require(sha(A_REPORT)==parity_sha,'Root-accepted actual full67 A report SHA differs');a=json.loads(Path(A_REPORT).read_text())
 require(a['status']=='passed' and a['stage']=='parity' and a['plan_sha256']==A_PLAN_SHA and a['all_checks_passed'] is True and len(a['checks'])==4 and all(x['passed'] is True for x in a['checks'].values()),'Actual forward A not passed')
 require(stage=='causal_hf32off','Only independent numeric diagnostic stage allowed')
 require(sha(DEFAULT_FAILURE)==DEFAULT_FAILURE_SHA,'Original default-HF32 failure evidence changed')
 failed=json.loads(Path(DEFAULT_FAILURE).read_text())
 expected={*(f'future_{v}' for v in CUTS),*(f'prefix_{v}' for v in CUTS),*(f'stream_{v}' for v in range(3))}
 require(failed['status']=='failed_no_retry' and failed['stage']=='causal' and failed['plan_sha256']==DEFAULT_PLAN_SHA and failed['optimizer_updates']==0 and failed['formal_training_allowed'] is False,'Unexpected original failure provenance')
 require(set(failed['checks_completed'])==expected and not all(c['passed'] is True for c in failed['checks_completed'].values()),'Original nine-check failure not preserved')
 return {'cpu':identity(CPU_REPORT),'parity_A':identity(A_REPORT),'original_default_HF32_failure':identity(DEFAULT_FAILURE)}


def load_latent(torch,row,sid,variant,shape):
 require(identity(row['path'])==row,'Source payload changed before load')
 x=torch.load(row['path'],map_location='cpu',weights_only=True)
 require(isinstance(x,dict) and set(x)=={'latent','sample_id','variant','contract_sha256'} and x['sample_id']==sid and x['variant']==variant and x['contract_sha256']==CONTRACT_SHA,'Wrong frozen payload schema')
 z=x['latent'];require(isinstance(z,torch.Tensor) and tuple(z.shape)==tuple(shape) and z.device.type=='cpu' and z.dtype==torch.float32 and bool(torch.isfinite(z).all()),'Wrong frozen CPU latent tensor')
 require(identity(row['path'])==row,'Source payload changed after load');return z[None]


def compare(torch,a,b):
 a=a.detach().cpu();b=b.detach().cpu()
 require(a.shape==b.shape and a.dtype==b.dtype==torch.float32 and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),'Invalid comparison tensors')
 d=(a-b).double();den=float(b.double().square().sum());num=float(d.square().sum())
 return {'passed':bool(torch.allclose(a,b,atol=ATOL,rtol=RTOL)),'max_abs':float(d.abs().max()),'mean_abs':float(d.abs().mean()),'relative_l2':(num/max(den,1e-30))**.5,'reference_l2':den**.5,'atol':ATOL,'rtol':RTOL}


def precision(torch,torch_npu):
 return {'torch':str(torch.__version__),'torch_npu':str(torch_npu.__version__),'storage':'float32','conv_allow_hf32':bool(torch.npu.conv.allow_hf32),'matmul_allow_hf32':bool(torch.npu.matmul.allow_hf32),'float32_matmul_precision':torch.get_float32_matmul_precision(),'logical_device':'npu:0'}


def memory(torch,device):
 torch.npu.synchronize(device)
 return {'npu_allocated_bytes':int(torch.npu.memory_allocated(device)),'npu_reserved_bytes':int(torch.npu.memory_reserved(device)),'npu_peak_allocated_bytes':int(torch.npu.max_memory_allocated(device)),'npu_peak_reserved_bytes':int(torch.npu.max_memory_reserved(device)),'CPU_peak_RSS_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)}


def cpu_parameter_hashes(model,schema,torch):
 require(set(model.state_dict())==set(schema),'Parameter schema changed');found={}
 for n,t in model.state_dict().items():
  c=t.detach().cpu().contiguous();require(c.dtype==torch.float32 and list(c.shape)==schema[n]['shape'] and bool(torch.isfinite(c).all()),'Parameter changed shape/dtype/finite')
  digest=hashlib.sha256(c.numpy().tobytes(order='C')).hexdigest();require(digest==schema[n]['tensor_sha256'],'Parameter value changed: '+n);found[n]=digest
 return found


def build(plan,torch):
 smoke=module('smoke_h3_production67_v5_full_cpu_v1',plan['files']['smoke_h3_production67_v5_full_cpu_v1.py']['path'])
 smoke.memory_guard(smoke.memory_snapshot())
 c=smoke.verify_contract(json.loads(Path(plan['files']['surrogate67_contract.json']['path']).read_text()));checkpoint=smoke.selected_identity(c);smoke.validate_checkpoint_archive(smoke.REAL_CHECKPOINT)
 loader=module('h3_production67_v5_loader_v1',plan['files']['h3_production67_v5_loader_v1.py']['path'])
 schema=loader.load_tensor_manifest(plan['files']['h3_causal_v5_tensor_validation.json']['path']);up=module('h3_npu_probe_original_upstream',plan['files']['upscaler.py']['path']);node=up.load(smoke.REAL_CHECKPOINT)
 require(node.__name__ not in sys.modules,'Unexpected node module');sys.modules[node.__name__]=node
 cfg=node._detect_arch({n:types.SimpleNamespace(shape=tuple(r['shape'])) for n,r in schema.items()});require(cfg==smoke.FACTORY_CFG,'Original architecture differs')
 with torch.device('cpu'):model=node.LatentResizer3D(**cfg)
 ids={n:id(v) for n,v in model.named_parameters()};paths={n:plan['files'][n]['path'] for n in loader.FROZEN_SOURCES}
 student,receipt=loader.load_initial_v5_cpu(smoke.REAL_CHECKPOINT,node=node,original_model=model,tensor_manifest_path=plan['files']['h3_causal_v5_tensor_validation.json']['path'],source_paths=paths,activation_checkpointing=False)
 require(ids=={n:id(v) for n,v in model.named_parameters()},'CPU load parameter identity changed')
 return student,loader,schema,paths,receipt,checkpoint


def await_owned_launch(path,stage,device,plan_sha,timeout=10):
 # The parent publishes ownership immediately after Popen. Wait only for that
 # CPU bookkeeping handoff, before importing Torch or constructing a model.
 deadline=time.monotonic()+timeout
 while True:
  d=json.loads(Path(path).read_text())
  if d.get('child_pid')==os.getpid():break
  require(d.get('status')=='starting_child' and time.monotonic()<deadline,'Unique parent/child launch handoff not published')
  time.sleep(.01)
 require(d['status']=='running' and d['stage']==stage and d['physical_npu']==device and d['plan_sha256']==plan_sha and d['all_three_locks_held'] is True and d['device_idle']['no_running_processes'] is True,'Launcher device/lock/idle proof missing')
 require(d['child_pid']==os.getpid() and d['child_pgid']==os.getpgrp() and d['parent_pid']==os.getppid(),'Not the unique owned stage child')
 return d


def run(args):
 plan=verify_plan(args.plan,args.plan_sha256);require(args.stage in STAGES and type(args.physical_device) is int and 0<=args.physical_device<=7,'Invalid stage/device')
 p=str(args.physical_device);require(os.environ.get('ASCEND_RT_VISIBLE_DEVICES')==p and os.environ.get('ASCEND_VISIBLE_DEVICES')==p and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0','Explicit physical routing/autoload environment differs')
 launch=Path(args.launch_receipt);d=await_owned_launch(launch,args.stage,args.physical_device,args.plan_sha256)
 parents=predecessor(args.stage,args.plan_sha256,args.cpu_report_sha256,args.parity_report_sha256);before_inputs=input_manifest(plan);before_sources={n:identity(r['path']) for n,r in plan['files'].items()};out=Path(OUTPUT)/args.stage;out.mkdir(exist_ok=False);write(out/'started.json',{'stage':args.stage,'physical_device':args.physical_device,'plan_sha256':args.plan_sha256,'predecessors':parents,'source_inputs':before_inputs,'sources':before_sources,'no_training':True})
 results={};started=time.monotonic();mem=[]
 try:
  sys.path[:]=[STAGE]+[q for q in sys.path if q and q!=STAGE]
  import_guard=module('h3_worker_cann_import_guard',plan['files']['h3_production67_cann_imports_v1.py']['path']);import_guard.staged_model_origins(plan,STAGE);vendor_before=import_guard.vendor_check(import_tbe=True)
  import torch
  torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.float32)
  require(str(torch.__version__)=='2.7.1+cpu','Unexpected Torch version')
  student,loader,schema,source_paths,load_receipt,checkpoint_before=build(plan,torch)
  import torch_npu
  require(str(torch_npu.__version__)=='2.7.1.post2','Unexpected NPU version');device=torch.device('npu:0');torch.npu.set_device(device)
  torch.npu.conv.allow_hf32=False;torch.npu.matmul.allow_hf32=False;torch.set_float32_matmul_precision('highest')
  prec=precision(torch,torch_npu);require(prec['conv_allow_hf32'] is False and prec['matmul_allow_hf32'] is False,'Actual HF32-off setting not applied');cpu_ids={n:id(v) for n,v in student.model.named_parameters()};student.to(device);ids={n:id(v) for n,v in student.model.named_parameters()}
  require(set(cpu_ids)==set(ids),'Device conversion changed parameter names')
  # A framework CPU->NPU conversion may replace Parameter objects. Record that
  # boundary honestly, then require unchanged identity throughout all forwards,
  # backward/checkpoint installation. Every value is checked before and after.
  conversion_identity_retained=cpu_ids==ids
  cpu_parameter_hashes(student.model,schema,torch)
  require(all(v.device==device and v.dtype==torch.float32 for v in student.parameters()),'Parameters not exactNPU FP32')
  student.eval();torch.npu.reset_peak_memory_stats(device)
  if args.stage=='causal_hf32off':
   x=load_latent(torch,plan['inputs']['VIDGEN_0001']['input_latent'],'VIDGEN_0001','input704',(24,67,44,80)).to(device)
   with torch.no_grad():
    full=student.forward_full(x,target_size=(67,68,120))
    for cut in CUTS:
     changed=x.clone();changed[:,:,cut:]=changed[:,:,cut:]*-2.3+1.7;y=student.forward_full(changed,target_size=(67,68,120));results['future_'+str(cut)]=compare(torch,full[:,:,:cut],y[:,:,:cut]);del changed,y
     y=student.forward_prefix(x[:,:,:cut],target_size=(cut,68,120));results['prefix_'+str(cut)]=compare(torch,full[:,:,:cut],y);del y
    for n,chunks in enumerate(CHUNKS):
     student.reset_stream();state=None;values=[];index=0
     for count in chunks:
      y,state=student.forward_chunk(x[:,:,index:index+count],state,latent_index=index,target_hw=(68,120));values.append(y);index+=count
     require(index==67 and state.next_latent_index==67,'Streaming schedule changed');results['stream_'+str(n)]=compare(torch,full,torch.cat(values,dim=2));del values,y
    student.reset_stream();mem.append(memory(torch,device));del x,full
  require(len(results)==9 and all(r.get('passed') is True for r in results.values()),'HF32-off independent numerical consistency diagnostic failed; default gate remains failed')
  require(predecessor(args.stage,args.plan_sha256,args.cpu_report_sha256,args.parity_report_sha256)==parents,'Predecessor evidence changed')
  require(precision(torch,torch_npu)==prec,'Precision changed');require(ids=={n:id(v) for n,v in student.model.named_parameters()},'Parameter identity changed');hashes=cpu_parameter_hashes(student.model,schema,torch)
  require(input_manifest(plan)==before_inputs,'Frozen inputs changed');after_sources={n:identity(r['path']) for n,r in plan['files'].items()};require(after_sources==before_sources,'Sources changed');smoke=sys.modules['smoke_h3_production67_v5_full_cpu_v1'];checkpoint_after=smoke.selected_identity(json.loads(Path(plan['files']['surrogate67_contract.json']['path']).read_text()));require(checkpoint_before==checkpoint_after,'Checkpoint changed')
  report={'status':'passed','stage':args.stage,'all_checks_passed':True,'plan_sha256':args.plan_sha256,'predecessors':parents,'physical_npu':args.physical_device,'logical_device':'npu:0','seconds':time.monotonic()-started,'precision':prec,'checks':results,'memory':mem,'cpu_load_receipt':load_receipt,'CANN_vendor_dependency':vendor_before,'CPU_to_NPU_parameter_identity_retained':conversion_identity_retained,'NPU_forward_parameter_identity_retained':True,'source_before':before_sources,'source_after':after_sources,'input_before':before_inputs,'input_after':before_inputs,'checkpoint_before':checkpoint_before,'checkpoint_after':checkpoint_after,'all322_parameter_values_unchanged':True,'parameter_tensor_hashes':hashes,'optimizer_updates':0,'new_latent_or_RGB_cache_files':0,'decoder_loaded':False,'formal_training_allowed':False,'original_default_precision_gate_passed':False,'diagnostic_only':True,'limits':'Only VIDGEN_0001 at exact67x44x80-to67x68x120 with convHF32false/matmulHF32false. This never changes or passes the original default-HF32 gate and does not prove arbitrary sizes, training, or decoder causality.'}
  write(out/'report.json',report);print(json.dumps({'status':'passed','stage':args.stage,'report':identity(out/'report.json')}),flush=True)
 except BaseException as e:
  if 'torch' in locals() and 'device' in locals():
   try:mem.append({'at_failure':memory(torch,device)})
   except BaseException as measurement_error:mem.append({'at_failure_memory_unavailable':str(measurement_error)})
  write(out/'failure.json',{'status':'failed_no_retry','stage':args.stage,'plan_sha256':args.plan_sha256,'exception_type':type(e).__name__,'exception':str(e),'checks_completed':results,'predecessors':parents,'original_default_precision_gate_passed':False,'diagnostic_only':True,'memory':mem,'optimizer_updates':0,'formal_training_allowed':False});raise


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=STAGES);p.add_argument('--physical-device',required=True,type=int);p.add_argument('--plan',required=True,type=Path);p.add_argument('--plan-sha256',required=True);p.add_argument('--cpu-report-sha256',required=True);p.add_argument('--parity-report-sha256',required=True);p.add_argument('--launch-receipt',required=True);run(p.parse_args())
if __name__=='__main__':main()
