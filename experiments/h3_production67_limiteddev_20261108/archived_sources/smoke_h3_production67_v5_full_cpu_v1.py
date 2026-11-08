"""Prepared CPU-only 345M strict v5 load smoke. No forward/decoder/optimizer.

Run only after independent review. Exact original checkpoint and source pins are
mandatory; this never searches for production data or falls back to an export.
"""
from __future__ import annotations
import argparse, datetime, hashlib, importlib.abc, importlib.util, json, os
from pathlib import Path
import pickletools, resource, stat, sys, time, types, zipfile

CONTRACT_SHA='41f3e018aa90760945659e42abf13adb91374452f884723fe3078096a4825879'
V5_SHA='c7aaaa396d877a84c3fed2a5536189b7ed5e61226a82ef1a9e6be1e831f7f914'
MANIFEST_SHA='071b3fd8b71dfde906b1f75f250d79c8d8df32f24b0c0c194b83387bf1705a75'
LOADER_SHA='03d043154f9d920a3c63ebeb39d713a825ea32dd7f9592b4264ced8e935a4512'
UPSTREAM_LOADER_SHA='e54959b609c5732711fd8e87b5e6cbe4e80977379d988df175afa1c7745f4521'
UPSTREAM_NODE_SHA='f5945b500ddb38afee23c377034bf15394ba64748511a7e9608953040c4be437'
PARAMETERS=345280216
TENSORS=322
PARAMETER_BYTES=PARAMETERS*4
GIB=1<<30
AS_CAP=12*GIB
MIN_HEADROOM=6*GIB
REMOTE_ROOT='/path/to/h3-project'
STAGE='/path/to/staging/h3_production67_v5_full_cpu_20261106_v1'
REAL_CHECKPOINT=REMOTE_ROOT+'/upscaler_conv3d_hybrid/kd_pixel_steps2000/best_step_001800.pt'
ALIAS_CHECKPOINT=REMOTE_ROOT+'/upscaler_conv3d_hybrid/kd_pixel_steps2000/best.pt'
ENV={'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','ASCEND_RT_VISIBLE_DEVICES':'','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONPATH':'','PYTHONNOUSERSITE':'1'}
FACTORY_CFG={'in_channels':24,'in_blocks':12,'out_blocks':12,'channels':512,'dropout':0.1,'attn':False,'temporal_every':2,'temporal_kernel':5}
FROZEN={
 'h3_production67_v5_loader_v1.py':LOADER_SHA,
 'h3_conv3d_explicit_target_v1.py':'8c5bcbd1702a252266c68cbdf131e40fa7b56366deaeef17c53a9b84dbd5b3c6',
 'h3_production67_geometry_v1.py':'147765ef79549bc31fd2b9e5026d5983be2328ee12ee04743f9202543b4af383',
 'causal_upscaler.py':'8032ebd29e558d2cad7f31cb91618022892270a5d4027052f251475b04f98677',
 'train_upscaler_trial.py':'ae0f5d0b185600ebc1b674977b1b338607aee5aab801a4ad4cd1e8e1b6710a71',
 'data_utils.py':'25678af75463515f13a86b2a71aae8066cb661417e2056a3500180d370a2c488',
 'prepare_data.py':'2b07f8e431b884d0cb1063033f5a965192f1d6a26bf40c5d47c156198503b9b8',
 'upscaler.py':UPSTREAM_LOADER_SHA,'upscaler_node.py':UPSTREAM_NODE_SHA,
 'h3_causal_v5_tensor_validation.json':MANIFEST_SHA,
 'surrogate67_contract.json':CONTRACT_SHA,
}
FORBIDDEN_PICKLE_KEYS={'optimizer','optimizer_state_dict','optim_state_dict','torch_rng','device_rng','window_rng','rng_state','cpu_rng_state','cuda_rng_state','npu_rng_state'}


def require(ok,msg):
 if not ok:raise ValueError(msg)


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
 p=Path(path);require(p.is_file() and not p.is_symlink(),'Expected regular, nonsymlink file: '+str(p))
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()


def identity(path):
 p=Path(path);s=p.stat();return {'path':str(p),'bytes':s.st_size,'sha256':sha(p),'inode':s.st_ino,'device':s.st_dev,'mtime_ns':s.st_mtime_ns}


def record(path,value):
 with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def load_plan(path,expected_sha):
 require(sha(path)==expected_sha,'Execution plan SHA mismatch')
 d=json.loads(Path(path).read_text())
 require(d['schema']=='h3_production67_v5_full_cpu_smoke_plan_v1','Wrong plan schema')
 require(d['stage']==STAGE and d['output_directory']==STAGE+'/results_attempt1','Wrong isolated stage/output')
 require(d['environment']==ENV and d['address_space_cap_bytes']==AS_CAP and d['minimum_headroom_bytes']==MIN_HEADROOM,'Wrong CPU/memory plan')
 require(d['forward_allowed'] is False and d['optimizer_allowed'] is False and d['decoder_allowed'] is False and d['NPU_allowed'] is False,'Wrong execution scope')
 expected={**FROZEN,Path(__file__).name:d['script_sha256']}
 require(set(d['files'])==set(expected),'Missing/extra execution files')
 for name,digest in expected.items():
  x=d['files'][name];require(x['path']==STAGE+'/'+name and x['sha256']==digest and type(x['bytes']) is int and x['bytes']>0,'Wrong file mapping: '+name)
 return d


def verify_stage(plan):
 require(Path(__file__).resolve()==Path(STAGE)/Path(__file__).name,'Execute only the new isolated staged source')
 require(sha(__file__)==plan['script_sha256'],'Executing script SHA differs')
 actual={}
 for name,row in plan['files'].items():
  x=identity(row['path']);require(x['sha256']==row['sha256'] and x['bytes']==row['bytes'],'Staged source mismatch: '+name);actual[name]=x
 return actual


def verify_contract(contract):
 c=contract
 require(c['selected_checkpoints']['v5']=={'path':ALIAS_CHECKPOINT,'sha256':V5_SHA,'bytes':1381232411,'selected_step':1800},'Wrong selected v5 lineage')
 require(c['files']['v5_checkpoint']=={'path':REAL_CHECKPOINT,'sha256':V5_SHA,'bytes':1381232411},'Wrong canonical v5 file')
 require(c['expected_input_latent']==[24,67,44,80] and c['target_latent']==[24,67,68,120] and c['frames']==226 and c['fps']==24,'Wrong geometry contract')
 require(c['files']['upscaler_loader']['path']==REMOTE_ROOT+'/upstream/open_h3/upscaler.py' and c['files']['upscaler_loader']['sha256']==UPSTREAM_LOADER_SHA,'Wrong upstream loader')
 require(c['files']['upscaler_node']['path']==REMOTE_ROOT+'/upstream/open_h3/upscaler_node.py' and c['files']['upscaler_node']['sha256']==UPSTREAM_NODE_SHA,'Wrong upstream node')
 return c


def selected_identity(contract):
 alias=Path(contract['selected_checkpoints']['v5']['path']);actual=Path(contract['files']['v5_checkpoint']['path'])
 require(alias.resolve(strict=True)==actual and actual.is_file() and not actual.is_symlink(),'Selected alias must resolve to the exact frozen canonical checkpoint')
 a=alias.lstat();item=identity(actual);require(item['sha256']==V5_SHA and item['bytes']==1381232411,'Full original v5 bytes mismatch')
 return {'alias':str(alias),'alias_is_symlink':alias.is_symlink(),'alias_link_text':os.readlink(alias) if alias.is_symlink() else None,'alias_inode':a.st_ino,'alias_mtime_ns':a.st_mtime_ns,'canonical':item}


def validate_checkpoint_archive(path,expected_tensors=TENSORS,expected_bytes=PARAMETER_BYTES):
 """Read ZIP headers/pickle opcode strings, never unpickle here."""
 with zipfile.ZipFile(path) as z:
  infos=z.infolist();names=[x.filename for x in infos]
  require(len(names)==len(set(names)),'Duplicate checkpoint ZIP entries')
  roots={n.split('/')[0] for n in names};require(len(roots)==1,'Multiple checkpoint ZIP roots')
  root=next(iter(roots));data_prefix=root+'/data/'
  data=[x for x in infos if x.filename.startswith(data_prefix)]
  require(len(data)==expected_tensors and all(x.filename[len(data_prefix):].isdigit() and x.compress_type==zipfile.ZIP_STORED for x in data),'Expected exactly322 uncompressed model tensor storages')
  require(sum(x.file_size for x in data)==expected_bytes,'Additional/missing tensor storage bytes, including possible optimizer/RNG')
  allowed={root+'/data.pkl',root+'/byteorder',root+'/version',root+'/.data/serialization_id',root+'/.format_version',root+'/.storage_alignment'}|{x.filename for x in data}
  require(set(names)<=allowed and root+'/data.pkl' in names,'Unexpected checkpoint archive members')
  require(sum(x.file_size for x in infos if not x.filename.startswith(data_prefix)) < 2<<20,'Excess non-tensor checkpoint data')
  p=z.getinfo(root+'/data.pkl');require(p.file_size<2<<20,'Unexpectedly large checkpoint pickle metadata')
  raw=z.read(p);strings=set()
  for op,arg,_ in pickletools.genops(raw):
   if op.name in ('UNICODE','BINUNICODE','SHORT_BINUNICODE','BINUNICODE8','STRING','BINSTRING','SHORT_BINSTRING') and isinstance(arg,str):strings.add(arg)
  forbidden=strings & FORBIDDEN_PICKLE_KEYS;require(not forbidden,'Optimizer/RNG pickle key found: '+repr(sorted(forbidden)))
  require({'model_state_dict','model_config','adapter_config','state_dict_format','step','contract'}<=strings,'Expected original checkpoint metadata missing')
  return {'storage_count':len(data),'storage_bytes':sum(x.file_size for x in data),'expected_model_parameter_bytes':expected_bytes,'pickle_bytes':len(raw),'pickle_sha256':hashlib.sha256(raw).hexdigest(),'forbidden_optimizer_rng_keys_found':[],'inspection':'ZIP/pickle opcodes only, no unpickling by this check'}


def parse_meminfo(text):
 values={}
 for line in text.splitlines():
  parts=line.split()
  if len(parts)==3 and parts[0].endswith(':') and parts[2]=='kB':values[parts[0][:-1]]=int(parts[1])*1024
 require('MemAvailable' in values,'No Linux MemAvailable evidence');return values


def memory_snapshot():
 info=parse_meminfo(Path('/proc/meminfo').read_text());status={}
 for line in Path('/proc/self/status').read_text().splitlines():
  if line.startswith(('VmPeak:','VmSize:','VmHWM:','VmRSS:','Threads:')):status[line.split(':')[0]]=line.split(':',1)[1].strip()
 cgroup={}
 for root,limit,current in [('/sys/fs/cgroup','memory.max','memory.current'),('/sys/fs/cgroup/memory','memory.limit_in_bytes','memory.usage_in_bytes')]:
  r=Path(root)
  if (r/limit).is_file() and (r/current).is_file():
   lim=(r/limit).read_text().strip();use=int((r/current).read_text());finite=None if lim=='max' or int(lim)>=(1<<60) else int(lim)
   cgroup={'limit_bytes':finite,'current_bytes':use,'headroom_bytes':None if finite is None else finite-use,'source_directory':root};break
 return {'mem_available_bytes':info['MemAvailable'],'proc_self':status,'ru_maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'ru_maxrss_units':'Linux KiB converted to bytes','cgroup':cgroup,'rlimit_as':list(resource.getrlimit(resource.RLIMIT_AS))}


def memory_guard(snapshot):
 require(snapshot['mem_available_bytes']>=MIN_HEADROOM,'Less than6GiB host CPU memory headroom')
 h=snapshot.get('cgroup',{}).get('headroom_bytes');require(h is None or h>=MIN_HEADROOM,'Less than6GiB cgroup CPU memory headroom')


class DenyNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch_npu' or fullname.startswith('torch_npu.'):
   raise ImportError('torch_npu is forbidden in this CPU-only smoke')
  return None


def import_file(name,path):
 require(name not in sys.modules,'Unexpected preimported module: '+name)
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def run(plan,plan_path,plan_sha):
 require(sys.platform.startswith('linux'),'This exact memory-audited execution plan requires Linux')
 require(all(os.environ.get(k)==v for k,v in ENV.items()),'CPU-only environment mismatch')
 require(not any(k=='torch' or k.startswith('torch.') or k=='torch_npu' or k.startswith('torch_npu.') for k in sys.modules),'Run in a fresh Python process')
 before_sources=verify_stage(plan);c=verify_contract(json.loads(Path(plan['files']['surrogate67_contract.json']['path']).read_text()))
 upstream_before={k:identity(c['files'][k]['path']) for k in ('upscaler_loader','upscaler_node')}
 for k,x in upstream_before.items():require(x['sha256']==c['files'][k]['sha256'] and x['bytes']==c['files'][k]['bytes'],'Original upstream identity changed')
 selected_before=selected_identity(c)
 archive=validate_checkpoint_archive(REAL_CHECKPOINT)
 out=Path(plan['output_directory']);out.mkdir(exist_ok=False)
 started=utc();record(out/'started.json',{'utc':started,'pid':os.getpid(),'argv':sys.argv,'environment':ENV,'plan':identity(plan_path),'scope':'CPU strict345M load only, no forward/optimizer/decoder','sources':before_sources,'selected_checkpoint':selected_before,'checkpoint_archive':archive})
 stages=[];models=[]
 try:
  mem0=memory_snapshot();memory_guard(mem0);stages.append({'stage':'before_import','memory':mem0})
  soft,hard=resource.getrlimit(resource.RLIMIT_AS)
  cap=min(AS_CAP,hard) if hard!=resource.RLIM_INFINITY else AS_CAP
  require(cap>=MIN_HEADROOM,'Preexisting address-space hard limit is too small')
  if soft!=resource.RLIM_INFINITY:cap=min(cap,soft)
  require(cap>=MIN_HEADROOM,'Preexisting address-space soft limit is too small')
  resource.setrlimit(resource.RLIMIT_AS,(cap,hard))
  sys.meta_path.insert(0,DenyNPU())
  sys.path[:]=[STAGE]+[p for p in sys.path if p and p!=STAGE]
  import torch
  require(torch.__version__=='2.7.1+cpu' and torch.version.cuda is None,'Require the previously audited torch2.7.1+cpu interpreter')
  require(not any(k=='torch_npu' or k.startswith('torch_npu.') for k in sys.modules),'Unexpected torch_npu import')
  torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.float32)
  loader=import_file('h3_production67_v5_loader_v1',plan['files']['h3_production67_v5_loader_v1.py']['path'])
  schema=loader.load_tensor_manifest(plan['files']['h3_causal_v5_tensor_validation.json']['path'])
  up=import_file('h3_full_cpu_upstream_loader_v1',plan['files']['upscaler.py']['path'])
  node=up.load(REAL_CHECKPOINT) # Only imports node; does not call load_model/read pretrained weights.
  require(node.__name__=='author_upscaler' and Path(node.__file__).resolve()==Path(STAGE)/'upscaler_node.py','Unexpected upstream node import')
  require('author_upscaler' not in sys.modules,'Unexpected upstream model module collision')
  sys.modules[node.__name__]=node # Required by inspect.getfile(type(model)); no scientific source change.
  cfg=node._detect_arch({n:types.SimpleNamespace(shape=tuple(r['shape'])) for n,r in schema.items()})
  require(cfg==FACTORY_CFG,'Original architecture inferred from pinned shape metadata differs')
  with torch.device('cpu'):
   model=node.LatentResizer3D(**cfg)
  models.append(model);model.eval()
  params=tuple(model.parameters());names=tuple(model.state_dict());ids={n:id(p) for n,p in model.named_parameters()}
  require(len(names)==TENSORS and sum(p.numel() for p in params)==PARAMETERS,'Not the322tensor345280216parameter architecture')
  require(all(p.device.type=='cpu' and p.dtype==torch.float32 for p in params),'Model constructed outside CPU FP32')
  loader.validate_inventory(loader._tensor_inventory(model.state_dict(),torch),schema)
  stages.append({'stage':'constructed_CPU_before_v5_load','memory':memory_snapshot()})
  def forbid_forward(_module,_args):raise RuntimeError('Forward is outside this CPU-load-only smoke')
  hooks=[m.register_forward_pre_hook(forbid_forward) for m in model.modules()]
  source_paths={n:plan['files'][n]['path'] for n in loader.FROZEN_SOURCES}
  student,receipt=loader.load_initial_v5_cpu(REAL_CHECKPOINT,node=node,original_model=model,tensor_manifest_path=plan['files']['h3_causal_v5_tensor_validation.json']['path'],source_paths=source_paths,activation_checkpointing=False)
  require(ids=={n:id(p) for n,p in model.named_parameters()} and names==tuple(student.state_dict()),'Real345M Parameter identity/original names changed')
  require(sum(p.numel() for p in student.parameters())==PARAMETERS and len(student.state_dict())==TENSORS,'Wrapped real architecture count differs')
  require(all(p.grad is None and p.device.type=='cpu' and p.dtype==torch.float32 for p in model.parameters()),'Unexpected gradient/device/dtype')
  loader._verify_tensor_values(student.original_state_dict(),schema,torch)
  require(not any(k=='torch_npu' or k.startswith('torch_npu.') for k in sys.modules),'Unexpected torch_npu import')
  require('train_upscaler_trial' not in sys.modules and 'prepare_data' not in sys.modules,'Inactive training/device helpers unexpectedly imported')
  stages.append({'stage':'strict_v5_load_and_wrap_complete','memory':memory_snapshot()})
  after_sources=verify_stage(plan);require(after_sources==before_sources,'Source files changed during smoke')
  upstream_after={k:identity(c['files'][k]['path']) for k in upstream_before};require(upstream_before==upstream_after,'Original upstream changed')
  selected_after=selected_identity(c);require(selected_before==selected_after,'Checkpoint/alias changed during smoke')
  import einops
  result={'status':'passed_actual345M_CPU_load_only','started_utc':started,'completed_utc':utc(),'pid':os.getpid(),'execution_plan_sha256':plan_sha,'torch_version':torch.__version__,'torch_source':identity(torch.__file__),'einops_version':einops.__version__,'einops_source':identity(einops.__file__),'cpu_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'torch_npu_imported':False,'source_before':before_sources,'source_after':after_sources,'upstream_before':upstream_before,'upstream_after':upstream_after,'checkpoint_before':selected_before,'checkpoint_after':selected_after,'archive_metadata_audit':archive,'architecture_config':cfg,'actual_parameter_count':sum(p.numel() for p in model.parameters()),'actual_tensor_count':len(model.state_dict()),'actual_model_parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),'parameter_identity_preserved_through_strict_load_and_wrapper':True,'bit_values_match_all322_published_original_tensor_hashes':True,'strict_loader_receipt':receipt,'effective_input':[1,24,67,44,80],'explicit_target':[67,68,120],'historical_scale2_is_only_provenance':True,'construction_note':'Original architecture CPU constructor executes its normal allocation/initialization before strict copy; no aligned-past initializer or kernel shift is called on saved v5 weights.','memory_samples':stages,'address_space_cap_bytes':cap,'measured_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'memory_scope':'CPU construction+checkpoint deserialize+strict copy/hash only; not NPU, inference activations, decoder, backward, optimizer or training peak.','base_pretrained_weights_read':False,'forward_executed':False,'gradient_executed':False,'optimizer_created_or_restored':False,'decoder_loaded':False,'NPU_model_execution':False,'formal_training_allowed':False}
  record(out/'report.json',result);print(json.dumps({'status':result['status'],'report':identity(out/'report.json')},sort_keys=True),flush=True)
 except BaseException as exc:
  record(out/'failure.json',{'status':'failed_no_retry','utc':utc(),'exception_type':type(exc).__name__,'exception':str(exc),'memory_samples':stages,'formal_training_allowed':False});raise


def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('mode',choices=['check-plan','run']);a.add_argument('--plan',required=True,type=Path);a.add_argument('--plan-sha256',required=True);args=a.parse_args()
 plan=load_plan(args.plan,args.plan_sha256)
 if args.mode=='check-plan':
  verify_stage(plan);verify_contract(json.loads(Path(plan['files']['surrogate67_contract.json']['path']).read_text()));print(json.dumps({'status':'plan_and_small_sources_checked_only','weights_read':False,'torch_imported':False}));return
 run(plan,args.plan,args.plan_sha256)

if __name__=='__main__':main()
