"""Candidate fixed3dev full226 evaluation; separate infer/pixels, no train/test.
Requires a future real terminal-v2 binding and an owned three-lock launcher.
No launcher, model/array access, device setup or output occurs on import.
"""
from pathlib import Path
import argparse, copy, hashlib, importlib.util, inspect, json, os, shutil, sys, time
ROOT='/path/to/h3-project'
STAGE='/path/to/staging/h3_production67_limiteddev_eval_20261107_v1'
OUTPUT=ROOT+'/upscaler_production67_limiteddev_evaluation_20261107_v1'
STAGES=('infer','pixels')
def require(v,m):
 if not v:raise ValueError(m)
def sha(p):
 p=Path(p);require(p.is_file() and not p.is_symlink(),'Regular file required');h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def record(p):
 p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def module(name,p):
 require(name not in sys.modules,'Unexpected module already loaded');spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def verify_plan(path,digest):
 require(sha(path)==digest,'Exact evaluation plan required');p=read(path)
 require(p['schema']=='h3-production67-limiteddev-full226-plan-v1' and p['stage_directory']==STAGE and p['output']==OUTPUT and p['stages']==list(STAGES),'Wrong evaluation plan')
 for name,d in p['files'].items():require(d['path']==STAGE+'/'+name and record(d['path'])==d,'Evaluation source changed '+name)
 require(p['files'][Path(__file__).name]['path']==str(Path(__file__).resolve()),'Exact staged worker required')
 require(record(p['protocol_descriptor']['path'])==p['protocol_descriptor'] and read(p['protocol_descriptor']['path'])==p['protocol'],'Embedded/frozen protocol differs')
 import h3_production67_limiteddev_eval_contract_v1 as C
 require(p['protocol']['schema']==C.SCHEMA and p['protocol']['dev_order']==list(C.DEV) and p['precision']==C.PRECISION and p['mode']==C.MODE,'Fixed3dev protocol changed')
 require(p['protocol']['frozen_metric_source_sha256']==p['files']['h3_production67_eval_metrics_v1.py']['sha256']=='d6f0d6c588930dd00b7536840e0d675bd0f001975728c7e3278aaf3db139b7f0','Metric implementation changed')
 require(p['protocol']['training_protocol_sha256']==C.TRAIN_PROTOCOL and p['protocol']['test_unlocked'] is False and p['protocol']['checkpoint_reselection'] is False,'No test/reselection allowed')
 return p,C

def verify_terminal(args,plan,C):
 require(record(args.terminal_binding)['sha256']==args.terminal_binding_sha256,'Actual terminal binding SHA required')
 t=read(args.terminal_binding);C.terminal_identity(t)
 import h3_production67_limiteddev_terminal_binding_v2 as T
 require(sha(inspect.getfile(T))==C.TERMINAL_SOURCE and sha(args.terminal_policy)==C.TERMINAL_POLICY,'Reviewed terminal verifier/policy changed')
 require(T.bind(args.terminal_policy,t['arms'][0]['outer']['sha256'],t['arms'][1]['outer']['sha256'])==t,'Actual terminal source/receipts no longer reproduce reviewed binding')
 return t

def runtime_mode(torch,C):
 actual={'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'warn_only':torch.is_deterministic_algorithms_warn_only_enabled()}
 require(actual==C.MODE,'Strict deterministic mode changed');return actual

def prepare_runtime(args,plan,C):
 p=str(args.physical_device);require(args.physical_device in plan['physical_devices'] and os.environ.get('ASCEND_RT_VISIBLE_DEVICES')==p and os.environ.get('ASCEND_VISIBLE_DEVICES')==p and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0','Physical device/environment not explicit')
 legacy=module('limited_eval_frozen_legacy',plan['files']['evaluate_h3_production67_wing_v4.py']['path']);legacy.launch_admission(args)
 guard=module('limited_eval_cann_guard',plan['files']['h3_production67_cann_imports_v1.py']['path']);guard.staged_model_origins(plan,STAGE);vendor=guard.vendor_check(import_tbe=True)
 for d in plan['runtime_evidence_files']:require(record(d['path'])==d,'Installed runtime source changed')
 require(os.environ.get('ASCEND_HOME_PATH')==plan['selected_cann_home'] and os.environ.get('ASCEND_OPP_PATH')==plan['selected_cann_home']+'/opp','Selected CANN environment changed')
 require('torch_npu' not in sys.modules,'NPU backend must not be loaded before strict mode')
 import torch
 require(torch.__version__=='2.7.1+cpu','Torch version changed');torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.float32)
 torch.use_deterministic_algorithms(True,warn_only=False);runtime_mode(torch,C)
 import torch_npu
 require(torch_npu.__version__=='2.7.1.post2' and not torch.npu.is_initialized(),'Mode must be fixed before NPU initialization')
 folder=Path(OUTPUT)/args.stage;folder.mkdir(exist_ok=False)
 write(folder/'mode_configured_before_NPU.json',{'stage':args.stage,'mode':runtime_mode(torch,C),'NPU_initialized':False,'torch':torch.__version__,'torch_npu':torch_npu.__version__,'CANN_vendor':vendor})
 device=torch.device('npu:0');torch.npu.set_device(device);torch.npu.conv.allow_hf32=False;torch.npu.matmul.allow_hf32=False;torch.set_float32_matmul_precision('highest')
 require(legacy.precision(torch,device)==C.PRECISION,'Precision changed');runtime_mode(torch,C)
 return torch,device,legacy

def checked_dataset(plan,t,C):
 import h3_production67_cohort_router_v1 as R
 ds=R.read_router(ROOT,plan['router'])
 require(ds['descriptor']['sha256']==C.REGISTRY and ds['cohort_science_sha256']==t['cohort_science_sha256'] and ds['generation_mode_semantics']==t['generation_mode_semantics'],'Original cohort identities changed')
 rows=R.rows_for(ds,'dev');require([r['sample_id'] for r in rows]==list(C.DEV) and [r['source_group'] for r in rows]==list(C.FAMILIES),'Exact three windows/families required')
 return R,ds

def current_stat(p):
 p=Path(p);require(p.is_file() and p.resolve()==p and not p.is_symlink(),'Immutable original regular payload required');s=p.stat()
 return {'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns,'inode':s.st_ino,'device':s.st_dev}

def load_clip(R,ds,sid,t):
 import h3_production67_pair_loader_v2 as L
 pair,teacher,contract=R.load_bundles(ds,sid,split='dev')
 expected=next(r for r in t['dev_input_identities'] if r['sample_id']==sid)
 require(pair['record']['split']==teacher['record']['split']=='dev' and pair['record']['sample_id']==teacher['record']['sample_id']==sid and pair['pair_record_sha256']==expected['pair_record_sha256'] and teacher['teacher_record_sha256']==expected['teacher_record_sha256'] and pair['record']['data_contract_sha256']==expected['child_contract_sha256'],'Current actual payloads differ from fixed dev identities')
 row=next(r for r in ds['rows'] if r['sample_id']==sid);files={}
 for base,descs in ((Path(ROOT)/row['pair_root'],pair['record']['arrays'].values()),(Path(ROOT)/row['teacher_root'],[teacher['record']['latent']])):
  for d in descs:
   p=base/d['path'];files[str(p)]={'sha256':d['sha256'],**current_stat(p)}
 native=pair['record']['arrays']['native_hr_rgb'];native_desc={'path':str(Path(ROOT)/row['pair_root']/native['path']),**{k:native[k] for k in ('bytes','sha256')}}
 proof={'sample_id':sid,'split':'dev','source_group':row['source_group'],'cohort':ds['cohort_for_sample'][sid],
  'pair_record_sha256':pair['pair_record_sha256'],'teacher_record_sha256':teacher['teacher_record_sha256'],'child_contract_sha256':pair['record']['data_contract_sha256'],
  'source_identity':ds['source_identities'][sid],'native_array':native_desc,'full_payload_SHA_shape_finite_verified_by_loader2':True,'payload_snapshot':files}
 return L,pair,teacher,proof

def unchanged_payloads(proofs):
 for proof in proofs.values():
  for p,v in proof['payload_snapshot'].items():require(current_stat(p)=={k:v[k] for k in current_stat(p)},'Loaded immutable dev payload changed')

def selected_states(torch,t,schema,C):
 values={};proofs={}
 for a in t['arms']:
  d=a['selected_model_descriptor'];require(record(d['path'])=={k:d[k] for k in ('path','bytes','sha256')},'Selected model-only whole-file SHA/size differs')
  v=torch.load(d['path'],map_location='cpu',weights_only=True,mmap=True);C.selected_header(v,a);state=v['model']
  require(isinstance(state,dict),'Selected model state must be dictionary')
  facts={}
  for k,z in state.items():
   require(isinstance(z,torch.Tensor),'Non-tensor selected state')
   facts[k]={'shape':list(z.shape),'dtype':str(z.dtype),'device':z.device.type,'finite':bool(torch.isfinite(z).all()),'numel':z.numel()}
  total=C.tensor_facts(facts,schema)
  require(record(d['path'])=={k:d[k] for k in ('path','bytes','sha256')},'Selected payload changed while loading')
  label='teacher_student' if a['arm']=='teacher' else 'control';values[label]=state
  proofs[label]={'selected_model':d,'schema':v['schema'],'metrics':v['metrics'],'tensor_count':322,'parameter_count':total,'tensor_facts_sha256':canonical(facts),'all_shapes_FP32_finite_verified':True,'optimizer_or_rng_loaded':False}
 return values,proofs

def save_npy(path,array):
 import numpy as np
 with Path(path).open('xb') as f:np.save(f,array,allow_pickle=False);f.flush();os.fsync(f.fileno())
 return record(path)

def infer(args,plan,t,C,torch,device,legacy):
 import h3_production67_train_step_v1 as core
 R,ds=checked_dataset(plan,t,C);folder=Path(OUTPUT)/'infer';recipe=core.frozen_recipe(STAGE)
 factory=module('limited_eval_original_v5_factory',plan['files']['check_h3_production67_explicit_hf32off_npu_v1.py']['path'])
 student,loader,schema,paths,initial,checkpoint=factory.build(plan,torch)
 values,model_proofs=selected_states(torch,t,schema,C)
 values['v5']={k:z.detach().clone() for k,z in student.original_state_dict().items()}
 model_proofs['v5']={'kind':'original_v5_step1800','checkpoint_sha256':C.V5,'strict_initial_load':initial,'optimizer_or_rng_loaded':False}
 predictions={};payloads={}
 for sid in C.DEV:
  L,pair,teacher,proof=load_clip(R,ds,sid,t);payloads[sid]=proof
  arrays=L.torch_arrays(pair);draft=arrays['draft'].unsqueeze(0).to(device);reference=arrays['reference_hr'].unsqueeze(0).to(device);teacher_z=L.torch_teacher(teacher).unsqueeze(0).to(device);flow=arrays['flow'].to(device);valid=arrays['flow_valid'].to(device)
  core.validate_example_tensors({'draft':draft,'reference':reference,'teacher':teacher_z,'flow':flow,'valid':valid})
  sub=folder/sid;sub.mkdir();predictions[sid]={}
  for label in C.MODELS:
   student.to('cpu');student.load_state_dict(values[label],strict=True,assign=False)
   require(all(torch.equal(student.original_state_dict()[k],z) for k,z in values[label].items()),'Loaded exact selected/v5 weights differ')
   student.to(device).eval().requires_grad_(False);legacy.frozen(student,device);before=legacy.frozen_versions(student);runtime_mode(torch,C);legacy.precision(torch,device);start=time.monotonic()
   with torch.no_grad():pred=core.explicit_forward67(student,draft);terms=core.latent_terms_all(pred,reference,teacher_z,flow,valid,recipe)
   torch.npu.synchronize(device);require(bool(torch.isfinite(pred).all()) and all(bool(torch.isfinite(v).all()) for v in terms.values()),'Nonfinite complete latent output/metrics')
   require(before==legacy.frozen_versions(student),'Frozen evaluated student changed');legacy.frozen(student,device);legacy.precision(torch,device);runtime_mode(torch,C)
   array=pred[0].float().cpu().contiguous().numpy();latent=save_npy(sub/(label+'.npy'),array)
   result={'status':'completed_fixed_prediction','sample_id':sid,'variant':label,'plan_sha256':args.plan_sha256,'terminal_binding_sha256':args.terminal_binding_sha256,'input_identity':{k:proof[k] for k in ('pair_record_sha256','teacher_record_sha256','child_contract_sha256')},'evaluated_model':model_proofs[label],
    'latent':latent,'shape':list(array.shape),'dtype':'float32','latent_metrics':{k:float(v) for k,v in terms.items()},'all67_tokens_and66_pairs':True,'raw_legacy_eWarp_preserved':True,'precision':legacy.precision(torch,device),'mode':runtime_mode(torch,C),'frozen_unchanged':True,'seconds':time.monotonic()-start,'optimizer_updates':0}
   rp=sub/(label+'.json');write(rp,result);predictions[sid][label]=record(rp);print(json.dumps({'stage':'infer','sample_id':sid,'variant':label}),flush=True)
   del pred,terms,array;student.to('cpu');torch.npu.empty_cache()
  del arrays,draft,reference,teacher_z,flow,valid,pair,teacher;torch.npu.empty_cache()
 unchanged_payloads(payloads)
 for a in t['arms']:
  d=a['selected_model_descriptor'];require(record(d['path'])=={k:d[k] for k in ('path','bytes','sha256')},'Selected model changed after all predictions')
 write(folder/'report.json',{'status':'completed','stage':'infer','plan_sha256':args.plan_sha256,'terminal_binding_sha256':args.terminal_binding_sha256,'dev_order':list(C.DEV),'predictions':predictions,'model_proofs':model_proofs,'payload_proofs':payloads,'actual_new_upscaler_forwards':9,'actual_new_decoder_forwards':0,'optimizer_updates':0,'precision':C.PRECISION,'mode':C.MODE,'formal16dev_gate_passed':False,'test_unlocked':False})

def verify_infer(args,plan,C):
 folder=Path(OUTPUT)/'infer';op=Path(OUTPUT)/'launches/infer_attempt1.json'
 require(args.infer_outer_sha256 is not None and sha(op)==args.infer_outer_sha256,'Actual completed infer outer SHA required')
 outer=read(op);require(outer['status']=='completed' and outer['stage']=='infer' and outer['exit_code']==0 and outer['optimizer_updates']==0 and outer['plan_sha256']==args.plan_sha256,'Inference parent not successful')
 require(outer['cleanup']['reaped'] is True and outer['cleanup']['returncode']==0 and outer['cleanup']['live_group_members_at_end']==[],'Inference child not reaped')
 require(not (folder/'failure.json').exists(),'Inference postcheck failed');done=read(folder/'complete.json');r=read(folder/'report.json')
 require(done['status']=='completed_after_all_checks' and done['terminal_binding_sha256']==args.terminal_binding_sha256 and record(folder/'report.json')==done['report']==outer['report'],'Completed inference report identity differs')
 require(r['status']=='completed' and r['stage']=='infer' and r['plan_sha256']==args.plan_sha256 and r['terminal_binding_sha256']==args.terminal_binding_sha256 and r['dev_order']==list(C.DEV) and r['actual_new_upscaler_forwards']==9 and r['actual_new_decoder_forwards']==0 and r['optimizer_updates']==0 and r['mode']==C.MODE and r['precision']==C.PRECISION,'Incomplete9 predictions')
 for sid in C.DEV:
  require(set(r['predictions'][sid])==set(C.MODELS),'Incomplete model coverage')
  for label,d in r['predictions'][sid].items():
   require(record(d['path'])==d,'Prediction receipt changed');x=read(d['path'])
   require(x['sample_id']==sid and x['variant']==label and x['plan_sha256']==args.plan_sha256 and x['terminal_binding_sha256']==args.terminal_binding_sha256 and x['shape']==[24,67,68,120] and x['dtype']=='float32' and x['frozen_unchanged'] is True and x['mode']==C.MODE,'Prediction identity changed')
   require(record(x['latent']['path'])==x['latent'],'Prediction full bytes/SHA changed')
 return r

def decode_visible(vae,z,device,torch,decode_latent):
 with torch.no_grad():raw=decode_latent(vae,z,device,keep_frames=None)
 require(tuple(raw.shape)==(226,1088,1920,3) and raw.dtype==torch.uint8,'Raw decoder output must be226x1088, no time trim')
 rgb=raw[:,4:1084].contiguous();require(tuple(rgb.shape)==(226,1080,1920,3),'Visible crop differs')
 return raw,rgb

def retain_rgb(folder,label,rgb,origin,raw_shape=None,native_array=None):
 require(tuple(rgb.shape)==(226,1080,1920,3) and str(rgb.dtype)=='torch.uint8' and rgb.device.type=='cpu' and rgb.is_contiguous(),'Exact lossless visible RGB required')
 rgbsha=hashlib.sha256(memoryview(rgb.numpy()).cast('B')).hexdigest()
 payload=native_array if native_array is not None else save_npy(folder/(label+'_RGB.npy'),rgb.numpy())
 return {'origin':origin,'payload':payload,'kind':'existing_verified_native_array' if native_array is not None else 'retained_same_metric_lossless_RGB',
  'raw_shape':raw_shape,'visible_shape':[226,1080,1920,3],'spatial_crop':None if native_array is not None else [4,1084,0,1920],'time_crop':None,'frames':226,'fps':[24,1],'duration_seconds':226/24,'expected_presentation_indices':[0,225],
  'visible_RGB_sha256':rgbsha,'video_encoded':False,'video_PTS_audited':False,'native_display_pixels':True}

def pixels(args,plan,t,C,torch,device,legacy):
 import numpy as np
 import h3_production67_eval_metrics_v1 as metrics
 from data_utils import load_video_vae,decode_latent
 evidence=verify_infer(args,plan,C);R,ds=checked_dataset(plan,t,C);folder=Path(OUTPUT)/'pixels'
 for d in plan['runtime_assets'].values():require(record(d['path'])==d,'Frozen decoder/LPIPS dependency changed')
 vae=load_video_vae(plan['base_model'],device).eval().requires_grad_(False);require(sha(inspect.getfile(type(vae)))==plan['decoder_implementation_sha256'],'Decoder implementation changed')
 import lpips
 require(sha(Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth')==plan['lpips_fingerprint']['backbone_sha256'] and sha(Path(lpips.__file__).resolve().parent/'weights/v0.1/alex.pth')==plan['lpips_fingerprint']['linear_sha256'],'Local LPIPS weights required; no network fallback')
 perceptual=lpips.LPIPS(net='alex',version='0.1').to(device).eval().requires_grad_(False);require(sha(inspect.getfile(type(perceptual)))==plan['lpips_fingerprint']['implementation_sha256'],'LPIPS implementation changed')
 legacy.frozen(vae,device);legacy.frozen(perceptual,device);before=[legacy.frozen_versions(m) for m in (vae,perceptual)];rows=[];receipts=[];payloads={}
 for sid in C.DEV:
  sub=folder/sid;sub.mkdir();L,pair,teacher,proof=load_clip(R,ds,sid,t);payloads[sid]=proof
  require(proof==evidence['payload_proofs'][sid],'Original dev payloads changed since inference')
  arrays=L.torch_arrays(pair,include_native_hr_rgb=True);teacher_z=L.torch_teacher(teacher);decoded={};retained={};targets={d:{} for d in C.TARGETS}
  for label,z,origin in (('teacher',teacher_z,{'teacher_record_sha256':proof['teacher_record_sha256']}),('decoded_hr',arrays['reference_hr'],{'pair_record_sha256':proof['pair_record_sha256']})):
   runtime_mode(torch,C);legacy.precision(torch,device);raw,rgb=decode_visible(vae,z,device,torch,decode_latent);decoded[label]=rgb;retained[label]=retain_rgb(sub,label,rgb,origin,list(raw.shape));del raw
  decoded['native_hr']=arrays['native_hr_rgb'];retained['native_hr']=retain_rgb(sub,'native_hr',decoded['native_hr'],proof['source_identity'],native_array=proof['native_array'])
  for label in C.MODELS:
   pr=read(evidence['predictions'][sid][label]['path']);a=np.load(pr['latent']['path'],allow_pickle=False)
   require(a.shape==(24,67,68,120) and a.dtype==np.float32 and np.isfinite(a).all(),'Complete finite prediction required');z=torch.from_numpy(a.copy())
   runtime_mode(torch,C);legacy.precision(torch,device);raw,rgb=decode_visible(vae,z,device,torch,decode_latent);retained[label]=retain_rgb(sub,label,rgb,pr['latent'],list(raw.shape));del raw
   for domain in C.TARGETS:targets[domain][label]=metrics.rgb_metrics(rgb,decoded[domain],decoded['decoded_hr'],perceptual,device)
   require(hashlib.sha256(memoryview(rgb.numpy()).cast('B')).hexdigest()==retained[label]['visible_RGB_sha256'],'Metric RGB changed after retention')
   write(sub/(label+'_metrics.json'),{'variant':label,'sample_id':sid,'targets':{d:targets[d][label] for d in C.TARGETS},'retained_RGB':retained[label]})
   del rgb,z,a;torch.npu.empty_cache()
  result={'sample_id':sid,'split':'dev','frames':226,'fps':[24,1],'duration_seconds':226/24,'plan_sha256':args.plan_sha256,'terminal_binding_sha256':args.terminal_binding_sha256,
   'targets':targets,'frame_regions':metrics.frame_regions226(),'transition_pair_groups':metrics.transition_pairs226(),'latent_metrics':{m:read(evidence['predictions'][sid][m]['path'])['latent_metrics'] for m in C.MODELS},
   'retained_RGB':retained,'payload_proof':proof,'optimizer_updates':0,'actual_decoder_forwards':5,'checkpoint_reselection':False,'test_unlocked':False,'formal16dev_gate_passed':False,'visual_quality_passed':False,
   'ROI_definition':'Same detached decodedHR abs adjacent RGB channel-mean change>=.05 for all targets/models','metric_domain':'Lossless decoder uint8/255 and separately verified native source uint8/255; LPIPS[-1,1]; nativeHR is not teacher.'}
  C.validate_clip(result);result['regressions']=metrics.regressions(targets);result['conditions']=C.condition_result(targets,metrics,'Individual fixed limited-dev clip; diagnostic only, no new selection/test gate.')
  for label,v in retained.items():require(record(v['payload']['path'])==v['payload'],'Retained/native RGB full-file SHA changed')
  rp=sub/'summary.json';write(rp,result);receipts.append(record(rp));rows.append(result)
  require(before==[legacy.frozen_versions(m) for m in (vae,perceptual)],'Frozen VAE/LPIPS changed');legacy.frozen(vae,device);legacy.frozen(perceptual,device);runtime_mode(torch,C);legacy.precision(torch,device)
  del arrays,teacher_z,decoded,pair,teacher;torch.npu.empty_cache();print(json.dumps({'stage':'pixels','sample_id':sid,'actual_decodes':5}),flush=True)
 unchanged_payloads(payloads);summary={'status':'completed_limiteddev3_full226_evaluation_not_visual_acceptance','plan_sha256':args.plan_sha256,'terminal_binding_sha256':args.terminal_binding_sha256,
  'clip_summaries':receipts,'equal_clip_summary':C.aggregate(rows,metrics),'cohort_science_sha256':t['cohort_science_sha256'],'generation_mode_semantics':t['generation_mode_semantics'],'precision':C.PRECISION,'mode':C.MODE,'selected_steps':{a['arm']:a['selected_step'] for a in t['arms']},
  'actual_decoder_forwards':15,'actual_new_upscaler_forwards_this_stage':0,'optimizer_updates':0,'formal16dev_gate_passed':False,'test_unlocked':False,'checkpoint_reselection':False,
  'media_ready_RGB_only':True,'video_encoded':False,'video_PTS_audited':False,'continuous_playback_reviewed':False,'visual_quality_passed':False,
  'limits':plan['protocol']['limits'],'future_media_routes':['native_hr','teacher','v5','control','teacher_student'],'additional_scientific_decode_needed_for_media':False}
 write(folder/'summary.json',summary);write(folder/'report.json',{'status':'completed','stage':'pixels','plan_sha256':args.plan_sha256,'terminal_binding_sha256':args.terminal_binding_sha256,'summary':record(folder/'summary.json'),'actual_decoder_forwards':15,'optimizer_updates':0})

def run(args):
 sys.path[:]=[STAGE]+[p for p in sys.path if p and p!=STAGE];plan,C=verify_plan(args.plan,args.plan_sha256);t=verify_terminal(args,plan,C)
 require(args.stage in STAGES and (args.infer_outer_sha256 is None if args.stage=='infer' else args.infer_outer_sha256 is not None),'Stages require explicit real predecessors, never all')
 # Check later-stage predecessor before any device action.
 if args.stage=='pixels':verify_infer(args,plan,C)
 folder=Path(OUTPUT)/args.stage
 require(not folder.exists(),'Exclusive one attempt; preserve failure, no retry')
 require(shutil.disk_usage(Path(OUTPUT).parent).free>=plan['minimum_free_bytes'][args.stage],'Insufficient storage for bounded retained evidence')
 try:
  torch,device,legacy=prepare_runtime(args,plan,C)
  {'infer':infer,'pixels':pixels}[args.stage](args,plan,t,C,torch,device,legacy)
  require(verify_terminal(args,plan,C)==t,'Terminal metadata changed during evaluation');verify_plan(args.plan,args.plan_sha256)
  runtime_mode(torch,C);legacy.precision(torch,device)
  for d in plan['runtime_assets'].values():require(record(d['path'])==d,'Runtime asset changed')
  write(folder/'complete.json',{'status':'completed_after_all_checks','stage':args.stage,'plan_sha256':args.plan_sha256,'terminal_binding_sha256':args.terminal_binding_sha256,'report':record(folder/'report.json'),'optimizer_updates':0})
 except BaseException as e:
  folder.mkdir(exist_ok=True);write(folder/'failure.json',{'status':'failed_no_auto_retry','stage':args.stage,'exception_type':type(e).__name__,'exception':str(e),'optimizer_updates':0});raise

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=STAGES);ap.add_argument('--physical-device',type=int,required=True)
 for n in ('plan','plan-sha256','terminal-binding','terminal-binding-sha256','terminal-policy','launch-receipt'):ap.add_argument('--'+n,required=True)
 ap.add_argument('--infer-outer-sha256');run(ap.parse_args())
