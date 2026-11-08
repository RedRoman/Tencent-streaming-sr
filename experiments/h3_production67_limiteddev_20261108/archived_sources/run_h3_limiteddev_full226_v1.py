"""Separate fixed3dev full226 infer/pixels owned launcher candidate.
No worker/science-plan mutation, all-mode, retries, background waiter or training.
A real terminal-v2 binding is mandatory before any accelerator admission.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,re,sys,shutil
from pathlib import Path
EXEC_STAGE=Path('/path/to/staging/h3_production67_limiteddev_eval_launch_20261107_v1')
STAGE=Path('/path/to/staging/h3_production67_limiteddev_eval_20261107_v1')
OUTPUT=Path('/path/to/h3-project/upscaler_production67_limiteddev_evaluation_20261107_v1')
SCIENCE_PLAN=STAGE/'h3_production67_limiteddev_full226_plan_v1.json'
SCIENCE_SHA='d031e4383c689bc03615a954bdb9b37fe7970b8425b24775a93b7f7262edb737'
MANIFEST=EXEC_STAGE/'h3_limiteddev_full226_execution_manifest_v1.json'
RUNTIME=Path('/path/to/model-assets/runtime/.venv-npu/bin/python')
ENV_SOURCE=Path('/path/to/h3-project/experiment/env.sh')
ENV_SHA='64451094fa9aca640bc5bfc9ba77080330f858d1706ba74798ca4eb8834e4524'
UTIL_SHA='fd50e29519d7deeee4ca0f7663ca5ec11575009a6a6e8b958af5e3fdf8855cc5'
STAGES=('infer','pixels')


def require(ok,message):
 if not ok:raise ValueError(message)


def sha(path):
 path=Path(path);require(path.is_file() and not path.is_symlink(),'Expected regular source');return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name,path):
 require(name not in sys.modules,'Unexpected imported module '+name);s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m


def locks(device):
 require(type(device) is int and 0<=device<=7,'Require an explicit physical device0..7')
 return tuple(Path(p) for p in (f'/path/to/device-locks/npu{device}.lock',f'/path/to/staging/SHARED_LOCK_OWNER_npu_{device}.lock',f'/path/to/staging/codex-h3-vae-npu-{device}.lock'))


def require_idle(smi,device):
 locks(device)
 require(f'No running processes found in NPU {device}' in smi,'Selected card has processes or unknown process state')
 lines=smi.splitlines()
 for line in lines:require(not re.match(r'^\|\s*'+str(device)+r'\s+\d+\s*\|\s*\d+\s*\|',line),'Selected card process row contradicts idle marker')
 rows=[i for i,line in enumerate(lines) if re.match(r'^\|\s*'+str(device)+r'\s+910B3\s*\|',line)]
 require(len(rows)==1,'Cannot uniquely read selected card hardware');i=rows[0]
 require(lines[i].split('|')[2].strip()=='OK' and i+1<len(lines),'Selected card health/usage not verified')
 fields=lines[i+1].split('|');require(len(fields)==5,'Unexpected chip usage layout');v=[int(x) for x in re.findall(r'\d+',fields[3])]
 require(len(v)==5 and v[0]==0 and v[-1]-v[-2]>=60000,'Require idle selected card and60000MiB HBM free')
 return {'physical_npu':device,'health':'OK','aicore_percent':v[0],'hbm_used_mb':v[-2],'hbm_total_mb':v[-1],'no_running_processes':True}



def record(path):
 p=Path(path);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}

def exact_path(path):
 p=Path(path);require(p.is_absolute() and p.resolve()==p,'Exact absolute path without symlink aliases required');return p

def load_import_guard():
 name='h3_limited_eval_launcher_cann_guard'
 return sys.modules.get(name) or module(name,STAGE/'h3_production67_cann_imports_v1.py')

def child_environment(device):
 locks(device);env=dict(os.environ);guard=load_import_guard();pythonpath=guard.child_pythonpath(STAGE,env.get('PYTHONPATH',''));env.update(ASCEND_VISIBLE_DEVICES=str(device),ASCEND_RT_VISIBLE_DEVICES=str(device),CUDA_VISIBLE_DEVICES='',TORCH_DEVICE_BACKEND_AUTOLOAD='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONNOUSERSITE='1',PYTHONPATH=pythonpath);return env

def arguments(args):
 require(args.stage in STAGES and type(args.physical_device) is int and args.physical_device in (4,7),'Only explicit infer/pixels on4or7; no all/retry')
 require((args.infer_outer_sha256 is None) == (args.stage=='infer'),'Pixels requires actual infer outerSHA; infer forbids it')
 for d in (args.execution_manifest_sha256,args.terminal_binding_sha256,*([args.infer_outer_sha256] if args.stage=='pixels' else [])):
  require(isinstance(d,str) and re.fullmatch('[0-9a-f]{64}',d) is not None,'Require full actual SHA256')
 require(exact_path(args.execution_manifest)==MANIFEST,'Wrong isolated execution manifest')
 require(exact_path(args.terminal_policy)==STAGE/'h3_production67_limiteddev_terminal_policy_v1.json','Exact frozen terminal policy required')
 exact_path(args.terminal_binding)
 args.plan=str(SCIENCE_PLAN);args.plan_sha256=SCIENCE_SHA

def predecessor(worker,args,plan,C):
 t=worker.verify_terminal(args,plan,C)
 if args.stage=='pixels':worker.verify_infer(args,plan,C)
 return t

def source_check(args):
 arguments(args);require(sha(args.execution_manifest)==args.execution_manifest_sha256,'Execution manifest SHA differs');m=json.loads(Path(args.execution_manifest).read_text())
 require(m['schema']=='h3-limiteddev-full226-execution-manifest-v1' and m['execution_stage']==str(EXEC_STAGE) and m['science_stage']==str(STAGE) and m['output']==str(OUTPUT) and m['stages']==list(STAGES) and m['physical_devices']==[4,7],'Execution scope changed')
 require(set(m['files'])=={'run_h3_limiteddev_full226_v1.py','run_h3_limiteddev_full226_v1.sh','run_h3_surrogate67_npu4_v1.py'},'Exact three launch files required')
 for name,d in m['files'].items():require(d['path']==str(EXEC_STAGE/name) and record(d['path'])==d,'Launch source changed '+name)
 require(m['files'][Path(__file__).name]['path']==str(Path(__file__).resolve()),'Run exact staged launcher')
 require(record(SCIENCE_PLAN)==m['scientific_plan'] and m['scientific_plan']['sha256']==SCIENCE_SHA,'Frozen scientific plan changed')
 require(record(ENV_SOURCE)==m['environment_source'] and sha(ENV_SOURCE)==ENV_SHA and m['runtime_python']==str(RUNTIME),'Frozen environment/runtime changed')
 require(sha(EXEC_STAGE/'run_h3_surrogate67_npu4_v1.py')==UTIL_SHA,'Owned process utility changed')
 sys.path[:]=[str(STAGE)]+[p for p in sys.path if p and p!=str(STAGE)]
 worker=module('h3_limited_eval_scientific_worker',STAGE/'evaluate_h3_production67_limiteddev_v1.py');plan,C=worker.verify_plan(args.plan,args.plan_sha256)
 require(plan['files']==m['scientific_files'] and plan['minimum_free_bytes']==m['minimum_free_bytes'],'Scientific source map/storage bounds changed')
 predecessor(worker,args,plan,C)
 return worker,plan,C,m

def command(args,receipt):
 require(args.stage in STAGES,'No all/retry supported')
 result=[str(RUNTIME),str(STAGE/'evaluate_h3_production67_limiteddev_v1.py'),args.stage,'--physical-device',str(args.physical_device),'--plan',str(SCIENCE_PLAN),'--plan-sha256',SCIENCE_SHA,'--terminal-binding',str(args.terminal_binding),'--terminal-binding-sha256',args.terminal_binding_sha256,'--terminal-policy',str(args.terminal_policy),'--launch-receipt',str(receipt)]
 if args.stage=='pixels':result+=['--infer-outer-sha256',args.infer_outer_sha256]
 return result

def output_paths(args):
 exact_path(OUTPUT)
 if args.stage=='infer':require(not OUTPUT.exists(),'Infer requires fresh evaluation root; no rerun/adopt partial')
 else:
  require(OUTPUT.is_dir() and not OUTPUT.is_symlink() and (OUTPUT/'launches').is_dir(),'Pixels requires actual inference root')
  exact_path(OUTPUT/'launches')
  allowed={'infer','launches'};require({p.name for p in OUTPUT.iterdir()}==allowed,'Unexpected evaluation-root entries before pixels')
  require({p.name for p in (OUTPUT/'launches').iterdir()}=={'stage.lock','infer_attempt1.json','infer_attempt1.log','infer_attempt1_npu_smi.txt'},'Unexpected launch evidence before pixels')
 require(not (OUTPUT/args.stage).exists() and not (OUTPUT/'launches'/(args.stage+'_attempt1.json')).exists(),'Stage already attempted; preserve evidence')

def completed(worker,args,plan,C,cleanup):
 require(cleanup is not None and cleanup['reaped'] is True and cleanup['returncode']==0 and cleanup['live_group_members_at_end']==[],'Child/group not successfully reaped')
 folder=OUTPUT/args.stage;require(not (folder/'failure.json').exists(),'Worker preserved failure')
 d=json.loads((folder/'complete.json').read_text());r=json.loads((folder/'report.json').read_text());mode=json.loads((folder/'mode_configured_before_NPU.json').read_text())
 require(d['status']=='completed_after_all_checks' and d['stage']==args.stage and d['plan_sha256']==SCIENCE_SHA and d['terminal_binding_sha256']==args.terminal_binding_sha256 and d['optimizer_updates']==0 and d['report']==record(folder/'report.json'),'Complete/report identity mismatch')
 require(r['status']=='completed' and r['stage']==args.stage and r['plan_sha256']==SCIENCE_SHA and r['terminal_binding_sha256']==args.terminal_binding_sha256 and r['optimizer_updates']==0,'Report identity differs')
 require(mode['stage']==args.stage and mode['mode']==plan['mode'] and mode['NPU_initialized'] is False and mode['torch']=='2.7.1+cpu' and mode['torch_npu']=='2.7.1.post2','Actual preinit mode evidence differs')
 if args.stage=='infer':
  require(r['actual_new_upscaler_forwards']==9 and r['actual_new_decoder_forwards']==0 and r['dev_order']==plan['protocol']['dev_order'] and r['mode']==plan['mode'] and r['precision']==plan['precision'],'Inference coverage differs')
 else:
  require(r['actual_decoder_forwards']==15 and record(r['summary']['path'])==r['summary'] and r['summary']['path']==str(folder/'summary.json'),'Pixel coverage/summary differs')
  s=json.loads((folder/'summary.json').read_text())
  require(s['status']=='completed_limiteddev3_full226_evaluation_not_visual_acceptance' and s['plan_sha256']==SCIENCE_SHA and s['terminal_binding_sha256']==args.terminal_binding_sha256 and s['mode']==plan['mode'] and s['precision']==plan['precision'] and s['actual_decoder_forwards']==15 and s['actual_new_upscaler_forwards_this_stage']==0 and s['optimizer_updates']==0 and all(s[k] is False for k in ('formal16dev_gate_passed','test_unlocked','checkpoint_reselection','video_encoded','visual_quality_passed')),'Pixel terminal scope differs')
 worker.verify_plan(args.plan,args.plan_sha256);worker.verify_terminal(args,plan,C)
 return {'report':record(folder/'report.json'),'complete':record(folder/'complete.json'),'mode_preinit':record(folder/'mode_configured_before_NPU.json')}

def run(args):
 worker,plan,C,manifest=source_check(args)
 require(shutil.disk_usage(OUTPUT.parent).free>=manifest['minimum_free_bytes'][args.stage],'Insufficient bounded output disk')
 imports=load_import_guard().cpu_preflight(plan,STAGE)
 if args.check_plan:
  print(json.dumps({'status':'terminal_and_execution_checked_CPU_only','stage':args.stage,'science_plan_sha256':SCIENCE_SHA,'execution_manifest_sha256':args.execution_manifest_sha256,'terminal_binding_sha256':args.terminal_binding_sha256,'physical_device_not_touched':args.physical_device,'CANN_import_preflight':imports}),flush=True);return
 output_paths(args)
 utils=module('h3_limited_eval_owned_process_utils',EXEC_STAGE/'run_h3_surrogate67_npu4_v1.py')
 if args.stage=='infer':OUTPUT.mkdir(exist_ok=False);(OUTPUT/'launches').mkdir(exist_ok=False)
 launches=OUTPUT/'launches';mutex=utils.acquire_file_lock(launches/'stage.lock');handles=[];cancel=utils.Cancellation();runner=utils.ChildRunner(cancel);receipt=launches/(args.stage+'_attempt1.json');owned=False
 state={'schema':'h3-limiteddev-full226-owned-launch-v1','stage':args.stage,'status':'admitting','parent_pid':os.getpid(),'physical_npu':args.physical_device,'plan_sha256':SCIENCE_SHA,'execution_manifest_sha256':args.execution_manifest_sha256,'terminal_binding':record(args.terminal_binding),'terminal_binding_sha256':args.terminal_binding_sha256,'terminal_policy':record(args.terminal_policy),'infer_outer_sha256':args.infer_outer_sha256,'started_utc':utils.now(),'all_three_locks_held':False,'optimizer_updates':0,'CANN_import_preflight':imports}
 try:
  require(not receipt.exists() and not (OUTPUT/args.stage).exists(),'Stage already attempted; no retry')
  with receipt.open('x') as f:json.dump(state,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  owned=True;cancel.install()
  for path in locks(args.physical_device):handles.append(utils.acquire_file_lock(path))
  state['all_three_locks_held']=True;state['lock_paths']=[str(p) for p in locks(args.physical_device)]
  env=child_environment(args.physical_device);smipath=launches/(args.stage+'_attempt1_npu_smi.txt')
  with smipath.open('x') as f:code=runner.run(['npu-smi','info'],env,f,timeout=20)
  require(code==0,'npu-smi failed');state['device_idle']=require_idle(smipath.read_text(),args.physical_device);state['npu_smi']=utils.record(smipath)
  predecessor(worker,args,plan,C);state['command']=command(args,receipt);state['environment']={k:env[k] for k in ('ASCEND_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','TORCH_DEVICE_BACKEND_AUTOLOAD','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONNOUSERSITE','PYTHONPATH')};state['status']='starting_child';utils.atomic_json(receipt,state)
  def started(process):
   state.update(status='running',child_pid=process.pid,child_pgid=process.pid);utils.atomic_json(receipt,state)
  log=launches/(args.stage+'_attempt1.log')
  with log.open('x') as f:code=runner.run(state['command'],env,f,on_started=started)
  state.update(cleanup=runner.cleanup,exit_code=code,log=utils.record(log));require(code==0,'Owned evaluator returned nonzero')
  result=completed(worker,args,plan,C,runner.cleanup)
  require(record(args.execution_manifest)['sha256']==args.execution_manifest_sha256 and sha(ENV_SOURCE)==ENV_SHA,'Execution inputs changed')
  for desc in manifest['files'].values():require(record(desc['path'])==desc,'Launch source changed during stage')
  state.update(status='completed',finished_utc=utils.now(),**result);utils.atomic_json(receipt,state);print(json.dumps({'status':'completed','receipt':utils.record(receipt)}),flush=True)
 except BaseException as e:
  if owned:
   state.update(status='failed_no_retry',error_type=type(e).__name__,error=str(e),finished_utc=utils.now(),cleanup=runner.cleanup);utils.atomic_json(receipt,state)
  raise
 finally:
  # Stable utility never releases these locks while the owned PGID can execute.
  if runner.active is not None:utils.terminate_owned(runner.active);utils.await_owned_quiescence(runner.active)
  for h in reversed(handles):h.close()
  mutex.close();cancel.restore_handlers()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=STAGES);p.add_argument('--physical-device',type=int,choices=(4,7),required=True)
 for name in ('execution-manifest','execution-manifest-sha256','terminal-binding','terminal-binding-sha256','terminal-policy'):p.add_argument('--'+name,required=True)
 p.add_argument('--infer-outer-sha256');p.add_argument('--check-plan',action='store_true');run(p.parse_args())
if __name__=='__main__':main()
