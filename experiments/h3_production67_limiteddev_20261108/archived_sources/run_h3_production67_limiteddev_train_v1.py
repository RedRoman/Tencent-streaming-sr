"""Three explicitly admitted deterministic optimizer stages: at most 5+4+2 updates.
Uses unchanged reviewed child-process ownership/cancellation utilities. Physical
card is explicit, actual npu-smi must prove idle, and all three card locks held.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,re,sys,traceback
from pathlib import Path
STAGE=Path('/path/to/staging/h3_production67_limiteddev_20261107_v1')
OUTPUT=Path('/path/to/h3-project/upscaler_production67_limiteddev_20261107')
RUNTIME=Path('/path/to/model-assets/runtime/.venv-npu/bin/python')
ENV_SOURCE=Path('/path/to/h3-project/experiment/env.sh')
ENV_SHA='64451094fa9aca640bc5bfc9ba77080330f858d1706ba74798ca4eb8834e4524'
UTIL_SHA='fd50e29519d7deeee4ca0f7663ca5ec11575009a6a6e8b958af5e3fdf8855cc5'
STAGES=('step0','control','teacher')


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


def load_import_guard():
 name='h3_launcher_cann_import_guard'
 return sys.modules.get(name) or module(name,STAGE/'h3_production67_cann_imports_v1.py')


def child_environment(device):
 locks(device);env=dict(os.environ);guard=load_import_guard();pythonpath=guard.child_pythonpath(STAGE,env.get('PYTHONPATH',''));env.update(ASCEND_VISIBLE_DEVICES=str(device),ASCEND_RT_VISIBLE_DEVICES=str(device),CUDA_VISIBLE_DEVICES='',TORCH_DEVICE_BACKEND_AUTOLOAD='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONNOUSERSITE='1',PYTHONPATH=pythonpath);return env


def command(args,receipt):
 require(args.stage in STAGES,'No all/retry stage supported')
 result=[str(RUNTIME),str(STAGE/'h3_production67_limiteddev_train_v1.py'),args.stage,'--physical-device',str(args.physical_device),'--plan',str(args.plan),'--plan-sha256',args.plan_sha256,'--launch-receipt',str(receipt)]
 if args.predecessor_report_sha256 is not None:result += ['--predecessor-report-sha256',args.predecessor_report_sha256]
 return result


def source_check(args):
 require(Path(args.plan).resolve()==STAGE/'h3_production67_limiteddev_plan_v1.json','Require new isolated plan path');require(sha(args.plan)==args.plan_sha256,'Plan SHA differs');require(sha(ENV_SOURCE)==ENV_SHA,'Frozen runtime environment source differs')
 require(sha(STAGE/'run_h3_surrogate67_npu4_v1.py')==UTIL_SHA,'Frozen process utility differs')
 worker=module('h3_leaf_AB_worker_plan',STAGE/'h3_production67_limiteddev_train_v1.py');plan=worker.verify_plan(args.plan,args.plan_sha256)
 require(plan['files'][Path(__file__).name]['path']==str(Path(__file__).resolve()),'Launcher must run from exact staged source')
 worker.predecessor(plan,args.stage,args.predecessor_report_sha256)
 return worker,plan


def run(args):
 require(args.stage in STAGES and args.physical_device in (4,7),'Require explicit deterministic stage on physical card4');locks(args.physical_device)
 worker,plan=source_check(args)
 import shutil
 require(shutil.disk_usage('/cache').free>=230*(1<<30),'Need230GiB free for both full optimizer checkpoint histories')
 imports=load_import_guard().cpu_preflight(plan,STAGE)
 if args.check_plan:
  print(json.dumps({'status':'plan_checked_CPU_only','stage':args.stage,'plan_sha256':args.plan_sha256,'physical_npu_selected_but_not_touched':args.physical_device,'Torch_imported':'torch' in sys.modules,'CANN_import_preflight':imports}),flush=True);return
 utils=module('h3_reviewed_owned_process_utils',STAGE/'run_h3_surrogate67_npu4_v1.py')
 OUTPUT.mkdir(parents=True,exist_ok=True);launches=OUTPUT/'launches';launches.mkdir(exist_ok=True)
 mutex=utils.acquire_file_lock(launches/(args.stage+'.lock'));handles=[];cancel=utils.Cancellation();runner=utils.ChildRunner(cancel);receipt=launches/(args.stage+'_attempt1.json')
 owned_receipt=False
 state={'stage':args.stage,'status':'admitting','parent_pid':os.getpid(),'physical_npu':args.physical_device,'plan_sha256':args.plan_sha256,'started_utc':utils.now(),'all_three_locks_held':False,'optimizer_updates':0,'CANN_import_preflight':imports}
 try:
  require(not receipt.exists() and not (OUTPUT/args.stage).exists(),'Stage already attempted; inspect preserved evidence, no rerun')
  with receipt.open('x') as f:json.dump(state,f,indent=2);f.write('\n')
  owned_receipt=True
  cancel.install()
  for path in locks(args.physical_device):handles.append(utils.acquire_file_lock(path))
  state['all_three_locks_held']=True;state['lock_paths']=[str(v) for v in locks(args.physical_device)]
  smipath=launches/(args.stage+'_attempt1_npu_smi.txt')
  with smipath.open('x') as f:
   code=runner.run(['npu-smi','info'],child_environment(args.physical_device),f,timeout=20)
  require(code==0,'npu-smi failed');state['device_idle']=require_idle(smipath.read_text(),args.physical_device);state['npu_smi']=utils.record(smipath)
  worker.predecessor(plan,args.stage,args.predecessor_report_sha256)
  state['command']=command(args,receipt);state['environment']={k:child_environment(args.physical_device)[k] for k in ('ASCEND_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','TORCH_DEVICE_BACKEND_AUTOLOAD','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONNOUSERSITE','PYTHONPATH')};state['status']='starting_child';utils.atomic_json(receipt,state)
  def started(process):
   state.update(status='running',child_pid=process.pid,child_pgid=process.pid);utils.atomic_json(receipt,state)
  log=launches/(args.stage+'_attempt1.log')
  with log.open('x') as f:code=runner.run(state['command'],child_environment(args.physical_device),f,on_started=started)
  state['cleanup']=runner.cleanup;state['exit_code']=code;state['log']=utils.record(log);require(code==0,'Owned probe stage returned nonzero')
  report=OUTPUT/args.stage/'report.json';d=json.loads(report.read_text());require(d['status']==('completed_shared_limiteddev_step0' if args.stage=='step0' else 'completed_limiteddev2000_awaiting_full226_guard') and d['optimizer_updates_this_attempt']==(0 if args.stage=='step0' else 2000) and d['actual_mode']=={'deterministic_algorithms':True,'warn_only':False} and d['plan_sha256']==args.plan_sha256 and d['physical_npu']==args.physical_device,'Stage did not complete its exact bounded updates')
  worker.verify_plan(args.plan,args.plan_sha256);require(sha(ENV_SOURCE)==ENV_SHA,'Runtime environment source changed')
  state.update(status='completed',optimizer_updates=d['optimizer_updates_this_attempt'],report=utils.record(report),finished_utc=utils.now());utils.atomic_json(receipt,state);print(json.dumps({'status':'completed','receipt':utils.record(receipt)}),flush=True)
 except BaseException as e:
  if owned_receipt:
   state.update(status='failed_no_retry',optimizer_updates='unknown_read_child_update_receipt',error_type=type(e).__name__,error=str(e),finished_utc=utils.now(),cleanup=runner.cleanup);utils.atomic_json(receipt,state)
  raise
 finally:
  # ChildRunner's finally has reaped its own PGID before returning/raising.
  if runner.active is not None:
   utils.terminate_owned(runner.active);utils.await_owned_quiescence(runner.active)
  for h in reversed(handles):h.close()
  mutex.close();cancel.restore_handlers()


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=STAGES);p.add_argument('--physical-device',type=int,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--predecessor-report-sha256');p.add_argument('--check-plan',action='store_true');run(p.parse_args())
if __name__=='__main__':main()
