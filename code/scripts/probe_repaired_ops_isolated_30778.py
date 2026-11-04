"""One fresh-process, one-card repaired-ops probe; no model or installation.

The caller must coordinate an idle physical 910B card. The worker is contained
by GNU timeout (TERM 100s, KILL after 5s); root must verify npu-smi afterwards.
--check-only validates paths/environment/source without importing Torch/NPU.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


WORKER = r'''
import gc,json,os,re,signal,time,traceback,warnings
config=json.loads(os.environ['SOL_H3_OPS_PROBE_CONFIG'])
report={
 'scope':'tiny repaired-ops primitives only; not pretrained H3/VAE inference',
 'physical_card':config['physical_card'],'logical_device':'npu:0',
 'single_visible_card':os.environ['ASCEND_RT_VISIBLE_DEVICES'],
 'results':[],'passed':False,'model_or_checkpoint_loaded':False,
 'resource_release_completed':False,'pid':os.getpid(),
 'selected_environment':{k:os.environ.get(k) for k in (
  'ASCEND_HOME_PATH','ASCEND_TOOLKIT_HOME','TOOLKIT_HOME','ASCEND_OPP_PATH',
  'ASCEND_AICPU_PATH','ASCEND_CUSTOM_OPP_PATH','LD_LIBRARY_PATH',
  'ASCEND_RT_VISIBLE_DEVICES','TORCH_DEVICE_BACKEND_AUTOLOAD')},
 'explicit_tensor_budget_bytes':64*1024**2,
 'tracked_peak_guard_bytes':1024**3,
}
started=time.monotonic();storage={};torch=None;initialized=False
phase='framework_import'
def save():
 report['worker_seconds']=round(time.monotonic()-started,3)
 temp=config['output']+'.worker.tmp'
 with open(temp,'w') as f:json.dump(report,f,indent=2);f.write('\n')
 os.replace(temp,config['output'])
def term(signum,frame):
 raise TimeoutError('External TERM deadline reached')
signal.signal(signal.SIGTERM,term)
def record(name,status,**details):
 entry={'name':name,'status':status,**details};report['results'].append(entry)
 save();print(json.dumps({'operator_progress':entry}),flush=True)
def step(name,fn):
 global phase
 phase=name
 begin=time.monotonic()
 try:
  with warnings.catch_warnings(record=True) as caught,torch.no_grad():
   warnings.simplefilter('always')
   value=fn()
   if initialized:torch.npu.synchronize(0)
  peak=torch.npu.max_memory_allocated(0) if initialized else None
  if peak is not None and peak>report['tracked_peak_guard_bytes']:
   raise MemoryError('Tracked allocation exceeded 1GiB guard')
  record(name,'passed',seconds=round(time.monotonic()-begin,4),
         output_device=str(value.device) if torch.is_tensor(value) else None,
         output_dtype=str(value.dtype) if torch.is_tensor(value) else None,
         output_shape=list(value.shape) if torch.is_tensor(value) else None,
         tracked_peak_bytes=peak,warnings=[str(w.message)[:1000] for w in caught])
  return True,value
 except TimeoutError:
  record(name,'failed',seconds=round(time.monotonic()-begin,4),
         exception_type='TimeoutError',error='External deadline reached')
  raise
 except Exception as error:
  text=str(error)
  codes=sorted(set(re.findall(r'(?<!\d)(?:\d{6}|ERR\d{5})(?!\d)',text)))
  record(name,'failed',seconds=round(time.monotonic()-begin,4),
         exception_type=type(error).__name__,error=text,
         observed_error_codes=codes,traceback=traceback.format_exc())
  return False,None
def skip(names,reason):
 for name in names:record(name,'not_run',reason=reason,
                        operator_support_conclusion='not_evaluated')
def stage_inputs(name,items):
 def copy_inputs():
  for key,value in items.items():storage[key]=value.to('npu:0')
  return storage[next(iter(items))]
 return step(name,copy_inputs)[0]
basic_names=['isfinite','isfinite_host_truth','cast_FP32_to_BF16','matmul_FP32','mean_FP32',
             'pow_FP32','conv2d_FP32']
advanced_names=['SDPA_inputs','SDPA_BF16_masked','Cholesky_inputs',
                'Cholesky_FP32','triangular_solve_FP32']
try:
 # Environment is selected by the parent before any framework import.
 import torch,torch_npu
 report['torch_version']=torch.__version__
 report['torch_npu_version']=torch_npu.__version__
 record('framework_import','passed',torch=report['torch_version'],torch_npu=report['torch_npu_version'])
 torch.set_num_threads(1)
 phase='single_card_initialization'
 if torch.npu.device_count()!=1:
  raise RuntimeError('Expected exactly one visible logical NPU; no set_device attempted')
 torch.npu.set_device(0);initialized=True
 report['device_name']=torch.npu.get_device_name(0)
 report['visible_device_count']=1
 if hasattr(torch.npu,'mem_get_info'):
  free,total=torch.npu.mem_get_info(0)
  report['device_free_bytes_before_inputs']=free
  report['device_total_bytes']=total
  if free<2*1024**3:raise RuntimeError('Less than 2GiB free on selected card; no input allocations attempted')
 record('single_card_initialization','passed',device=report['device_name'],visible_device_count=1,
        physical_card=config['physical_card'])
 cpu=torch.arange(64,dtype=torch.float32).reshape(8,8)/64
 ok=stage_inputs('basic_input_CPU_to_NPU',{'basic':cpu})
 if not ok:
  skip(basic_names+advanced_names,'Basic input transfer failed; no operator inference')
 else:
  def finite_op():
   storage['finite']=torch.isfinite(storage['basic'])
   return storage['finite']
  def finite_host_truth():
   if not bool(storage['finite'].cpu().all().item()):
    raise RuntimeError('Finite CPU input became nonfinite on NPU')
   return True
  operations=[
   ('isfinite',finite_op),
   ('isfinite_host_truth',finite_host_truth),
   ('cast_FP32_to_BF16',lambda:storage['basic'].to(torch.bfloat16)),
   ('matmul_FP32',lambda:storage['basic']@storage['basic'].T),
   ('mean_FP32',lambda:storage['basic'].mean()),
   ('pow_FP32',lambda:storage['basic'].pow(2)),
   ('conv2d_FP32',lambda:torch.nn.functional.conv2d(
      storage['basic'].reshape(1,1,8,8),storage['basic'][:3,:3].reshape(1,1,3,3),padding=1)),
  ]
  for index,(name,fn) in enumerate(operations):
   ok,value=step(name,fn)
   if not ok:
    skip([n for n,_ in operations[index+1:]]+advanced_names,
         'Basic operator '+name+' failed; later primitives not evaluated')
    break
  if ok:
   report['basic_primitives_passed']=True
   q=torch.arange(1*2*4*128,dtype=torch.float32).reshape(1,2,4,128).div(1024).to(torch.bfloat16)
   k=torch.arange(1*2*6*128,dtype=torch.float32).reshape(1,2,6,128).div(1536).to(torch.bfloat16)
   mask=torch.arange(6)[None,:]<=torch.arange(4)[:,None]
   if stage_inputs('SDPA_inputs',{'q':q,'k':k,'v':k.clone(),'mask':mask}):
    step('SDPA_BF16_masked',lambda:torch.nn.functional.scaled_dot_product_attention(
       storage['q'],storage['k'],storage['v'],attn_mask=storage['mask'],dropout_p=0.))
   else:skip(['SDPA_BF16_masked'],'SDPA input construction failed; support not evaluated')
   a=torch.arange(2*128*128,dtype=torch.float32).reshape(2,128,128)/32768
   identity=torch.eye(128,dtype=torch.float32).expand(2,-1,-1).contiguous()
   spd=a@a.transpose(-1,-2)+identity*128
   if stage_inputs('Cholesky_inputs',{'spd':spd,'identity':identity}):
    chol_ok,factor=step('Cholesky_FP32',lambda:torch.linalg.cholesky(storage['spd']))
    if chol_ok:
     storage['factor']=factor
     step('triangular_solve_FP32',lambda:torch.linalg.solve_triangular(
        storage['factor'],storage['identity'],upper=False))
    else:skip(['triangular_solve_FP32'],'Cholesky failed; triangular solve not evaluated')
   else:skip(['Cholesky_FP32','triangular_solve_FP32'],
             'Cholesky input construction failed; support not evaluated')
  report['passed']=bool(report['results']) and all(r['status']=='passed' for r in report['results'])
except BaseException as error:
 report['fatal_stage']=phase
 report['fatal_exception_type']=type(error).__name__
 report['fatal_error']=str(error)
 report['fatal_traceback']=traceback.format_exc()
 report['passed']=False
 recorded={r['name'] for r in report['results']}
 if phase not in recorded:
  record(phase,'failed',exception_type=type(error).__name__,error=str(error),
         observed_error_codes=sorted(set(re.findall(r'(?<!\d)(?:\d{6}|ERR\d{5})(?!\d)',str(error)))),
         traceback=report['fatal_traceback'])
 skip([n for n in ['basic_input_CPU_to_NPU']+basic_names+advanced_names if n not in recorded],
      'Probe stopped at '+phase+'; later primitives not evaluated')
finally:
 try:
  storage.clear()
  for name in ('cpu','value','factor','q','k','mask','a','identity','spd'):
   globals().pop(name,None)
  gc.collect()
  if torch is not None and initialized:
   # Cleanup only: no new tensors/model/operations or synchronization retries.
   torch.npu.empty_cache()
   report['tracked_allocated_bytes_after_release']=torch.npu.memory_allocated(0)
   report['tracked_reserved_bytes_after_release']=torch.npu.memory_reserved(0)
  report['resource_release_completed']=True
  report['resource_release_is_process_exit_confirmation']=False
 except BaseException as error:
  report['cleanup_exception_type']=type(error).__name__
  report['cleanup_error']=str(error)
 try:
  with open('/proc/self/maps') as stream:
   report['loaded_selected_cann_libraries']=sorted({
    line.split()[-1] for line in stream if any(name in line for name in (
       'libopapi','libascendcl','libnnopbase','libascend910b'))})
 except OSError as error:report['loaded_library_inspection_error']=str(error)
 save();print(json.dumps({'final_worker_report':report}),flush=True)
raise SystemExit(0 if report['passed'] else 1)
'''


def absolute_dir(value, label):
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError(label + " must resolve to a directory")
    return path


def prepare_environment(project, runtime, opp, api, physical_card):
    if not (runtime / "lib64").is_dir() or not (api / "lib64").is_dir():
        raise ValueError("runtime-root and api-root must contain resolved lib64 directories")
    api_files = [str(p.resolve()) for p in (api / "lib64").glob("libopapi*.so*")]
    if not api_files:
        raise ValueError("No matching libopapi library found under api-root/lib64")
    libdirs = [api / "lib64", runtime / "lib64", runtime / "lib64/plugin/opskernel",
               runtime / "lib64/plugin/nnengine", opp / "built-in/op_impl/ai_core/tbe/op_tiling"]
    # The project view explicitly references allowed Toolkit libraries. Remove
    # inherited CANN library roots so a missing new ops library cannot silently
    # fall back to the old A3 package. Driver/non-CANN roots remain available.
    old_cann = Path('/usr/local/Ascend/cann-8.5.2').resolve()
    old_opp = old_cann / 'opp'
    inherited = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    preserved = [p for p in inherited if p and not Path(p).resolve().is_relative_to(old_cann)]
    # Process-local library precedence; global Toolkit/driver files are unchanged.
    paths = list(dict.fromkeys([str(p) for p in libdirs if p.is_dir()] + [p for p in preserved if p]))
    env = dict(os.environ, ASCEND_HOME_PATH=str(api), ASCEND_TOOLKIT_HOME=str(runtime),
               TOOLKIT_HOME=str(runtime), ASCEND_OPP_PATH=str(opp), ASCEND_AICPU_PATH=str(runtime),
               LD_LIBRARY_PATH=":".join(paths), ASCEND_RT_VISIBLE_DEVICES=str(physical_card),
               TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    python_paths = [runtime / "python/site-packages", opp / "built-in/op_impl/ai_core/tbe"]
    env["PYTHONPATH"] = ":".join(dict.fromkeys(
        [str(p) for p in python_paths if p.is_dir()]
        + [p for p in os.environ.get("PYTHONPATH", "").split(":")
           if p and not Path(p).resolve().is_relative_to(old_opp)]))
    env.pop("ASCEND_CUSTOM_OPP_PATH", None)
    env.pop("ASCEND_VISIBLE_DEVICES", None)
    return env, api_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True,
                        help="Resolved complete matching project CANN root, containing lib64")
    parser.add_argument("--opp-root", type=Path, required=True)
    parser.add_argument("--api-root", type=Path, required=True,
                        help="Resolved matching CANN API home, containing lib64/libopapi*.so")
    parser.add_argument("--physical-card", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh JSON under project results")
    parser.add_argument("--allow-device-probe", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only and not args.allow_device_probe:
        raise SystemExit("Explicit coordinated one-card probe flag required; no Torch import performed")
    project = absolute_dir(args.project_root, "project-root")
    runtime = absolute_dir(args.runtime_root, "runtime-root")
    opp = absolute_dir(args.opp_root, "opp-root")
    api = absolute_dir(args.api_root, "api-root")
    if not 0 <= args.physical_card < 8:
        raise SystemExit("Physical card must be one of the inventoried 0..7 cards")
    output = args.output.absolute()
    parent = output.parent.resolve(strict=True)
    if (not parent.is_relative_to(project / "results") or output.exists() or output.is_symlink()
            or output.with_suffix(".log").exists()):
        raise SystemExit("A fresh output inside existing project results is required")
    interpreter = project / ".venv/bin/python"
    if not interpreter.is_file():
        raise SystemExit("Existing isolated interpreter absent")
    env, libraries = prepare_environment(project, runtime, opp, api, args.physical_card)
    report = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "project_root": str(project), "runtime_root": str(runtime), "opp_root": str(opp),
              "api_root": str(api), "api_libraries": libraries,
              "physical_card": args.physical_card, "logical_device": "npu:0",
              "installation_performed": False, "checkpoint_loaded": False,
              "environment_selected_before_framework_import": True,
              "check_only": args.check_only, "worker_started": False,
              "old_CANN_library_fallback_removed": True, "old_OPP_python_fallback_removed": True,
              "external_timeout": "GNU timeout: TERM at 100s; KILL 5s later",
              "root_must_verify_npu_smi_process_exit": True}
    if args.check_only:
        print(json.dumps(report, indent=2), flush=True)
        return 0
    timeout = shutil.which("timeout")
    if not timeout:
        raise SystemExit("GNU external timeout unavailable; no framework import performed")
    started = time.monotonic()
    preflight = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=5)
    if preflight.returncode or f"No running processes found in NPU {args.physical_card}" not in preflight.stdout:
        report.update(passed=False, error="Physical card idle process snapshot cannot be confirmed",
                      npu_smi_exit=preflight.returncode, npu_smi_snapshot=preflight.stdout)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return 2
    report["preflight_card_idle_snapshot"] = True
    config = {"physical_card": args.physical_card, "output": str(output)}
    env["SOL_H3_OPS_PROBE_CONFIG"] = json.dumps(config)
    command = [timeout, "--signal=TERM", "--kill-after=5s", "100s",
               str(interpreter), "-B", "-c", WORKER]
    report["worker_started"] = True
    # A separate process group allows bounded recovery even if GNU timeout itself
    # stalls. No backend worker can outlive this wrapper's fallback group kill.
    process = subprocess.Popen(command, env=env, cwd=project, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    def kill_group():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    def wrapper_term(signum, frame):
        kill_group()
        raise TimeoutError("Outer wrapper received TERM; worker group killed")
    previous_handler = signal.signal(signal.SIGTERM, wrapper_term)
    try:
        stdout, stderr = process.communicate(timeout=108)
        report["worker_exit_code"] = process.returncode
        report["worker_stdout"] = stdout
        report["worker_stderr"] = stderr
    except (subprocess.TimeoutExpired, TimeoutError) as error:
        kill_group()
        try:
            stdout, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", "Worker group pipes did not close after SIGKILL"
            if process.stdout:process.stdout.close()
            if process.stderr:process.stderr.close()
        report.update(worker_exit_code=process.returncode,
                      error="External timeout failed/interrupted: " + str(error),
                      worker_stdout=stdout, worker_stderr=stderr,
                      fallback_entire_worker_group_killed=True,
                      passed=False, resource_release_confirmed=False)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    if output.is_file():
        try:
            report["worker_report"] = json.loads(output.read_text())
        except (OSError, json.JSONDecodeError) as error:
            report["worker_report_read_error"] = str(error)
    report["wall_seconds"] = round(time.monotonic() - started, 3)
    report["passed"] = (report.get("worker_exit_code") == 0
                        and report.get("worker_report", {}).get("passed") is True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    log = output.with_suffix(".log")
    log.write_text(report.get("worker_stdout", "") + report.get("worker_stderr", ""))
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
