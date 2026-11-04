"""Prepare/run one independently guarded remaining-budget primitive probe.

Root executes this after reviewing the code. One fixed physical card (7), four
fresh workers, no model/weights/install, 75s shared deadline, outer TERM78/KILL3.
An exclusive persistent claim prevents repeating this remaining authorization.
--check-only never claims the budget, imports Torch/NPU, or runs npu-smi.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid


CASES = ("SDPA_BF16_mask128", "Cholesky_triSolve_FP32_128",
         "depthwise_Conv2d_BF16_C128_k5", "VAE_Conv3d_FP32master_FP16autocast")
PRIOR_SPENT = 32.634635020047426
TOTAL_AUTHORIZED = 120.0
SHARED_SECONDS = 75.0
OUTER_TERM_SECONDS = 78
OUTER_KILL_SECONDS = 3
FALLBACK_PATTERN = re.compile(r"(?:fall[\s_-]*back.{0,160}cpu|cpu.{0,160}fall[\s_-]*back)", re.I | re.S)


WORKER = r'''
import gc,json,math,os,re,signal,time,traceback,warnings
cfg=json.loads(os.environ['SOL_H3_REMAINING_WORKER'])
start=time.monotonic();data={};cpu={};torch=None;initialized=False
report={'case':cfg['case'],'pid':os.getpid(),'physical_card':7,'logical_device':'npu:0',
        'steps':[],'passed':False,'native_verified':False,
        'native_evidence_scope':'NPU device/dtype, numerical reference, and no observed CPU fallback; no kernel profiling claim',
        'model_or_checkpoint_loaded':False,'explicit_tensors_below_64MiB':True,
        'resource_cleanup_completed':False,'device_process_exit_confirmation':False}
fallback=re.compile(r'(?:fall[\s_-]*back.{0,160}cpu|cpu.{0,160}fall[\s_-]*back)',re.I|re.S)
def error_codes(text):
 symbolic=re.findall(r'\b(?:ERR\d{5}|E[ZX]\d{4,6})\b',text)
 numeric=re.findall(r'(?:\berr(?:or)?[\s_-]*code|\bcode|\b(?:aclnn|acl|rt)\w*[\s_-]*(?:ret|result|status))\s*(?:is\s*)?[:=]?\s*(-?\d{4,9})\b',text,re.I)
 return sorted(set(symbolic+numeric))
def save():
 report['worker_seconds']=time.monotonic()-start
 temp=cfg['output']+'.tmp'
 with open(temp,'w') as stream:json.dump(report,stream,indent=2);stream.write('\n')
 os.replace(temp,cfg['output'])
def record(entry):
 report['steps'].append(entry);save();print(json.dumps({'step':entry}),flush=True)
def terminated(signum,frame):raise TimeoutError('Fresh worker TERM deadline')
signal.signal(signal.SIGTERM,terminated)
def step(name,fn,expected_dtype=None,reference=None,atol=0.,rtol=0.):
 begin=time.monotonic();caught=[]
 try:
  with warnings.catch_warnings(record=True) as caught,torch.no_grad():
   warnings.simplefilter('always')
   value=fn()
   if initialized:torch.npu.synchronize(0)
   metadata={}
   if expected_dtype is not None:
    if not torch.is_tensor(value) or value.device.type!='npu' or value.dtype!=expected_dtype:
     raise RuntimeError('Unexpected output device/dtype; native operator not verified')
    metadata={'output_device':str(value.device),'output_dtype':str(value.dtype),
              'output_shape':list(value.shape),'output_strides':list(value.stride())}
    if hasattr(torch_npu,'get_npu_format'):
     metadata['npu_format']=torch_npu.get_npu_format(value)
   if reference is not None:
    actual=value.detach().float().cpu()
    expected=reference.detach().float().cpu()
    finite=bool(torch.isfinite(actual).all().item())
    error=(actual-expected).abs()
    metadata.update(cpu_reference_allclose=finite and bool(torch.allclose(actual,expected,atol=atol,rtol=rtol)),
                    max_abs_error=float(error.max().item()),atol=atol,rtol=rtol)
    if not metadata['cpu_reference_allclose']:
     raise RuntimeError('NPU output differs from meaningful CPU reference')
  messages=[str(w.message) for w in caught]
  if any(fallback.search(message) for message in messages):
   raise RuntimeError('Automatic CPU fallback warning; native NPU support not verified')
  peak=torch.npu.max_memory_allocated(0) if initialized else None
  if peak is not None and peak>1024**3:raise MemoryError('Tracked allocation exceeded 1GiB')
  record({'name':name,'status':'passed','seconds':time.monotonic()-begin,
          'warnings':messages,'tracked_peak_bytes':peak,**metadata})
  return value
 except BaseException as error:
  record({'name':name,'status':'failed','seconds':time.monotonic()-begin,
          'exception_type':type(error).__name__,'error':str(error),
          'observed_error_codes':error_codes(str(error)),
          'warnings':[str(w.message) for w in caught],'traceback':traceback.format_exc()})
  raise
def copy_inputs():
 for name,value in cpu.items():data[name]=value.to('npu:0')
 return data[next(iter(data))]
phase='framework_import'
try:
 import torch,torch_npu
 torch.set_num_threads(1)
 record({'name':phase,'status':'passed','torch':torch.__version__,'torch_npu':torch_npu.__version__})
 phase='one_card_setup'
 if torch.npu.device_count()!=1:raise RuntimeError('Exactly one visible logical device required')
 torch.npu.set_device(0);initialized=True
 if hasattr(torch.npu,'mem_get_info'):
  free,total=torch.npu.mem_get_info(0)
  if free<2*1024**3:raise RuntimeError('Less than 2GiB free; no inputs allocated')
  report.update(device_free_bytes_before_inputs=free,device_total_bytes=total)
 record({'name':phase,'status':'passed','physical_card':7,'logical_device':'npu:0',
         'device_name':torch.npu.get_device_name(0)})
 phase='CPU_inputs_and_reference'
 case=cfg['case']
 if case=='SDPA_BF16_mask128':
  q=(torch.arange(1024,dtype=torch.float32).reshape(1,2,4,128)%97/97).to(torch.bfloat16)
  k=(torch.arange(1536,dtype=torch.float32).reshape(1,2,6,128)%89/89).to(torch.bfloat16)
  v=(torch.arange(1536,dtype=torch.float32).reshape(1,2,6,128)%79/79-.5).to(torch.bfloat16)
  mask=torch.arange(6)[None,:]<=torch.arange(4)[:,None]
  scores=(q.float()@k.float().transpose(-1,-2))/math.sqrt(128)
  reference=(torch.softmax(scores.masked_fill(~mask,float('-inf')),dim=-1)@v.float()).to(torch.bfloat16).float()
  cpu.update(q=q,k=k,v=v,mask=mask)
 elif case=='Cholesky_triSolve_FP32_128':
  x=(torch.arange(2*128*128,dtype=torch.float32).reshape(2,128,128)%113/113-.5)/8
  identity=torch.eye(128,dtype=torch.float32).expand(2,-1,-1).contiguous()
  a=x@x.transpose(-1,-2)+identity*4
  reference=torch.linalg.cholesky(a)
  solve_reference=torch.linalg.solve_triangular(reference,identity,upper=False)
  cpu.update(a=a,identity=identity)
 elif case=='depthwise_Conv2d_BF16_C128_k5':
  # The native KV spatial feature is a channels-last view with NCHW addresses.
  x=(torch.arange(6*6*128,dtype=torch.float32).reshape(1,6,6,128)%83/83-.5).permute(0,3,1,2).to(torch.bfloat16)
  w=(torch.arange(128*25,dtype=torch.float32).reshape(128,1,5,5)%67/67-.5).to(torch.bfloat16)
  reference=torch.nn.functional.conv2d(x.float(),w.float(),padding=2,groups=128)
  cpu.update(x=x,w=w)
 elif case=='VAE_Conv3d_FP32master_FP16autocast':
  x=torch.arange(24*2*2*2,dtype=torch.float32).reshape(1,24,2,2,2)/192-.5
  w=torch.arange(24*24,dtype=torch.float32).reshape(24,24,1,1,1)/576-.5
  bias=torch.arange(24,dtype=torch.float32)/100
  reference=torch.nn.functional.conv3d(x,w,bias)
  cpu.update(x=x,w=w,bias=bias)
 else:raise RuntimeError('Unapproved primitive case')
 record({'name':phase,'status':'passed','inputs':{
     n:{'shape':list(t.shape),'dtype':str(t.dtype),'device':str(t.device),'strides':list(t.stride())}
     for n,t in cpu.items()},'reference_shape':list(reference.shape),
     'reference_finite':bool(torch.isfinite(reference).all().item())})
 phase='input_CPU_to_NPU'
 step(phase,copy_inputs)
 if case=='SDPA_BF16_mask128':
  phase='SDPA_BF16_mask128'
  step(phase,lambda:torch.nn.functional.scaled_dot_product_attention(
      data['q'],data['k'],data['v'],attn_mask=data['mask'],dropout_p=0.,is_causal=False),
      torch.bfloat16,reference,atol=.004,rtol=.025)
 elif case=='Cholesky_triSolve_FP32_128':
  phase='Cholesky_FP32_128'
  with torch.autocast('npu',enabled=False):
   data['factor']=step(phase,lambda:torch.linalg.cholesky(data['a']),
                       torch.float32,reference,atol=2e-5,rtol=2e-4)
   phase='triangular_solve_FP32_128'
   step(phase,lambda:torch.linalg.solve_triangular(data['factor'],data['identity'],upper=False),
        torch.float32,solve_reference,atol=2e-5,rtol=2e-4)
 elif case=='depthwise_Conv2d_BF16_C128_k5':
  phase=case
  step(phase,lambda:torch.nn.functional.conv2d(data['x'],data['w'],padding=2,groups=128),
       torch.bfloat16,reference,atol=.03,rtol=.025)
 elif case=='VAE_Conv3d_FP32master_FP16autocast':
  phase=case
  with torch.autocast('npu',dtype=torch.float16):
   step(phase,lambda:torch.nn.functional.conv3d(data['x'],data['w'],data['bias']),
        torch.float16,reference,atol=.004,rtol=.006)
 report['passed']=True;report['native_verified']=True
except BaseException as error:
 report.update(fatal_stage=phase,exception_type=type(error).__name__,error=str(error),
               traceback=traceback.format_exc(),passed=False,native_verified=False)
 if not any(s['name']==phase and s['status']=='failed' for s in report['steps']):
  record({'name':phase,'status':'failed','exception_type':type(error).__name__,'error':str(error)})
 report['operator_support_conclusion']='not_evaluated' if phase in {
     'framework_import','one_card_setup','CPU_inputs_and_reference','input_CPU_to_NPU'} else 'case_failed_or_cpu_fallback'
finally:
 try:
  data.clear();cpu.clear()
  for name in ('x','a','identity','q','k','v','w','bias','reference','solve_reference','scores','mask'):
   globals().pop(name,None)
  gc.collect()
  if torch is not None and initialized:
   torch.npu.empty_cache()
   report['tracked_allocated_after_cleanup']=torch.npu.memory_allocated(0)
   report['tracked_reserved_after_cleanup']=torch.npu.memory_reserved(0)
  report['resource_cleanup_completed']=True
 except BaseException as error:report['cleanup_error']=str(error)
 try:
  with open('/proc/self/maps') as stream:
   report['loaded_cann_libraries']=sorted({line.split()[-1] for line in stream
       if any(s in line for s in ('libopapi','libnnopbase','libascendcl','libascend910b'))})
 except OSError as error:report['maps_error']=str(error)
 save();print(json.dumps({'worker_final':report}),flush=True)
raise SystemExit(0 if report['passed'] else 1)
'''


def atomic_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def kill_groups(group_ids):
    for group in list(group_ids):
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass


def snapshot_idle(timeout):
    result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=timeout)
    return {"exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
            "card7_idle": result.returncode == 0 and "No running processes found in NPU 7" in result.stdout}


def coordinator():
    cfg = json.loads(os.environ["SOL_H3_REMAINING_COORDINATOR"])
    output, claim = Path(cfg["output"]), Path(cfg["claim"])
    claimed = json.loads(claim.read_text())
    if claimed["token"] != cfg["token"] or claimed["phase"] != "claimed":
        raise SystemExit("Independent remaining-budget claim mismatch")
    started = time.monotonic()
    deadline = started + SHARED_SECONDS
    active = {}
    shared_precondition_failure = None
    report = {"cases": [], "active_child_groups": [], "shared_seconds": SHARED_SECONDS,
              "prior_spent_seconds": cfg["prior_spent"], "coordinator_started_utc": datetime.now(timezone.utc).isoformat(),
              "passed": False, "training_or_checkpoint_run": False, "physical_card": 7}
    def save():
        report["wall_seconds"] = time.monotonic() - started
        report["remaining_shared_seconds"] = max(0., deadline-time.monotonic())
        report["active_child_groups"] = list(active)
        atomic_json(output, report)
    def terminated(signum, frame):
        kill_groups(active)
        report["outer_TERM_received"] = True
        save()
        raise TimeoutError("Shared probe outer TERM; all active child groups killed")
    signal.signal(signal.SIGTERM, terminated)
    save()
    try:
        for index, case in enumerate(CASES):
            remaining = deadline - time.monotonic()
            if remaining < 12:
                report["cases"].append({"case": case, "status": "not_run_budget", "remaining": max(0., remaining)})
                save();continue
            try:
                before = snapshot_idle(min(2., remaining-8.))
            except subprocess.TimeoutExpired:
                before = {"card7_idle": False, "error": "Pre-case npu-smi timeout"}
            remaining = deadline-time.monotonic()
            if not before["card7_idle"]:
                report["cases"].append({"case": case, "status": "not_run_card_not_idle", "npu_smi_before": before})
                save();break
            limit = min(24., remaining - 4.)
            if limit < 8:
                report["cases"].append({"case": case, "status": "not_run_budget", "remaining": max(0., remaining)})
                save();continue
            case_output = Path(cfg["case_outputs"][index])
            env = dict(os.environ, SOL_H3_REMAINING_WORKER=json.dumps({"case": case, "output": str(case_output)}))
            command = [cfg["timeout_binary"], "--signal=TERM", "--kill-after=2s", f"{limit:.3f}s",
                       cfg["python"], "-B", "-c", WORKER]
            begin = time.monotonic()
            # An early outer TERM must not land between detached spawn and PID
            # registration. Unblocking delivers it only after the journal exists.
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
            try:
                process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, text=True, start_new_session=True, cwd=cfg["project"],
                                           # Coordinator is single-threaded; restore only
                                           # the original mask before timeout execs.
                                           preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, old_mask))
                active[process.pid] = process
                entry = {"case": case, "status": "running", "group_id": process.pid,
                         "TERM_limit_seconds": limit, "KILL_after_seconds": 2, "npu_smi_before": before}
                report["cases"].append(entry);save()
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            try:
                stdout, stderr = process.communicate(timeout=min(limit+2.5, max(.1,deadline-time.monotonic())))
            except subprocess.TimeoutExpired:
                kill_groups([process.pid])
                try:stdout, stderr = process.communicate(timeout=.5)
                except subprocess.TimeoutExpired:
                    stdout, stderr = "", "Pipes did not close after worker-group SIGKILL"
                    if process.stdout:process.stdout.close()
                    if process.stderr:process.stderr.close()
                entry["fallback_group_kill"] = True
            entry.update(exit=process.returncode, worker_wall_seconds=time.monotonic()-begin,
                         stdout=stdout, stderr=stderr)
            if case_output.is_file():
                try:entry["worker_report"] = json.loads(case_output.read_text())
                except (OSError,json.JSONDecodeError) as error:entry["worker_report_read_error"] = str(error)
            fallback = bool(FALLBACK_PATTERN.search(stdout+"\n"+stderr))
            entry["cpu_fallback_detected_in_captured_logs"] = fallback
            if fallback and "worker_report" in entry:
                entry["worker_report"]["native_verified"] = False
                entry["worker_report"]["native_evidence_rejected_by_coordinator_logs"] = True
            entry["status"] = ("passed" if process.returncode == 0 and
                               entry.get("worker_report",{}).get("native_verified") is True and not fallback else "failed")
            worker_report = entry.get("worker_report", {})
            if worker_report.get("fatal_stage") in {"framework_import", "one_card_setup"}:
                shared_precondition_failure = {
                    "failed_case": case, "fatal_stage": worker_report["fatal_stage"],
                    "exception_type": worker_report.get("exception_type"),
                    "error": worker_report.get("error"),
                }
                report["shared_precondition_failure"] = shared_precondition_failure
            save()
            active.pop(process.pid,None)
            post_remaining = deadline-time.monotonic()
            if post_remaining <= .1:
                entry["post_case_npu_smi"] = {"card7_idle": False, "error": "Budget exhausted before exit confirmation"}
                save();break
            try:after = snapshot_idle(min(2.,post_remaining))
            except subprocess.TimeoutExpired:after = {"card7_idle":False,"error":"Post-case npu-smi timeout"}
            entry["post_case_npu_smi"] = after
            save()
            if not after["card7_idle"]:break
            if shared_precondition_failure is not None:break
    except BaseException as error:
        report["coordinator_error"] = str(error)
        report["coordinator_exception_type"] = type(error).__name__
    finally:
        kill_groups(active)
        for process in list(active.values()):
            try:process.wait(timeout=.2)
            except subprocess.TimeoutExpired:pass
        active.clear()
        represented = {entry["case"] for entry in report["cases"]}
        for case in CASES:
            if case not in represented:
                entry = {"case":case,"status":"not_run_precondition" if shared_precondition_failure is not None
                         else "not_run_budget_or_prior_cleanup", "operator_support_conclusion":"not_evaluated"}
                if shared_precondition_failure is not None:
                    entry["shared_precondition_failure"] = shared_precondition_failure
                report["cases"].append(entry)
        report["passed"] = all(entry["status"] == "passed" for entry in report["cases"])
        save()
        print(json.dumps(report,indent=2),flush=True)
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--opp-root", type=Path)
    parser.add_argument("--api-root", type=Path)
    parser.add_argument("--environment-helper", type=Path)
    parser.add_argument("--prior-report", type=Path, help="Authoritative actualdeviceprobe_driver result JSON")
    parser.add_argument("--output", type=Path, help="Fresh JSON inside project/results")
    parser.add_argument("--allow-device-probe", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.coordinator:return coordinator()
    if not args.check_only and not args.allow_device_probe:
        raise SystemExit("Root-reviewed remaining-budget probe flag required; no device/framework import attempted")
    if any(getattr(args,name) is None for name in ("project_root","runtime_root","opp_root","api_root",
                                                  "environment_helper","prior_report","output")):
        parser.error("Explicit project/runtime/opp/api/helper/prior/output paths required")
    project = args.project_root.resolve(strict=True)
    if str(project)!="/home/ma-user/modelarts/user-job-dir/sol_h3":
        raise SystemExit("Scope restricted to approved 30778 project")
    prior_path = args.prior_report.resolve(strict=True)
    helper_path = args.environment_helper.resolve(strict=True)
    output = args.output.absolute()
    if (not prior_path.is_relative_to(project/"results") or not helper_path.is_relative_to(project/"scripts")
            or not output.parent.resolve(strict=True).is_relative_to(project/"results")):
        raise SystemExit("Scope paths escape approved project")
    prior = json.loads(prior_path.read_text())
    spent = float(prior["probe_wall_seconds"])
    if (not math.isfinite(spent) or abs(spent-PRIOR_SPENT)>1e-6 or prior.get("physical_card")!=7
            or prior.get("selected_card_processes_gone") is not True
            or prior.get("training_or_checkpoint_run") is not False):
        raise SystemExit("Authoritative first-device probe/budget/exit evidence mismatch")
    if spent+OUTER_TERM_SECONDS+OUTER_KILL_SECONDS>TOTAL_AUTHORIZED:
        raise SystemExit("Remaining outer hard ceiling exceeds cumulative 120s authorization")
    case_outputs = [output.with_name(output.stem+f".case{i+1}.json") for i in range(len(CASES))]
    if any(path.exists() or path.is_symlink() for path in [output,output.with_suffix('.log'),*case_outputs]):
        raise SystemExit("Fresh output and case paths required; no replacement")
    spec = importlib.util.spec_from_file_location("sol_h3_frozen_probe_env",helper_path)
    helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    env,libraries = helper.prepare_environment(project,args.runtime_root.resolve(strict=True),
                                               args.opp_root.resolve(strict=True),args.api_root.resolve(strict=True),7)
    timeout = shutil.which("timeout")
    python = project/".venv/bin/python"
    if not timeout or not python.is_file():raise SystemExit("GNU timeout/existing isolated Python required")
    claim = project/"results/.remaining_ops_device_budget_120s.claim.json"
    cfg = {"project":str(project),"prior_report":str(prior_path),"prior_spent":spent,
           "prior_sha256":hashlib.sha256(prior_path.read_bytes()).hexdigest(),
           "output":str(output),"case_outputs":[str(p) for p in case_outputs],
           "claim":str(claim),"token":uuid.uuid4().hex,"timeout_binary":timeout,"python":str(python)}
    report = {"scope":"Independent remaining authorized primitive budget; no model/weights/install",
              "physical_card":7,"authorized_cumulative_seconds":TOTAL_AUTHORIZED,
              "prior_spent_seconds":spent,"prior_report":str(prior_path),"prior_sha256":cfg["prior_sha256"],
              "shared_monotonic_seconds":SHARED_SECONDS,"outer_TERM_seconds":OUTER_TERM_SECONDS,
              "outer_KILL_after_seconds":OUTER_KILL_SECONDS,"planned_cumulative_GNU_hard_seconds":spent+81,
              "maximum_cumulative_fallback_kill_and_reap_seconds":spent+83,
              "claim":str(claim),"environment_helper":str(helper_path),"api_libraries":libraries,
              "check_only":args.check_only,"models_or_checkpoints_loaded":False,"installation_performed":False}
    if args.check_only:
        report["claim_already_exists"] = claim.exists() or claim.is_symlink()
        print(json.dumps(report,indent=2));return 0
    descriptor = os.open(claim,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
    claim_value = {**cfg,"phase":"claimed","authorized_total_seconds":TOTAL_AUTHORIZED,
                   "reserved_outer_seconds":81,"created_utc":datetime.now(timezone.utc).isoformat()}
    with os.fdopen(descriptor,'w') as stream:json.dump(claim_value,stream,indent=2);stream.write('\n')
    env["SOL_H3_REMAINING_COORDINATOR"] = json.dumps(cfg)
    script = Path(__file__).resolve(strict=True)
    command = [timeout,"--signal=TERM","--kill-after=3s","78s",str(python),"-B",str(script),"--coordinator"]
    started = time.monotonic()
    process = subprocess.Popen(command,env=env,cwd=project,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                               text=True,start_new_session=True)
    def stop_all():
        # Stop the coordinator first; it cannot spawn a new detached group while
        # the outer process reads the last PID checkpoint and kills those groups.
        kill_groups([process.pid])
        groups=[]
        if output.is_file():
            try:groups.extend(json.loads(output.read_text()).get("active_child_groups",[]))
            except (OSError,json.JSONDecodeError):pass
        kill_groups(groups)
    def terminated(signum,frame):stop_all();raise TimeoutError("Outer main TERM; all active worker groups killed")
    previous = signal.signal(signal.SIGTERM,terminated)
    try:
        stdout,stderr = process.communicate(timeout=82)
    except (subprocess.TimeoutExpired,TimeoutError) as error:
        stop_all()
        try:stdout,stderr=process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            stdout,stderr="","Output pipes failed to close after all-group SIGKILL"
            if process.stdout:process.stdout.close()
            if process.stderr:process.stderr.close()
        report["outer_fallback_error"] = str(error)
    finally:signal.signal(signal.SIGTERM,previous)
    report.update(outer_exit=process.returncode,probe_wall_seconds=time.monotonic()-started,
                  coordinator_stdout=stdout,coordinator_stderr=stderr)
    report["cumulative_accounted_wall_seconds"] = spent+report["probe_wall_seconds"]
    if output.is_file():
        try:report["coordinator_report"] = json.loads(output.read_text())
        except (OSError,json.JSONDecodeError) as error:report["coordinator_report_error"] = str(error)
    # Independent post-exit evidence; if unavailable root must inspect npu-smi.
    try:report["final_npu_smi"] = snapshot_idle(2)
    except subprocess.TimeoutExpired:report["final_npu_smi"] = {"card7_idle":False,"error":"Final npu-smi timeout"}
    report["selected_card_processes_gone"] = report["final_npu_smi"]["card7_idle"]
    report["passed"] = (process.returncode==0 and report.get("coordinator_report",{}).get("passed") is True
                        and report["selected_card_processes_gone"] and report["cumulative_accounted_wall_seconds"]<=120)
    atomic_json(output,report)
    output.with_suffix('.log').write_text(stdout+stderr)
    claim_value.update(phase="complete",report=str(output),actual_new_wall_seconds=report["probe_wall_seconds"],
                       cumulative_accounted_wall_seconds=report["cumulative_accounted_wall_seconds"],
                       selected_card_processes_gone=report["selected_card_processes_gone"])
    atomic_json(claim,claim_value)
    print(json.dumps(report,indent=2),flush=True)
    return 0 if report["passed"] else 1


if __name__=="__main__":sys.exit(main())
