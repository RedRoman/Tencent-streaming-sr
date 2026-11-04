"""Authorized A-only real H3 DMD block; GNU phase group contains its worker.

Separate durable supervisor owns GNU phase PGID. This file never installs,
downloads, trains, bypasses shutdown, or starts B/C after A fails.
"""
import argparse
import atexit
from datetime import datetime, timezone, timedelta
import faulthandler
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import signal
import socket
import subprocess
import sys
import time
import traceback
import warnings

PROJECT=Path('/home/ma-user/modelarts/user-job-dir/sol_h3')
SOURCE=PROJECT/'software/actual-weight-CPU-a99e858f02f9'
AUTH='Sentinel_56c29a89fad48191850d55812c093d08'
GLOBAL_CUTOFF=datetime.fromisoformat('2026-10-08T18:19:22.199807+00:00')
PREVIOUS_A_SECONDS=3.6677750255912542
CARD=7
LIMIT=6*1024**3
RECEIPT=PROJECT/'results/h3_completed35_integrity_receipt_normalized_20261008.json'
RECEIPT_SHA='34d7782d74d16417ee5f6c7a476bab91682587d2761718743a01139cdcda3aee'
ENV_HELPER=PROJECT/'scripts/probe_repaired_ops_isolated_30778.py'
ENV_SHA='4b582c70e840f547231b7c5f0057c5baffecb26e099fe8e107137524b375a6ac'
FATAL=re.compile(r'Exception ignored in atexit callback|Fatal Python error|free\(\): invalid pointer|double free|corrupted (?:size|double)',re.I)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def identity(pid):
    raw=Path(f'/proc/{pid}/stat').read_text()
    return dict(pid=pid,pgid=os.getpgid(pid),sid=os.getsid(pid),starttime=raw[raw.rfind(')')+2:].split()[19])
def save(path,value):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)

def pinned_module(name,path,expected):
    if sha(path)!=expected:raise ValueError('frozen source mismatch: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module

def inspection():
    sys.path[:0]=[str(SOURCE),str(SOURCE/'scripts')]
    manifest=json.loads((SOURCE/'source_manifest.json').read_text())
    if hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=\
            'a99e858f02f94df38f27a144113989e4c5b8b55f0f898036d2cd7691f76266b6':
        raise ValueError('exact source manifest binding differs')
    for relative,digest in manifest.items():
        path=SOURCE/relative
        if path.is_symlink() or not path.resolve().is_relative_to(SOURCE) or sha(path)!=digest:
            raise ValueError('imported frozen source differs: '+relative)
    wrapper=pinned_module('sol_weighted_CPU_guards',SOURCE/'scripts/real_weight_block_cpu_strictload_v3.py',
        'f1c2e14dd7bbe4d125e363dfe06a78f66830e025cb77c925902dfef4c64e3f78')
    for relative in ('scripts/real_weight_smoke_inspect.py','scripts/real_weight_smoke_block.py'):
        if sha(SOURCE/relative)!=manifest[relative]:raise ValueError('frozen helper differs')
    wrapper.metadata_gate(PROJECT/'ckpts','dmd',Path(os.environ['SOL_WEIGHTED_A_METADATA']))
    info=wrapper.inspect_checkpoint(PROJECT/'ckpts',block_index=0,route='dmd',
        inventory=PROJECT/'download_state/bridge_manifest.json',verify_sha256=False)
    if not info['ready']:raise ValueError('selected headers unavailable')
    bound=wrapper.bind_cached_integrity(PROJECT/'ckpts',info,RECEIPT,RECEIPT_SHA,
                                      PROJECT/'download_state/bridge_manifest.json')
    return info,bound

def smi(timeout):
    r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=timeout)
    return dict(exit=r.returncode,stdout=r.stdout,stderr=r.stderr,
                card_idle=r.returncode==0 and f'No running processes found in NPU {CARD}' in r.stdout)

def card_observation(snapshot):
    if snapshot['exit']!=0:raise ValueError('npu-smi failed during memory observation')
    lines=snapshot['stdout'].splitlines();hbm=None;pids=[]
    for index,line in enumerate(lines):
        if re.match(r'^\|\s*7\s+910B',line):
            if 'OK' not in line:raise ValueError('selected card health not OK')
            match=re.search(r'(\d+)\s*/\s*(\d+)\s*\|\s*$',lines[index+1])
            if not match:raise ValueError('selected card HBM observation cannot be parsed')
            hbm=int(match.group(1))
        fields=line.replace('|',' ').split()
        if len(fields)>=5 and fields[:2]==['7','0'] and fields[2].isdigit():pids.append(int(fields[2]))
    if hbm is None:raise ValueError('selected910B HBM not observed')
    return dict(HBM_usage_raw_MB=hbm,process_pids=pids,
        guard_raw_MB_to_bytes_upper_bound_factor=1024**2,MB_unit_not_independently_certified_as_MiB=True)

def worker(root):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));faulthandler.enable(all_threads=True)
    cfg=json.loads((root/'worker_config.json').read_text())
    if cfg['authorization']!=AUTH or cfg['physical_card']!=CARD or os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!='7':
        raise ValueError('authorized single-card mapping required')
    if os.getpgid(0)!=cfg['phase_group_id'] or os.getppid()!=cfg['phase_process_pid']:raise ValueError('worker identity escaped supervised phase')
    alarm_remaining=signal.getitimer(signal.ITIMER_REAL)[0]
    if signal.getsignal(signal.SIGALRM)!=signal.SIG_DFL or signal.SIGALRM in signal.pthread_sigmask(signal.SIG_BLOCK,set()) or not 0<alarm_remaining<=99:
        raise ValueError('independent actual-worker hard timeout missing')
    started=time.monotonic();report=dict(passed=False,numeric_checks_passed=False,pid=os.getpid(),
        parent_pid=os.getppid(),authorization=AUTH,pgid=os.getpgid(0),physical_card=CARD,logical_device='npu:0',scope='real_DMD_block0_unit_probe_not_video',
        memory_limit_bytes=LIMIT,memory_is_observed_not_hard_allocator_cap=True,steps=[],memory=[],warnings=[])
    report['independent_alarm_remaining_at_entry']=alarm_remaining
    torch=None;holder=None;module=None;handles=[];checkpoint_count=0
    def store():
        report['worker_seconds']=time.monotonic()-started;save(root/'worker.json',report)
    def marker(name):
        print(json.dumps(dict(lifecycle=name,pid=os.getpid())),flush=True)
    atexit.register(lambda:marker('after_torch_npu_registered_atexit_hook'))
    def terminated(signum,frame):raise TimeoutError('authorized A worker TERM')
    signal.signal(signal.SIGTERM,terminated)
    def memory(phase):
        nonlocal checkpoint_count
        if torch is None or not torch.npu.is_initialized():return
        allocated=torch.npu.max_memory_allocated(0);reserved=torch.npu.max_memory_reserved(0)
        checkpoint_count+=1
        record=dict(phase=phase,peak_allocated_bytes=allocated,peak_reserved_bytes=reserved)
        if not report['memory'] or allocated>report['memory'][-1]['peak_allocated_bytes'] or reserved>report['memory'][-1]['peak_reserved_bytes']:
            report['memory'].append(record);store()
        if max(allocated,reserved)>LIMIT:raise MemoryError('observed allocated/reserved peak exceeded6GiB')
    try:
        info,bound=inspection();report['selected_cached_integrity']=bound;store()
        import torch as framework
        import torch_npu
        torch=framework
        atexit.register(lambda:marker('before_torch_npu_registered_atexit_hook'))
        torch.set_num_threads(1)
        if torch.npu.device_count()!=1:raise ValueError('expected one visible NPU')
        torch.npu.set_device(0);torch.npu.reset_peak_memory_stats(0)
        free,total=torch.npu.mem_get_info(0)
        if free<LIMIT+1024**3:raise MemoryError('selected card lacks7GiB free headroom')
        report.update(NPU_initialized=True,device_name=torch.npu.get_device_name(0),free_before_load=free,
            torch_version=torch.__version__,torch_npu_version=torch_npu.__version__)
        memory('after_init');store()
        from scripts.real_weight_smoke_block import construct_and_load,run_probe
        args=argparse.Namespace(snapshot_root=PROJECT/'ckpts',block_index=0,route='dmd',window_frames=1,
            condition_history_frames=1,device='npu:0',atol=.05,rtol=.01,artifact_dir=root/'artifacts')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            holder,loading=construct_and_load(args,info,torch);report['loading']=loading
            if any(p.is_meta or p.device.type!='npu' for p in holder.parameters()):
                raise ValueError('active native parameter escaped NPU')
            memory('after_strict_load')
            def hook(module,inputs,output):
                def tensors(x):
                    if torch.is_tensor(x):yield x
                    elif isinstance(x,(tuple,list)):
                        for y in x:yield from tensors(y)
                if any(t.device.type!='npu' for t in tensors(output)):
                    raise ValueError('actual module output unexpectedly on CPU')
                memory(type(module).__name__)
            for module in holder.modules():handles.append(module.register_forward_hook(hook))
            report['probe']=run_probe(holder,args,info,torch)
            torch.npu.synchronize(0);memory('after_probe')
            report['warnings']=[str(w.message) for w in caught]
        report['numeric_checks_passed']=report['probe']['passed']
        artifact=args.artifact_dir/'weighted_block_probe_hidden_not_demo.safetensors'
        with artifact.open('rb') as f:os.fsync(f.fileno())
        report['saved_actual_hidden_artifact']=dict(path=str(artifact),bytes=artifact.stat().st_size,
            sha256=sha(artifact),label='actual_weighted_block_hidden_from_deterministic_stimuli_not_C24_video')
        report['stimulus_and_checkpoint_provenance']=dict(native_revision=info['revision'],source_manifest_sha256=
            'a99e858f02f94df38f27a144113989e4c5b8b55f0f898036d2cd7691f76266b6',
            receipt_sha256=RECEIPT_SHA,hidden_shape=[1,8,5376],LR_shape=[1,24,3,2,2],
            text_rows=2,temb_native_times=[1.,.5],future_suffix_frame=2,
            deterministic_input_recipe='frozen real_weight_smoke_block.run_probe arange/remainder recipes',
            is_actual_generated_LR=False,trained_VSR_quality_claim=False)
        report['passed']=report['numeric_checks_passed'];store()
    except BaseException as error:
        report.update(passed=False,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        try:
            for handle in handles:handle.remove()
            handles.clear();module=None;holder=None;gc.collect()
            if torch is not None and torch.npu.is_initialized():
                torch.npu.synchronize(0);torch.npu.empty_cache()
                report.update(allocated_after_cleanup=torch.npu.memory_allocated(0),
                              reserved_after_cleanup=torch.npu.memory_reserved(0))
                report['cleanup_passed']=report['allocated_after_cleanup']==0 and report['reserved_after_cleanup']==0
            else:report['cleanup_passed']=True
        except BaseException as error:report.update(cleanup_passed=False,cleanup_error=str(error))
        report['memory_checkpoint_calls']=checkpoint_count;store();marker('weighted_A_final_report')
    raise SystemExit(0 if report['passed'] and report['cleanup_passed'] else 1)

def phase(root):
    started=time.monotonic();deadline=started+118
    report=dict(passed=False,started_utc=datetime.now(timezone.utc).isoformat(),physical_card=CARD,
                worker_cap_seconds=100,phase_cap_seconds=125,stop_B_C_on_any_failure=True)
    process=None
    def stop():
        if process is not None and process.poll() is None:
            # Popen owns actual worker; kill only owned descendants before reaping it.
            children=Path(f'/proc/{process.pid}/task/{process.pid}/children')
            if children.exists():
                for child in children.read_text().split():
                    try:
                        if os.getpgid(int(child))==os.getpgrp():os.kill(int(child),signal.SIGKILL)
                    except ProcessLookupError:pass
            process.kill()
            try:process.wait(timeout=1)
            except subprocess.TimeoutExpired:report['owned_worker_reap_failed']=True
    def terminated(signum,frame):stop();raise TimeoutError('A phase TERM')
    signal.signal(signal.SIGTERM,terminated)
    try:
        report['before']=smi(2)
        if not report['before']['card_idle']:raise RuntimeError('card7 not idle')
        baseline=card_observation(report['before']);report['baseline_card_memory']=baseline
        info,bound=inspection();report['selected_cached_integrity']=bound
        cfg=dict(authorization=AUTH,physical_card=CARD,phase_group_id=os.getpgrp(),phase_process_pid=os.getpid())
        save(root/'worker_config.json',cfg)
        worker_started=time.monotonic()
        def worker_child_setup():
            signal.pthread_sigmask(signal.SIG_SETMASK,old_mask-{signal.SIGTERM,signal.SIGALRM})
            signal.signal(signal.SIGALRM,signal.SIG_DFL)
            signal.alarm(99)  # independent exec-preserved hard timeout for actual worker
        with (root/'worker.stdout.txt').open('x') as out,(root/'worker.stderr.txt').open('x') as err:
            # Same phase group plus exec-preserved 99s alarm: independent actual-worker timer.
            old_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM})
            try:
                process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',str(root)],
                    stdout=out,stderr=err,start_new_session=False,
                    preexec_fn=worker_child_setup)
                report['worker_direct_pid']=process.pid;report['worker_direct_identity']=identity(process.pid)
                report['phase_group_id']=os.getpgrp();save(root/'phase.json',report)
            finally:signal.pthread_sigmask(signal.SIG_SETMASK,old_mask)
        if os.getpgid(process.pid)!=os.getpgrp():raise RuntimeError('child not in guarded phase group')
        report['worker_direct_pid']=process.pid;report['phase_group_id']=os.getpgrp();save(root/'phase.json',report)
        worker_deadline=min(deadline,worker_started+99.8)
        soft_deadline=worker_started+98;TERM_sent=False;next_observation=worker_started
        report['worker_soft_TERM_seconds']=98;report['worker_hard_KILL_seconds']=99.8
        report['worker_independent_alarm_seconds']=99
        report['returncode_is_actual_worker_not_timeout_guard']=True
        report['periodic_card_memory']=[]
        while process.poll() is None:
            if time.monotonic()>=worker_deadline:
                report['timeout']=True;stop();break
            if time.monotonic()>=soft_deadline and not TERM_sent:
                report['timeout']=True;process.terminate();TERM_sent=True
            if time.monotonic()>=next_observation and worker_deadline-time.monotonic()>3.2:
                snapshot=smi(min(3.,worker_deadline-time.monotonic()-.2));observation=card_observation(snapshot)
                current=json.loads((root/'worker.json').read_text()) if (root/'worker.json').is_file() else {}
                actual_pid=current.get('pid') if current.get('authorization')==AUTH and current.get('parent_pid')==os.getpid() and current.get('pid')==process.pid and current.get('pgid')==os.getpgrp() else None
                if actual_pid:report['worker_pid']=actual_pid
                if any(pid!=actual_pid for pid in observation['process_pids']):
                    raise RuntimeError('foreign process observed on selected physicalcard')
                delta=max(0,observation['HBM_usage_raw_MB']-baseline['HBM_usage_raw_MB'])*1024**2
                report['periodic_card_memory'].append(dict(seconds=time.monotonic()-worker_started,
                    observation=observation,baseline_delta_upper_bound_bytes=delta,raw_snapshot=snapshot))
                save(root/'phase.json',report)
                if delta>LIMIT:raise MemoryError('observed card HBM growth upper bound exceeded6GiB')
                next_observation=time.monotonic()+1.
            time.sleep(.05)
        report['worker_exitcode']=process.returncode
    except BaseException as error:report.update(error_type=type(error).__name__,error=str(error));stop()
    finally:
        stop()
        stdout=(root/'worker.stdout.txt').read_text(errors='replace') if (root/'worker.stdout.txt').exists() else ''
        stderr=(root/'worker.stderr.txt').read_text(errors='replace') if (root/'worker.stderr.txt').exists() else ''
        report.update(worker_stdout=stdout,worker_stderr=stderr,worker_exitcode=process.returncode if process else None,
                      owned_worker_gone=process is not None and process.poll() is not None)
        if (root/'worker.json').is_file():
            report['worker_report']=json.loads((root/'worker.json').read_text())
            w=report['worker_report']
            if w.get('authorization')==AUTH and w.get('parent_pid')==os.getpid() and w.get('pid')==report.get('worker_direct_pid'):
                report['worker_pid']=w.get('pid')
        classifier=pinned_module('weighted_fallback_classifier',Path(__file__).with_name('classify_cpu_fallback_logs.py'),
            '8dcae3980e09d5a0ef724e6573c80a9030fb6af177ba1ddc6b99c822b2b3a491')
        report['cpu_fallback']=classifier.classify_cpu_fallback_logs(stdout=stdout,stderr=stderr,
            warnings=report.get('worker_report',{}).get('warnings',[]))
        report['fatal_exit_diagnostic']=bool(FATAL.search(stderr))
        lifecycle=[]
        for line in stdout.splitlines():
            try:item=json.loads(line)
            except json.JSONDecodeError:continue
            if item.get('pid')==report.get('worker_pid') and item.get('lifecycle'):lifecycle.append(item['lifecycle'])
        report['exit_lifecycle_markers']=lifecycle
        marker_passed=lifecycle==['weighted_A_final_report','before_torch_npu_registered_atexit_hook','after_torch_npu_registered_atexit_hook']
        report['phase_wall_seconds']=time.monotonic()-started
        w=report.get('worker_report',{})
        report['passed']=bool(report.get('worker_exitcode')==0 and w.get('passed') and w.get('cleanup_passed')
            and not report['cpu_fallback']['cpu_fallback_observed'] and not report['fatal_exit_diagnostic']
            and not report.get('timeout') and not report.get('error') and report['owned_worker_gone'] and marker_passed)
        report['BC_status']='not_started_A_failed' if not report['passed'] else 'entry_conditions_pending'
        save(root/'phase.json',report)
    return 0 if report['passed'] else 1

def supervisor(root):
    if socket.gethostname()!='ma-job-969927e6-bc48-4b1d-b345-93c74c1ee24b-worker-0' or os.getuid()!=1000:
        raise ValueError('trusted identity required')
    began=float(os.environ['SOL_WEIGHTED_LAUNCH_MONOTONIC'])
    utc=datetime.fromisoformat(os.environ['SOL_WEIGHTED_LAUNCH_UTC'])
    ledger=dict(authorization=AUTH,approved_utc='2026-10-08T17:43:16+00:00',started_utc=utc.isoformat(),
        hard_global_cutoff_utc=GLOBAL_CUTOFF.isoformat(),A_hard_cutoff_utc=(utc+timedelta(seconds=125)).isoformat(),
        global_limit_seconds=1095,stageA_limit_seconds=135,retry_limit_seconds=125,previous_A_wall_seconds=PREVIOUS_A_SECONDS,physical_cards_used=[CARD],scope='A_retry_only_B_C_not_started',passed=False)
    claim=PROJECT/'results'/('.weighted1095.'+hashlib.sha256(AUTH.encode()).hexdigest()+'.retry1.claim.json')
    with os.fdopen(os.open(claim,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w') as stream:
        json.dump(ledger,stream);stream.flush();os.fsync(stream.fileno())
    save(root/'wall.json',ledger)
    helper=pinned_module('weighted_environment',ENV_HELPER,ENV_SHA)
    view=PROJECT/'software/cann-8.5.2-910b-project'
    env,libraries=helper.prepare_environment(PROJECT,view,view/'opp',view,CARD)
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SOL_WEIGHTED_A_METADATA=str(root/'H3_metadata_receipt.json'))
    process=None
    def stop_group():
        if process is None:return
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        if process.poll() is None:
            try:process.wait(timeout=min(1.,max(0.,122-(time.monotonic()-began))))
            except subprocess.TimeoutExpired:ledger['phase_guard_reap_failed']=True
    def terminated(signum,frame):stop_group();raise TimeoutError('global supervisor TERM')
    signal.signal(signal.SIGTERM,terminated)
    try:
        with (root/'phase.stdout.txt').open('x') as out,(root/'phase.stderr.txt').open('x') as err:
            guard_TERM=min(123-(time.monotonic()-began),(GLOBAL_CUTOFF-datetime.now(timezone.utc)).total_seconds()-2)
            if guard_TERM<8:raise TimeoutError('A prelude exhausted fixed deadline before device launch')
            old_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM})
            try:
                process=subprocess.Popen(['timeout','-k','2s',f'{guard_TERM:.6f}s',sys.executable,'-B',str(Path(__file__).resolve()),'--phase',str(root)],
                    env=env,stdout=out,stderr=err,start_new_session=True,
                    preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_SETMASK,old_mask-{signal.SIGTERM}))
                ledger['phase_guard_pid']=process.pid;ledger['phase_guard_identity']=identity(process.pid)
                ledger['phase_guard_TERM_seconds_from_spawn']=guard_TERM;save(root/'wall.json',ledger)
            finally:signal.pthread_sigmask(signal.SIG_SETMASK,old_mask)
        process.wait(timeout=max(.1,119-(time.monotonic()-began)))
        ledger['phase_exitcode']=process.returncode
    except BaseException as error:ledger.update(error_type=type(error).__name__,error=str(error));stop_group()
    finally:
        stop_group()
        try:
            remaining=122-(time.monotonic()-began)
            if remaining<=0:raise TimeoutError('no final idle check reserve remains')
            ledger['final_card']=smi(min(3,remaining))
        except BaseException as error:ledger['final_card']=dict(card_idle=False,error=str(error))
        phase_result=json.loads((root/'phase.json').read_text()) if (root/'phase.json').exists() else {'passed':False}
        ledger.update(phase_exitcode=process.returncode if process else None,phase_result=phase_result,
            complete_wall_seconds=time.monotonic()-began,cumulative_A_wall_seconds=PREVIOUS_A_SECONDS+time.monotonic()-began,api_libraries=libraries,
            BC_status='skipped_A_failed' if not phase_result.get('passed') else 'not_started_retry_A_only_full_loader_and_decoder_missing',
            passed=bool(process is not None and process.returncode==0 and phase_result.get('passed')
                and ledger['final_card']['card_idle'] and not ledger.get('error')),
            no_automatic_relaunch=True,authorization_consumed_for_A=True)
        ledger['budget_passed']=ledger['complete_wall_seconds']<=125 and ledger['cumulative_A_wall_seconds']<=135 and datetime.now(timezone.utc)<=GLOBAL_CUTOFF
        ledger['passed']=ledger['passed'] and ledger['budget_passed']
        ledger['A_passed']=ledger['passed'];ledger['full_A_B_C_pipeline_passed']=False
        save(root/'wall.json',ledger);save(claim,{**ledger,'phase':'complete'})
        print(json.dumps({'complete':True,'A_passed':ledger['passed'],'wall_seconds':ledger['complete_wall_seconds'],
            'A_worker_exit':phase_result.get('worker_exitcode'),'B_C':ledger['BC_status'],'card_idle':ledger['final_card']['card_idle']}),flush=True)
    return 0 if ledger['passed'] else 1

if __name__=='__main__':
    # Modes are positional strings beginning with '--'; dispatch explicitly.
    mode=sys.argv[1];root=Path(sys.argv[2]).resolve()
    if not root.is_relative_to(PROJECT/'results'):raise ValueError('project-owned results required')
    if mode=='--worker':worker(root)
    elif mode=='--phase':raise SystemExit(phase(root))
    elif mode=='--supervisor':raise SystemExit(supervisor(root))
    else:raise ValueError('explicit authorized mode required')
