"""One approved 1800s primitive/shutdown matrix; default source check only.

Root alone stages/executes this file under GNU timeout TERM1798/KILL2. Every
worker, debugger, exit, reap and idle check shares the coordinator's 1700s clock.
The older 300s module is a pinned source/function template; its execute entry is
never called. No checkpoint, training, installer, shutdown bypass or stack edit.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import time
import uuid

PROJECT = Path('/home/ma-user/modelarts/user-job-dir/sol_h3')
CARD = 7
AUTHORIZATION = 'Sentinel_6ed0232ab588819191b34034cd6ad902'
APPROVED_UTC = '2026-10-08T16:16:52Z'
SCOPE = 'sol_h3.primitive_shutdown_1800s.v1'
TOTAL, SHARED = 1800., 1700.
COORDINATOR_TERM, COORDINATOR_KILL = 1740, 2
MAIN_WAIT, MAIN_REAP, FINAL_IDLE = 1745., 2., 3.
TEMPLATE_SHA = 'e2aad73c6558ffafb5e6e51d74888e066372cbcae4440cad9fb5bb5adab141a2'
FACTORY_SHA = '5878c0ce06d3d09b0c44731bb021b4776561131755a904fcc62859b631eb0751'
FROZEN_SHA = 'af1c4e2326a976134399a059b440c7e42c12540269dd82abe74ba31034097a7a'
CLASSIFIER_SHA = '8dcae3980e09d5a0ef724e6573c80a9030fb6af177ba1ddc6b99c822b2b3a491'
GDB_ENABLED = False  # Parent-approved skip until inferior cleanup is validated.
MINIMAL = ('CPU_import_only', 'NPU_init_only', 'NPU_copy_release_no_empty_cache',
           'NPU_copy_release_empty_cache', 'NPU_cast_BF16', 'NPU_SDPA_BF16')
JOBS = [dict(id=name, kind='minimal', case=name, cap=cap) for name, cap in zip(MINIMAL, (45,45,45,45,60,60))] + [
    dict(id='smallest_failure_LD_DEBUG', kind='ld_debug', cap=60),
    dict(id='smallest_failure_gdb', kind='gdb', cap=90),
    dict(id='smallest_failure_fallthrough', kind='fallthrough', cap=60),
    dict(id='Conv2d_BF16_cold', kind='primitive', case='depthwise_Conv2d_BF16_C128_k5', cap=300),
    dict(id='Conv2d_BF16_warm', kind='primitive', case='depthwise_Conv2d_BF16_C128_k5', cap=120, after='Conv2d_BF16_cold'),
    dict(id='VAE_Conv3d_cold', kind='primitive', case='VAE_Conv3d_FP32master_FP16autocast', cap=300),
    dict(id='VAE_Conv3d_warm', kind='primitive', case='VAE_Conv3d_FP32master_FP16autocast', cap=120, after='VAE_Conv3d_cold')]
FATAL_LOG = re.compile(r'Exception ignored in atexit callback|Traceback \(most recent call last\):|'
                       r'free\(\): invalid pointer|double free|corrupted (?:size|double)|Fatal Python error|'
                       r'Program received signal SIG(?:ABRT|SEGV)|Assertion .* failed', re.I)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def load_template():
    path = Path(__file__).with_name('probe_primitive_diagnostics_30778.py')
    if sha(path) != TEMPLATE_SHA: raise ValueError('Audited 300s template SHA mismatch')
    if sha(path.with_name('classify_cpu_fallback_logs.py')) != CLASSIFIER_SHA:raise ValueError('Classifier SHA mismatch before import')
    spec = importlib.util.spec_from_file_location('sol_h3_source_template_only', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


BASE = load_template()  # Pure stdlib definitions; never BASE.main/BASE.coordinator.


def parse_maps_source():
    return """def maps(phase):
    with open('/proc/self/maps') as stream: raw=stream.read()
    report.setdefault('proc_maps_raw_by_phase',{})[phase]=raw
    parsed=[]
    for line in raw.splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)>=5:
            parsed.append(dict(zip(('address','perms','offset','device','inode','path'),
                fields+[None] if len(fields)==5 else fields)))
    report.setdefault('proc_maps_parsed_by_phase',{})[phase]=parsed
    report.setdefault('loaded_shared_objects_by_phase',{})[phase]=[
        item for item in parsed if item['path'] and '.so' in item['path']]
    save()
"""


def workers():
    factory = Path(__file__).with_name('prepare_shutdown_minimal_workers_30778.py')
    if sha(factory) != FACTORY_SHA: raise ValueError('Minimal factory SHA mismatch')
    tree = ast.parse(factory.read_text())
    minimal = next(n.value.value for n in tree.body if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=='WORKER' for t in n.targets))
    old_begin = minimal.index('def maps(phase):')
    old_end = minimal.index('\ndef record(',old_begin)
    save_source = """def save():
    temp=cfg['output']+'.tmp'
    with open(temp,'w') as stream: json.dump(report,stream);stream.write('\\n')
    os.replace(temp,cfg['output'])
"""
    minimal = minimal[:old_begin]+save_source+'\n'+parse_maps_source()+minimal[old_end:]
    minimal = BASE.replace_once(minimal, "    emit({'step':value})", "    save();emit({'step':value})")
    minimal = BASE.replace_once(minimal, 'try:\n    import torch,torch_npu',
        "try:\n    maps('before_framework_import')\n    import torch,torch_npu")
    minimal = BASE.replace_once(minimal, "    maps('after_import')", "    maps('after_import')\n"
        "    if cfg['case']=='CPU_import_only' and torch.npu.is_initialized():\n"
        "        raise RuntimeError('CPU-only import unexpectedly initialized NPU')")
    minimal = BASE.replace_once(minimal, "    emit({'shutdown_worker_final':report})", "    save()\n"
        "    emit({'shutdown_worker_final_summary':{'case':cfg['case'],"
        "'passed_before_interpreter_exit':report['passed_before_interpreter_exit'],"
        "'report_path':cfg['output']}})")
    primitive = BASE.build_worker(Path(__file__).with_name('probe_remaining_ops_budget_30778.py'))
    primitive = BASE.replace_once(primitive, 'import atexit,gc,json,math,os,re,signal,time,traceback,warnings',
        'import atexit,faulthandler,gc,json,math,os,re,resource,signal,time,traceback,warnings\n'
        'resource.setrlimit(resource.RLIMIT_CORE,(0,0));faulthandler.enable(all_threads=True)')
    primitive = BASE.replace_once(primitive, "def record(entry):", parse_maps_source()+"\ndef record(entry):")
    primitive = BASE.replace_once(primitive, " report['steps'].append(entry);save();print",
        " report['steps'].append(entry);maps(entry['name']);save();print")
    primitive = BASE.replace_once(primitive, " save();print(json.dumps({'worker_final':report}),flush=True)",
        " maps('after_cleanup');save();print(json.dumps({'worker_final_summary':"
        "{'case':cfg['case'],'passed':report['passed'],'report_path':cfg['output']}}),flush=True)")
    ast.parse(minimal); ast.parse(primitive)
    return {'minimal':minimal,'primitive':primitive}


def gdb_driver(config):
    # starti stops before Python/Torch. Persist inferior identity before continue.
    source = r'''
import gdb,json,os,time
cfg=json.loads(CONFIG_JSON)
state={'claim_token':cfg['claim_token'],'case_id':cfg['case_id'],'inferior_history':[]}
def save():
    temp=cfg['journal']+'.tmp'
    with open(temp,'w') as stream:json.dump(state,stream)
    os.replace(temp,cfg['journal'])
def identity():
    pid=gdb.selected_inferior().pid
    if pid:
        with open('/proc/%d/stat'%pid) as stream:raw=stream.read()
        start=raw[raw.rfind(')')+2:].split()[19]
        item={'pid':pid,'pgid':os.getpgid(pid),'starttime':start}
        if item not in state['inferior_history']:state['inferior_history'].append(item)
        state.update(inferior_pid=pid,inferior_pgid=item['pgid'],inferior_starttime=start);save()
def exited(event):
    state['exited_event']=True;state['event_exit_code']=getattr(event,'exit_code',None);save()
gdb.events.exited.connect(exited)
for command in ('set pagination off','set confirm off','set startup-with-shell off',
                'set disable-randomization off','set auto-load off','set debuginfod enabled off',
                'handle SIGABRT stop print pass'):
    try:gdb.execute(command)
    except gdb.error as error:
        if command=='set debuginfod enabled off':print('GDB disabled URL configuration:',str(error))
        else:
            state['configuration_error']=command+': '+str(error);save()
            raise RuntimeError('Required GDB setting failed before inferior launch')
try:
    gdb.execute('starti');identity()
    if not state.get('inferior_pid'):raise RuntimeError('No journaled inferior; refusing continue')
    gdb.execute('continue')
    for command in ('thread apply all bt 16','info sharedlibrary','info proc mappings','info program'):
        try:gdb.execute(command)
        except gdb.error as error:print('GDB diagnostic:',command,str(error))
    if gdb.selected_inferior().pid:
        identity();gdb.execute('handle SIGABRT nostop print pass');gdb.execute('continue')
except BaseException as error:
    state['diagnostic_error']=str(error);save();print('GDB driver error:',str(error))
finally:
    for name in ('_exitcode','_exitsignal'):
        try:
            value=gdb.parse_and_eval('$'+name)
            state[name]=int(value) if value.type.code != gdb.TYPE_CODE_VOID else None
        except BaseException as error:state[name+'_read_error']=str(error)
    save();print('GDB_OWNED_INFERIOR_RESULT',json.dumps(state))
'''
    return source.replace('CONFIG_JSON',repr(json.dumps(config)),1)


def idle(timeout):
    result=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=timeout)
    return {'exit':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
            'card_idle':result.returncode==0 and f'No running processes found in NPU {CARD}' in result.stdout}


def blocked_term(): return signal.SIGTERM in signal.pthread_sigmask(signal.SIG_BLOCK,set())


def decode(value): return value.decode(errors='replace') if isinstance(value,bytes) else value or ''


def evaluate(entry, job):
    report=entry.get('worker_report',{});steps=report.get('steps',[])
    if job['kind']=='primitive':entry['operator_checks_passed']=report.get('passed') is True
    else:
        required={'CPU_import_only':'framework_import','NPU_init_only':'NPU_init',
            'NPU_copy_release_no_empty_cache':'copy_roundtrip','NPU_copy_release_empty_cache':'copy_roundtrip',
            'NPU_cast_BF16':'cast_FP32_BF16','NPU_SDPA_BF16':'SDPA_BF16_mask128'}[job['resolved_case']]
        entry['operator_checks_passed']=any(s.get('name')==required and s.get('status')=='passed' for s in steps)
    initialized=any(step.get('name')=='NPU_init' and step.get('status')=='passed' for step in steps)
    if job['kind']=='primitive':
        entry['cleanup_passed']=(report.get('resource_cleanup_completed') is True and not report.get('cleanup_error')
            and report.get('tracked_allocated_after_cleanup')==0 and report.get('tracked_reserved_after_cleanup')==0)
    else:
        entry['cleanup_passed']=(report.get('main_cleanup_completed') is True and not report.get('cleanup_error')
            and (not initialized or report.get('tracked_allocated_after_cleanup')==0))
    entry['tracked_allocator_observation']={key:report.get(key) for key in
        ('tracked_allocated_after_cleanup','tracked_reserved_after_cleanup','empty_cache_called')}
    if job.get('resolved_case')=='CPU_import_only':
        entry['CPU_import_no_initialization']=any(s.get('name')=='framework_import' and s.get('npu_initialized') is False for s in steps)
        entry['operator_checks_passed'] &= entry['CPU_import_no_initialization']
    expected={'NPU_cast_BF16':'cast_FP32_BF16','NPU_SDPA_BF16':'SDPA_BF16_mask128'}
    if job.get('resolved_case') in expected:
        native=next((s for s in steps if s.get('name')==expected[job['resolved_case']]),{})
        entry['actual_output_device_dtype_passed']=(native.get('output_device')=='npu:0' and native.get('output_dtype')=='torch.bfloat16')
        entry['operator_checks_passed'] &= entry['actual_output_device_dtype_passed']
    warning_text=[w for step in steps for w in step.get('warnings',[]) if isinstance(w,str)]
    entry['cpu_fallback_classification']=BASE.classify_cpu_fallback_logs(stdout=entry['stdout'],stderr=entry['stderr'],warnings=warning_text)
    entry['fatal_exit_diagnostic']=bool(FATAL_LOG.search(entry['stderr']+'\n'+entry['stdout']))
    entry['process_exit_passed']=entry['exit']==0 and not entry.get('forced_group_kill') and not entry['fatal_exit_diagnostic']
    if job['kind']=='gdb':
        inferior=entry.get('inferior_journal',{})
        entry['process_exit_passed'] &= (inferior.get('exited_event') is True and inferior.get('_exitcode')==0
                                         and inferior.get('_exitsignal') is None and not inferior.get('diagnostic_error'))
    if job['kind']!='primitive':
        markers=[]
        for line in entry['stdout'].splitlines():
            try:value=json.loads(line)
            except json.JSONDecodeError:continue
            if 'shutdown_marker' in value:markers.append(value['shutdown_marker'])
        entry['exit_markers']=markers
        entry['process_exit_passed'] &= markers==['before_torch_npu_registered_atexit_hook','after_torch_npu_registered_atexit_hook']
    else:
        events=[];summaries=[]
        for line in entry['stdout'].splitlines():
            try:value=json.loads(line)
            except json.JSONDecodeError:continue
            if value.get('pid')==report.get('pid') and 'lifecycle' in value:events.append(value['lifecycle'])
            if 'worker_final_summary' in value:summaries.append(value['worker_final_summary'])
        entry['primitive_lifecycle_events']=events
        entry['primitive_final_report_observed']=(len(summaries)==1 and summaries[0].get('case')==job['resolved_case']
            and summaries[0].get('report_path')==entry.get('worker_report_path'))
        entry['process_exit_passed'] &= (events==['worker_final_report_emitted','python_atexit_callback']
            and entry['primitive_final_report_observed'])
    entry['status']='passed' if (entry['operator_checks_passed'] and entry['cleanup_passed'] and entry['process_exit_passed']
        and not entry['cpu_fallback_classification']['cpu_fallback_observed'] and not entry.get('kill_errors')
        and not entry.get('reap_errors')) else 'failed'
    entry['operator_support_conclusion']=('not_determined_timeout' if entry.get('forced_group_kill') or
        entry['exit'] in {124,137,-9,-15} else 'observed_cpu_fallback' if entry['cpu_fallback_classification']['cpu_fallback_observed']
        else 'NPU_numerical_checks_passed' if entry['operator_checks_passed'] else 'precondition_or_operator_check_failed')


def coordinator():
    if blocked_term():raise SystemExit('Inherited blocked TERM rejected before workers')
    cfg=json.loads(os.environ['SOL_H3_30MIN_COORDINATOR']);claim_path=Path(cfg['claim'])
    claim=json.loads(claim_path.read_text())
    if (claim.get('phase')!='claimed' or claim.get('claim_token')!=cfg['claim_token']
        or claim.get('authorization_reference')!=AUTHORIZATION or sha(Path(__file__))!=cfg['source_hashes']['runner']):
        raise SystemExit('1800s exclusive claim/source mismatch')
    sources=workers()
    if {name:hashlib.sha256(source.encode()).hexdigest() for name,source in sources.items()}!=cfg['worker_hashes']:
        raise SystemExit('Worker source hash mismatch')
    started=time.monotonic();deadline=started+SHARED;active={};owned=set();failure=None;smallest=None
    # Adopt a killed debugger's orphan inferior so it can be explicitly reaped.
    gdb_subreaper=False
    if GDB_ENABLED and cfg.get('gdb'):
        import ctypes
        gdb_subreaper=ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)==0
    report={'scope':SCOPE,'authorization_reference':AUTHORIZATION,'physical_card':CARD,'cases':[],
        'active_child_groups':[],'owned_groups_seen':[],'cleanup_events':[],'kill_errors':[],'reap_errors':[],'passed':False}
    progress=Path(cfg['progress']);results={}
    def save():
        report.update(shared_elapsed_seconds=time.monotonic()-started,remaining_shared_seconds=max(0.,deadline-time.monotonic()),
                      active_child_groups=list(active),owned_groups_seen=sorted(owned),shared_failure=failure)
        BASE.atomic_json(progress,report)
    def kill_owned(path):
        errors=[]
        # Debugger may create an inferior process group different from its own.
        for record in list(active.values()):
            journal=Path(record['inferior_journal']) if record.get('inferior_journal') else None
            if journal and journal.is_file():
                try:
                    data=json.loads(journal.read_text())
                    if data.get('claim_token')!=cfg['claim_token']:raise ValueError('Inferior claim mismatch')
                    for identity in data.get('inferior_history',[]):
                        pid=int(identity['pid']);stat=Path(f'/proc/{pid}/stat')
                        if not stat.exists():continue
                        raw=stat.read_text();start=raw[raw.rfind(')')+2:].split()[19]
                        if start!=identity['starttime']:raise ValueError('Inferior PID was reused; no kill')
                        pgid=os.getpgid(pid)
                        if pgid!=identity['pgid']:raise ValueError('Inferior group changed; no group kill')
                        if pgid in {pid,record['process'].pid}:owned.add(pgid);errors.extend(BASE.kill_groups([pgid]))
                        else:os.kill(pid,signal.SIGKILL)
                except (OSError,ValueError,json.JSONDecodeError) as error:errors.append({'cleanup_path':path,'inferior_error':str(error)})
        errors.extend(BASE.kill_groups(active))
        report['kill_errors'].extend(errors);report['cleanup_events'].append({'path':path,'owned_groups':list(active),'errors':errors})
        return errors
    def finish_inferior(record,entry):
        journal=Path(record['inferior_journal'])
        if not journal.is_file():return
        data=json.loads(journal.read_text());entry['inferior_journal']=data
        if data.get('claim_token')!=cfg['claim_token']:raise ValueError('Inferior token mismatch')
        identities=data.get('inferior_history',[])
        for identity in identities:
            pid=int(identity['pid']);stat=Path(f'/proc/{pid}/stat')
            if stat.exists():
                raw=stat.read_text()
                if raw[raw.rfind(')')+2:].split()[19]!=identity['starttime']:raise ValueError('Inferior PID reused')
                kill_owned('gdb_inferior_after_debugger_exit')
            until=min(deadline-3.,time.monotonic()+1.)
            while True:
                try:reaped,_=os.waitpid(pid,os.WNOHANG)
                except ChildProcessError:reaped=0
                if reaped or not stat.exists():break
                if time.monotonic()>=until:
                    item={'inferior_pid':pid,'error':'Owned inferior exit/reap not confirmed'}
                    report['reap_errors'].append(item);entry.setdefault('reap_errors',[]).append(item);break
                time.sleep(.02)
        entry['inferior_exit_confirmed']=not entry.get('reap_errors')
    def terminated(signum,frame):
        report['TERM_received']=True;kill_owned('TERM');save();raise TimeoutError('Coordinator TERM')
    signal.signal(signal.SIGTERM,terminated)
    save()
    try:
        for index,original in enumerate(JOBS):
            job=dict(original);entry={'id':job['id'],'kind':job['kind'],'requested_cap_seconds':job['cap']}
            if job['kind'] in {'ld_debug','gdb','fallthrough'}:
                if smallest is None:entry['status']='skipped_no_failing_baseline';report['cases'].append(entry);save();continue
                job['resolved_case']=smallest
                if job['kind']=='gdb' and not GDB_ENABLED:
                    entry.update(status='skipped_unvalidated_inferior_cleanup',reason='Use LD_DEBUG and faulthandler; no debugger is launched')
                    report['cases'].append(entry);save();continue
                if job['kind']=='gdb' and (not cfg.get('gdb') or not gdb_subreaper):
                    entry['status']='skipped_gdb_unavailable';report['cases'].append(entry);save();continue
            else:job['resolved_case']=job['case']
            if job.get('after') and not results.get(job['after'],{}).get('cold_completed'):
                entry['status']='skipped_cold_not_completed';report['cases'].append(entry);save();continue
            remaining=deadline-time.monotonic()
            if remaining<14:entry['status']='not_run_budget';report['cases'].append(entry);save();continue
            try:before=idle(min(3.,remaining-10.))
            except subprocess.TimeoutExpired:before={'card_idle':False,'error':'Pre-case idle check timeout'}
            if not before['card_idle']:
                failure='Card not confirmed idle';entry.update(status='not_run_card_not_idle',npu_smi_before=before)
                report['cases'].append(entry);save();break
            limit=min(float(job['cap']),deadline-time.monotonic()-8.)
            if limit<8:entry['status']='not_run_budget';report['cases'].append(entry);save();continue
            worker_path=Path(cfg['case_outputs'][index]);kind='primitive' if job['kind']=='primitive' else 'minimal'
            worker_config={'case':job['resolved_case'],'output':str(worker_path),'physical_card':CARD,
                'new_authorization_reference':AUTHORIZATION,'coordinator_claim_token':cfg['claim_token'],
                'normal_exit_mode':'fallthrough' if job['kind']=='fallthrough' else 'system_exit'}
            env=dict(os.environ)
            env['SOL_H3_DIAGNOSTIC_WORKER' if kind=='primitive' else 'SOL_H3_SHUTDOWN_CASE_CONFIG']=json.dumps(worker_config)
            inner=[cfg['python'],'-B','-c',sources[kind]];inferior_journal=None
            if job['kind']=='ld_debug':env['LD_DEBUG']='libs,versions'
            if job['kind']=='gdb':
                env['DEBUGINFOD_URLS']=''
                inferior_journal=str(worker_path.with_suffix('.inferior.json'))
                driver=worker_path.with_suffix('.gdb.commands')
                driver.write_text('python\n'+gdb_driver({'journal':inferior_journal,'claim_token':cfg['claim_token'],'case_id':job['id']})+'\nend\n')
                inner=[cfg['gdb'],'--batch','--nx','--return-child-result','-x',str(driver),'--args',*inner]
            command=[cfg['timeout'],'--signal=TERM','--kill-after=2s',f'{limit:.3f}s',*inner]
            began=time.monotonic();old_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM})
            def before_exec():
                signal.pthread_sigmask(signal.SIG_SETMASK,old_mask-{signal.SIGTERM});resource.setrlimit(resource.RLIMIT_CORE,(0,0))
            try:
                process=subprocess.Popen(command,env=env,cwd=PROJECT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                    text=True,start_new_session=True,preexec_fn=before_exec)
                active[process.pid]={'process':process,'inferior_journal':inferior_journal};owned.add(process.pid)
                entry.update(status='running',resolved_case=job['resolved_case'],group_id=process.pid,
                    effective_TERM_seconds=limit,npu_smi_before=before,inferior_journal_path=inferior_journal,
                    worker_report_path=str(worker_path))
                report['cases'].append(entry);save()
            finally:signal.pthread_sigmask(signal.SIG_SETMASK,old_mask)
            try:out,err=process.communicate(timeout=min(limit+2.5,max(.1,deadline-time.monotonic()-3.)))
            except subprocess.TimeoutExpired as error:
                entry['forced_group_kill']=True;entry['kill_errors']=kill_owned('case_timeout:'+job['id'])
                try:out,err=process.communicate(timeout=min(.5,max(0.,deadline-time.monotonic()-3.)))
                except subprocess.TimeoutExpired as reap_error:
                    out,err=decode(reap_error.output),decode(reap_error.stderr)
                    item={'group_id':process.pid,'error':str(reap_error)};entry['reap_errors']=[item];report['reap_errors'].append(item)
                    if process.stdout:process.stdout.close()
                    if process.stderr:process.stderr.close()
            entry.update(exit=process.returncode,worker_observed_seconds=time.monotonic()-began,stdout=out,stderr=err)
            if worker_path.is_file():
                try:entry['worker_report']=json.loads(worker_path.read_text())
                except (OSError,json.JSONDecodeError) as error:entry['worker_report_error']=str(error)
            if inferior_journal and Path(inferior_journal).is_file():entry['inferior_journal']=json.loads(Path(inferior_journal).read_text())
            if job['kind']=='gdb':finish_inferior(active[process.pid],entry)
            evaluate(entry,job)
            framework=next((s for s in entry.get('worker_report',{}).get('steps',[]) if s.get('name')=='framework_import'),{})
            if job['kind']=='gdb' and not framework:
                entry['status']='skipped_gdb_unavailable';entry['diagnostic_unavailable_reason']='Debugger did not reach framework worker'
            elif not framework or framework.get('status')!='passed':failure='Shared framework import precondition failed'
            elif job['resolved_case']=='CPU_import_only' and framework.get('npu_initialized') is not False:failure='CPU-only initialization control failed'
            elif job['kind']=='minimal' and job['resolved_case']!='CPU_import_only' and not any(
                s.get('name')=='NPU_init' and s.get('status')=='passed' for s in entry['worker_report']['steps']):failure='Shared NPU initialization precondition failed'
            elif job['kind']=='primitive' and entry.get('worker_report',{}).get('fatal_stage') in {'framework_import','one_card_setup'}:
                failure='Shared primitive framework/NPU setup precondition failed'
            if entry.get('reap_errors') or report['kill_errors']:failure='Owned cleanup/reap failed'
            if process.returncode is not None:active.pop(process.pid,None)
            else:failure='Owned worker process not reaped'
            remaining=deadline-time.monotonic()
            if remaining<=0:after={'card_idle':False,'error':'Shared deadline before post-case confirmation'}
            else:
                try:after=idle(min(3.,remaining))
                except subprocess.TimeoutExpired:after={'card_idle':False,'error':'Post-case idle timeout'}
            entry['npu_smi_after']=after
            if not after['card_idle']:failure='Device process exit not confirmed'
            if job['kind']=='minimal' and entry['status']=='failed' and smallest is None:smallest=job['resolved_case']
            operation=next((s for s in entry.get('worker_report',{}).get('steps',[]) if s.get('name')==job['resolved_case']),{})
            entry['cold_completed']=(job['id'].endswith('_cold') and entry['operator_checks_passed'] and entry['cleanup_passed']
                and operation.get('status')=='passed' and after['card_idle'] and process.returncode is not None
                and process.returncode not in {124,137,-9,-15} and not entry.get('forced_group_kill')
                and not entry['cpu_fallback_classification']['cpu_fallback_observed']
                and entry.get('worker_report',{}).get('final_report_emitted_monotonic') is not None)
            entry['warm_eligibility_scope']='Completed native numerical operator and resource cleanup, confirmed card idle; shutdown failure stays separate'
            results[job['id']]=entry;save()
            print(json.dumps({'case':job['id'],'status':entry['status'],'exit':entry['exit'],
                'operator_checks_passed':entry['operator_checks_passed'],'cleanup_passed':entry['cleanup_passed'],
                'card_idle':after['card_idle'],'remaining_seconds':max(0.,deadline-time.monotonic())}),flush=True)
            if failure is not None:break
    except BaseException as error:report.update(coordinator_error=str(error),coordinator_error_type=type(error).__name__)
    finally:
        kill_owned('coordinator_finally')
        for item in list(active.values()):
            process=item['process'];remaining=max(0.,deadline-time.monotonic())
            try:
                if remaining>0:process.wait(timeout=min(.3,remaining))
                elif process.poll() is None:raise subprocess.TimeoutExpired(process.args,0)
            except (OSError,subprocess.TimeoutExpired) as error:report['reap_errors'].append({'group_id':process.pid,'error':str(error)})
        active.clear()
        present={entry['id'] for entry in report['cases']}
        for job in JOBS:
            if job['id'] not in present:report['cases'].append({'id':job['id'],'status':'not_run_precondition' if failure else 'not_run_budget_or_cleanup'})
        permitted_skips={'skipped_no_failing_baseline','skipped_gdb_unavailable','skipped_unvalidated_inferior_cleanup'}
        report['shared_failure']=failure
        report['passed']=(len(report['cases'])==len(JOBS) and len({e['id'] for e in report['cases']})==len(JOBS)
            and all(e['status']=='passed' and e.get('npu_smi_after',{}).get('card_idle') is True or e['status'] in permitted_skips for e in report['cases'])
            and not failure and not report.get('TERM_received') and not report.get('coordinator_error_type')
            and not report['kill_errors'] and not report['reap_errors'])
        save();print(json.dumps({'coordinator_complete':True,'passed':report['passed'],'progress_path':str(progress)}),flush=True)
    return 0 if report['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true');mode.add_argument('--check-only',action='store_true')
    parser.add_argument('--coordinator',action='store_true',help=argparse.SUPPRESS)
    for name in ('project-root','runtime-root','opp-root','api-root','environment-helper','cpu-preflight','authorization','output'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--authorization-token');args=parser.parse_args()
    if args.coordinator:return coordinator()
    script=Path(__file__).resolve(strict=True);sources=workers()
    hashes={'runner':sha(script),'template_300s':TEMPLATE_SHA,'minimal_factory':FACTORY_SHA,
        'frozen_tinyops':FROZEN_SHA,'classifier':CLASSIFIER_SHA}
    worker_hashes={name:hashlib.sha256(source.encode()).hexdigest() for name,source in sources.items()}
    if sha(script.with_name('classify_cpu_fallback_logs.py'))!=CLASSIFIER_SHA:raise SystemExit('Classifier source mismatch')
    report={'scope':SCOPE,'check_only':not args.execute,'execution_performed':False,'physical_card':CARD,
        'authorization_reference':AUTHORIZATION,'approved_utc':APPROVED_UTC,'budget_seconds':TOTAL,'shared_seconds':SHARED,
        'required_caller_outer_timeout':{'TERM_seconds':1798,'KILL_after_seconds':2,'total_seconds':1800},
        'coordinator_GNU_TERM_seconds':COORDINATOR_TERM,'coordinator_GNU_KILL_seconds':COORDINATOR_KILL,
        'main_absolute_wait_seconds':MAIN_WAIT,'main_reap_reserve_seconds':MAIN_REAP,'final_idle_reserve_seconds':FINAL_IDLE,
        'device_envelope_planned_seconds':MAIN_WAIT+MAIN_REAP+FINAL_IDLE,'matrix':JOBS,
        'all_case_caps_including_KILL_seconds':sum(job['cap']+2 for job in JOBS),
        'source_hashes':hashes,'worker_hashes':worker_hashes,'300s_execute_called':False,'old120s_budget_reused':False,
        'metadata_tail_requires_outer_wall_guard':True,'full_maps_and_diagnostics_saved_to_files':True,
        'gdb_execution_enabled':GDB_ENABLED,'gdb_skip_reason':'inferior cleanup is not validated; LD_DEBUG/faulthandler retained',
        'normal_exit_preserved':True,'model_weights_loaded':False,'installation_performed':False}
    names=('project_root','runtime_root','opp_root','api_root','environment_helper','cpu_preflight','authorization','output')
    if not args.execute and any(getattr(args,name) is None for name in names):
        report['check_status']='source/matrix contract only; no device/claim/subprocess/authorization write'
        report['authorization_schema']={'schema':'sol_h3.primitive_1800s_authorization.v1','authorized':True,
            'authorization_token':AUTHORIZATION,'approved_utc':APPROVED_UTC,'scope':SCOPE,'project_root':str(PROJECT),
            'server_port':30778,'physical_card':CARD,'budget_seconds':1800,'shared_seconds':1700,'matrix':JOBS,
            'source_hashes':{**hashes,'environment_helper':'REQUIRED'},'worker_hashes':worker_hashes,'cpu_preflight_sha256':'REQUIRED',
            'caller_outer_timeout':report['required_caller_outer_timeout']}
        report['cpu_preflight_schema']={'schema':'sol_h3.primitive_1800s_cpu_preflight.v1','status':'passed','passed':True,
            'created_utc':'FRESH_TIMEZONE_AWARE_UTC','project_root':str(PROJECT),'physical_card':CARD,'torch_npu_initialized':False,
            'loader_preflight_passed':True,'legacy_import_passed':True,'runtime_roots':'EXACT_RESOLVED_ROOTS',
            'source_hashes':{**hashes,'environment_helper':'REQUIRED'},'worker_hashes':worker_hashes,
            'evidence':[{'path':'ACTUAL_PROJECT_RESULTS_CPU_EVIDENCE','sha256':'REQUIRED'}]}
        print(json.dumps(report,indent=2));return 0
    if any(getattr(args,name) is None for name in names):parser.error('Explicit execution paths required')
    if args.project_root.resolve(strict=True)!=PROJECT:raise SystemExit('Only verified 30778 project allowed')
    paths={name:getattr(args,name).resolve(strict=True) for name in names if name not in {'project_root','output'}}
    if any(not path.is_relative_to(PROJECT) for path in paths.values()):raise SystemExit('Paths escape project')
    if not paths['environment_helper'].is_relative_to(PROJECT/'scripts') or any(not paths[name].is_relative_to(PROJECT/'results') for name in ('authorization','cpu_preflight')):
        raise SystemExit('Helper/proof scope mismatch')
    output=args.output.absolute()
    if not output.parent.resolve(strict=True).is_relative_to(PROJECT/'results'):raise SystemExit('Output must stay in project/results')
    hashes['environment_helper']=sha(paths['environment_helper']);roots={name:str(paths[name]) for name in ('runtime_root','opp_root','api_root')}
    proof=json.loads(paths['cpu_preflight'].read_text())
    expected_proof={'schema':'sol_h3.primitive_1800s_cpu_preflight.v1','status':'passed','passed':True,
        'project_root':str(PROJECT),'physical_card':CARD,'torch_npu_initialized':False,'loader_preflight_passed':True,
        'legacy_import_passed':True,'runtime_roots':roots,'source_hashes':hashes,'worker_hashes':worker_hashes}
    if any(proof.get(k)!=v for k,v in expected_proof.items()) or not 0<=BASE.utc_age(proof)<=3600:raise SystemExit('Fresh CPU preflight/source/runtime mismatch')
    if not proof.get('evidence'):raise SystemExit('Actual CPU evidence references required')
    for item in proof['evidence']:
        path=Path(item['path']).resolve(strict=True)
        if not path.is_relative_to(PROJECT/'results') or sha(path)!=item['sha256']:raise SystemExit('CPU evidence hash/path mismatch')
    authorization=json.loads(paths['authorization'].read_text())
    expected={'schema':'sol_h3.primitive_1800s_authorization.v1','authorized':True,'authorization_token':AUTHORIZATION,
        'approved_utc':APPROVED_UTC,'scope':SCOPE,'project_root':str(PROJECT),'server_port':30778,'physical_card':CARD,
        'budget_seconds':1800,'shared_seconds':1700,'matrix':JOBS,'source_hashes':hashes,'worker_hashes':worker_hashes,
        'cpu_preflight_sha256':sha(paths['cpu_preflight']),'caller_outer_timeout':report['required_caller_outer_timeout']}
    if any(authorization.get(k)!=v for k,v in expected.items()):raise SystemExit('Approved Sentinel/source/matrix/budget mismatch')
    claim=PROJECT/('results/.primitive_1800s.'+hashlib.sha256(AUTHORIZATION.encode()).hexdigest()+'.claim.json')
    progress=output.with_name(output.stem+'.coordinator.json');case_outputs=[output.with_name(output.stem+f'.case{i+1}.json') for i in range(len(JOBS))]
    fresh=[output,output.with_suffix('.log'),progress,*case_outputs,*[p.with_suffix('.inferior.json') for p in case_outputs],*[p.with_suffix('.gdb.commands') for p in case_outputs]]
    if any(path.exists() or path.is_symlink() for path in fresh):raise SystemExit('Fresh project-owned output files required')
    report.update(claim_path=str(claim),claim_already_exists=claim.exists() or claim.is_symlink(),progress_path=str(progress),
        authorization_path=str(paths['authorization']),authorization_sha256=sha(paths['authorization']),cpu_preflight_path=str(paths['cpu_preflight']))
    if not args.execute:print(json.dumps(report,indent=2));return 0
    if args.authorization_token!=AUTHORIZATION or blocked_term():raise SystemExit('Exact approved Sentinel/unblocked TERM required')
    spec=importlib.util.spec_from_file_location('sol_h3_30min_environment',paths['environment_helper'])
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    env,libraries=helper.prepare_environment(PROJECT,paths['runtime_root'],paths['opp_root'],paths['api_root'],CARD)
    timeout=shutil.which('timeout');python=PROJECT/'.venv/bin/python'
    if not timeout or not python.is_file():raise SystemExit('Existing GNU timeout and project Python required')
    cfg={'claim':str(claim),'claim_token':uuid.uuid4().hex,'authorization_reference':AUTHORIZATION,'source_hashes':hashes,
        'worker_hashes':worker_hashes,'progress':str(progress),'case_outputs':[str(p) for p in case_outputs],
        'timeout':timeout,'python':str(python),'gdb':shutil.which('gdb')}
    claim_value={**cfg,'phase':'claimed','scope':SCOPE,'approved_utc':APPROVED_UTC,'budget_seconds':1800,
        'authorization_sha256':report['authorization_sha256'],'created_utc':datetime.now(timezone.utc).isoformat()}
    descriptor=os.open(claim,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
    with os.fdopen(descriptor,'w') as stream:json.dump(claim_value,stream);stream.write('\n')
    env['SOL_H3_30MIN_COORDINATOR']=json.dumps(cfg)
    command=[timeout,'--signal=TERM','--kill-after=2s','1740s',str(python),'-B',str(script),'--coordinator']
    began=time.monotonic();wait_deadline=began+MAIN_WAIT;finish_deadline=began+MAIN_WAIT+MAIN_REAP+FINAL_IDLE
    process=None;owned=set();cleanup_errors=[];out=err=''
    def stop_all():
        if process is None:return
        owned.add(process.pid);cleanup_errors.extend(BASE.kill_groups([process.pid]))
        if progress.is_file():
            try:
                snapshot=json.loads(progress.read_text());groups=snapshot.get('active_child_groups',[])
                owned.update(groups);cleanup_errors.extend(BASE.kill_groups(groups))
                # Inferior journals remain available after terminating coordinator.
                for path in [p.with_suffix('.inferior.json') for p in case_outputs]:
                    if not path.is_file():continue
                    data=json.loads(path.read_text())
                    if data.get('claim_token')!=cfg['claim_token']:continue
                    for identity in data.get('inferior_history',[]):
                        pid=int(identity['pid']);stat=Path(f'/proc/{pid}/stat')
                        if not stat.exists():continue
                        raw=stat.read_text()
                        if raw[raw.rfind(')')+2:].split()[19]!=identity['starttime']:continue
                        pgid=os.getpgid(pid)
                        if pgid==identity['pgid'] and pgid in {pid,*groups}:owned.add(pgid);cleanup_errors.extend(BASE.kill_groups([pgid]))
                        else:os.kill(pid,signal.SIGKILL)
            except (OSError,ValueError,json.JSONDecodeError) as error:cleanup_errors.append({'journal_error':str(error)})
    def terminated(signum,frame):stop_all();raise TimeoutError('30min outer main TERM; owned groups killed')
    old_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM});previous=signal.signal(signal.SIGTERM,terminated)
    try:
        try:
            process=subprocess.Popen(command,env=env,cwd=PROJECT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                start_new_session=True,preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_SETMASK,old_mask-{signal.SIGTERM}))
            owned.add(process.pid);claim_value['owned_coordinator_group']=process.pid;BASE.atomic_json(claim,claim_value)
        except BaseException:stop_all();raise
        finally:signal.pthread_sigmask(signal.SIG_SETMASK,old_mask)
        remaining=wait_deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Spawn exhausted absolute wait deadline')
        out,err=process.communicate(timeout=remaining)
    except BaseException as error:
        stop_all();report.update(outer_error=str(error),outer_error_type=type(error).__name__)
        if process is not None:
            try:out,err=process.communicate(timeout=min(MAIN_REAP,max(0.,finish_deadline-time.monotonic()-FINAL_IDLE)))
            except subprocess.TimeoutExpired as reap_error:
                out,err=decode(reap_error.output),decode(reap_error.stderr);report['outer_reap_error']=str(reap_error)
                if process.stdout:process.stdout.close()
                if process.stderr:process.stderr.close()
    finally:stop_all();signal.pthread_sigmask(signal.SIG_SETMASK,old_mask);signal.signal(signal.SIGTERM,previous)
    report.update(execution_performed=process is not None,outer_exit=process.returncode if process else None,
        coordinator_stdout=out,coordinator_stderr=err,owned_groups_seen=sorted(owned),cleanup_errors=cleanup_errors,api_libraries=libraries)
    if progress.is_file():
        try:report['coordinator_report']=json.loads(progress.read_text())
        except (OSError,json.JSONDecodeError) as error:report['coordinator_report_error']=str(error)
    remaining=min(FINAL_IDLE,max(0.,finish_deadline-time.monotonic()))
    if remaining<=0:report['final_npu_smi']={'card_idle':False,'error':'No final confirmation budget'}
    else:
        try:report['final_npu_smi']=idle(remaining)
        except subprocess.TimeoutExpired:report['final_npu_smi']={'card_idle':False,'error':'Final idle timeout'}
    report['observed_device_envelope_seconds']=time.monotonic()-began
    report['selected_card_processes_gone']=report['final_npu_smi']['card_idle']
    report['passed']=(process is not None and process.returncode==0 and not cleanup_errors and 'outer_error' not in report
        and report.get('coordinator_report',{}).get('passed') is True and report['selected_card_processes_gone'])
    claim_value.update(phase='complete',report=str(output),observed_device_envelope_seconds=report['observed_device_envelope_seconds'],
        selected_card_processes_gone=report['selected_card_processes_gone'],complete_wall_ledger_requires_root_outer_measurement=True,
        authorization_consumed=True,no_automatic_relaunch_or_300s_addition=True)
    try:BASE.atomic_json(output,report);output.with_suffix('.log').write_text(out+err);BASE.atomic_json(claim,claim_value)
    except OSError as error:stop_all();report.update(passed=False,record_write_error=str(error),owned_groups_seen=sorted(owned),cleanup_errors=cleanup_errors)
    print(json.dumps({'scope':SCOPE,'passed':report['passed'],'outer_exit':report['outer_exit'],'report_path':str(output),
        'observed_device_envelope_seconds':report['observed_device_envelope_seconds'],'card_idle':report['selected_card_processes_gone'],
        'record_write_error':report.get('record_write_error'),'owned_groups_seen':report['owned_groups_seen']}),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':sys.exit(main())
