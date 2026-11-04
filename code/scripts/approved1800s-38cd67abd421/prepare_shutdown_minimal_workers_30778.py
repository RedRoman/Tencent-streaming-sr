"""Prepare (never execute) small, normal-exit TorchNPU shutdown diagnostics.

The parent must put WORKER in its separately authorized, pinned-source,
exclusive-claim, cumulative-deadline coordinator. This file has no execute mode.
"""
import ast
import hashlib
import json

CASES = (
    'CPU_import_only', 'NPU_init_only', 'NPU_copy_release_no_empty_cache',
    'NPU_copy_release_empty_cache', 'NPU_cast_BF16', 'NPU_SDPA_BF16',
)

WORKER = r'''
import atexit,faulthandler,gc,json,os,resource,signal,sys,time,traceback

cfg=json.loads(os.environ['SOL_H3_SHUTDOWN_CASE_CONFIG'])
cases=('CPU_import_only','NPU_init_only','NPU_copy_release_no_empty_cache',
       'NPU_copy_release_empty_cache','NPU_cast_BF16','NPU_SDPA_BF16')
project='/home/ma-user/modelarts/user-job-dir/sol_h3'
view=project+'/software/cann-8.5.2-910b-project'
if (cfg.get('case') not in cases or cfg.get('physical_card') not in range(8) or
    cfg.get('new_authorization_reference') in (None,'','pending')):
    raise SystemExit('New approved coordinator authorization reference required before any Torch import')
if cfg.get('coordinator_claim_token') in (None,'','pending'):
    raise SystemExit('New unique coordinator claim token required before any Torch import')
if (os.environ.get('ASCEND_HOME_PATH') != view or
    os.environ.get('ASCEND_OPP_PATH') != view+'/opp' or
    os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != str(cfg.get('physical_card')) or
    os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD') != '0'):
    raise SystemExit('Expected original project-only process environment before any Torch import')
if cfg.get('normal_exit_mode','system_exit') not in ('system_exit','fallthrough'):
    raise SystemExit('Only ordinary Python exits are allowed')
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
faulthandler.enable(all_threads=True)
started=time.monotonic()
report={'case':cfg['case'],'pid':os.getpid(),'physical_card':cfg['physical_card'],
        'new_authorization_reference':cfg['new_authorization_reference'],
        'coordinator_claim_token':cfg['coordinator_claim_token'],
        'normal_exit_mode':cfg.get('normal_exit_mode','system_exit'),
        'models_or_checkpoints_loaded':False,'steps':[],'passed_before_interpreter_exit':False,
        'interpreter_exit_observed_by_worker':False,'loaded_shared_objects_by_phase':{}}
torch=None
initialized=False
data={}

def emit(value):
    print(json.dumps(value),flush=True)

def marker(name):
    # No framework/device calls in either atexit marker.
    emit({'shutdown_marker':name,'pid':os.getpid(),
          'monotonic_elapsed':time.monotonic()-started})

# Registered before explicit TorchNPU import: LIFO puts this after its hook.
atexit.register(marker,'after_torch_npu_registered_atexit_hook')

def maps(phase):
    with open('/proc/self/maps') as stream:
        report['loaded_shared_objects_by_phase'][phase]=sorted({
            line.split()[-1] for line in stream
            if len(line.split())>=6 and '.so' in line.split()[-1]})

def record(name,**details):
    value={'name':name,**details}
    report['steps'].append(value)
    emit({'step':value})

def timed_out(signum,frame):
    raise TimeoutError('Owned worker TERM deadline')

signal.signal(signal.SIGTERM,timed_out)

def allocate_copy():
    cpu=torch.arange(64,dtype=torch.float32).reshape(8,8)/64
    data['basic']=cpu.to('npu:0')
    torch.npu.synchronize(0)
    copied=data['basic'].cpu()
    if not torch.equal(cpu,copied):
        raise RuntimeError('Copy-only roundtrip differs')
    record('copy_roundtrip',status='passed',output_device=str(data['basic'].device),
           output_dtype=str(data['basic'].dtype),output_shape=list(data['basic'].shape))

def cast_bf16():
    data['cast']=data['basic'].to(torch.bfloat16)
    torch.npu.synchronize(0)
    actual=data['cast'].cpu()
    expected=(torch.arange(64,dtype=torch.float32).reshape(8,8)/64).to(torch.bfloat16)
    if not torch.equal(actual,expected):
        raise RuntimeError('BF16 cast differs from CPU reference')
    record('cast_FP32_BF16',status='passed',output_device=str(data['cast'].device),
           output_dtype=str(data['cast'].dtype))

def sdpa():
    q=torch.arange(1024,dtype=torch.float32).reshape(1,2,4,128).div(1024).to(torch.bfloat16)
    k=torch.arange(1536,dtype=torch.float32).reshape(1,2,6,128).div(1536).to(torch.bfloat16)
    v=k.clone()
    mask=torch.arange(6)[None,:]<=torch.arange(4)[:,None]
    reference=torch.nn.functional.scaled_dot_product_attention(q.float(),k.float(),v.float(),
                 attn_mask=mask,dropout_p=0.)
    for name,value in (('q',q),('k',k),('v',v),('mask',mask)):
        data[name]=value.to('npu:0')
    torch.npu.synchronize(0)
    data['sdpa']=torch.nn.functional.scaled_dot_product_attention(data['q'],data['k'],data['v'],
                  attn_mask=data['mask'],dropout_p=0.)
    torch.npu.synchronize(0)
    actual=data['sdpa'].cpu().float()
    expected=reference.to(torch.bfloat16).float()
    error=float((actual-expected).abs().max().item())
    matches=bool(torch.allclose(actual,expected,atol=.004,rtol=.025))
    if not matches:
        raise RuntimeError('SDPA differs from CPU reference; max_abs_error='+str(error))
    record('SDPA_BF16_mask128',status='passed',output_device=str(data['sdpa'].device),
           output_dtype=str(data['sdpa'].dtype),output_shape=list(data['sdpa'].shape),
           cpu_reference_allclose=matches,max_abs_error=error)

try:
    import torch,torch_npu
    # Registered after import: LIFO puts this before TorchNPU's registered hook.
    atexit.register(marker,'before_torch_npu_registered_atexit_hook')
    record('framework_import',status='passed',torch=torch.__version__,torch_npu=torch_npu.__version__,
           npu_initialized=bool(torch.npu.is_initialized()))
    maps('after_import')
    torch.set_num_threads(1)
    if cfg['case']!='CPU_import_only':
        if torch.npu.device_count()!=1:
            raise RuntimeError('Expected only one visible logical NPU')
        torch.npu.set_device(0)
        initialized=True
        free,total=torch.npu.mem_get_info(0)
        if free<2*1024**3:
            raise RuntimeError('Selected card has less than 2GiB free; no tensors allocated')
        record('NPU_init',status='passed',logical_device='npu:0',
               device_name=torch.npu.get_device_name(0),free_bytes=free,total_bytes=total)
        maps('after_init')
        if cfg['case']!='NPU_init_only':
            allocate_copy()
            if cfg['case']=='NPU_cast_BF16':
                cast_bf16()
            elif cfg['case']=='NPU_SDPA_BF16':
                sdpa()
            peak=torch.npu.max_memory_allocated(0)
            if peak>1024**3:
                raise MemoryError('Tracked memory exceeded 1GiB guard')
            report['tracked_peak_bytes']=peak
            maps('after_operator')
    report['passed_before_interpreter_exit']=True
except BaseException as error:
    report.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    report['passed_before_interpreter_exit']=False
finally:
    try:
        data.clear()
        gc.collect()
        if initialized:
            # Separate no-empty-cache case isolates this explicit cleanup call.
            if cfg['case'] in ('NPU_copy_release_empty_cache','NPU_cast_BF16','NPU_SDPA_BF16'):
                torch.npu.empty_cache()
                report['empty_cache_called']=True
            else:
                report['empty_cache_called']=False
            report['tracked_allocated_after_cleanup']=torch.npu.memory_allocated(0)
            report['tracked_reserved_after_cleanup']=torch.npu.memory_reserved(0)
            if report['tracked_allocated_after_cleanup']!=0:
                raise RuntimeError('Tracked tensor allocation remains at end of main')
        maps('after_cleanup')
        report['main_cleanup_completed']=True
    except BaseException as error:
        report.update(main_cleanup_completed=False,cleanup_error_type=type(error).__name__,
                      cleanup_error=str(error),passed_before_interpreter_exit=False)
    report['worker_seconds_before_atexit']=time.monotonic()-started
    emit({'shutdown_worker_final':report})

# Neither manual _npu_shutdown nor atexit unregister nor os._exit is used.
if not report['passed_before_interpreter_exit']:
    raise SystemExit(1)
if cfg.get('normal_exit_mode','system_exit')=='system_exit':
    raise SystemExit(0)
# Optional ordinary EOF/fallthrough counterpart: interpreter runs its hooks.
'''


def preparation_report():
    ast.parse(WORKER)
    return {
        'scope': 'prepared source only; no execution mode in this preparation module',
        'execution_enabled': False, 'torch_imported': False, 'device_api_called': False,
        'worker_sha256': hashlib.sha256(WORKER.encode()).hexdigest(),
        'cases': list(CASES),
        'required_new_authorization_and_claim': True,
        'normal_exit_only': True,
        'must_run_through_parent_guarded_cumulative_deadline_coordinator': True,
        'operator_numeric_success_does_not_imply_clean_exit': True,
        'marker_scope': 'relative to TorchNPU atexit registration; native static destructors can run later',
    }


if __name__ == '__main__':
    print(json.dumps(preparation_report(), indent=2))
