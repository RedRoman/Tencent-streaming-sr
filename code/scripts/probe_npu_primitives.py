"""Budget-gated tiny operator probe. No checkpoint/model/visualization.
Never run before a device/time budget is coordinated. Runtime initialization and
workspace memory are measured, not claimed to fit the small tensor budget.
"""
import argparse,json,time,signal,warnings

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--allow-device-probe',action='store_true')
 p.add_argument('--device',type=int,default=0)
 p.add_argument('--timeout-seconds',type=int,default=90)
 a=p.parse_args()
 if not a.allow_device_probe:raise SystemExit('Coordinated device/time budget required; no Torch/NPU import attempted')
 if not 10<=a.timeout_seconds<=120:raise SystemExit('Probe timeout must be10..120s')
 def timeout(*_):raise TimeoutError('operator probe deadline exceeded')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(a.timeout_seconds)
 import torch,torch_npu
 torch.set_num_threads(1);device=f'npu:{a.device}';torch.npu.set_device(a.device)
 results=[]
 def run(name,fn):
  started=time.monotonic();record={'name':name}
  try:
   with warnings.catch_warnings(record=True) as ws,torch.no_grad():
    warnings.simplefilter('always');out=fn();torch.npu.synchronize(a.device)
    record.update(passed=True,device=str(out.device) if torch.is_tensor(out) else None,
        finite=bool(torch.isfinite(out).all().item()) if torch.is_tensor(out) else None,
        warnings=[str(w.message)[:500] for w in ws])
    if record.get('finite') is False:record['passed']=False
  except TimeoutError:raise
  except Exception as error:record.update(passed=False,error_type=type(error).__name__,error=str(error)[:1000])
  record['seconds']=round(time.monotonic()-started,4)
  if hasattr(torch.npu,'max_memory_allocated'):
   record['torch_peak_bytes']=torch.npu.max_memory_allocated(a.device)
   if record['torch_peak_bytes']>1024**3:raise MemoryError('1GiB tracked allocation guard exceeded')
  results.append(record)
  print(json.dumps({'operator_progress':record}),flush=True)
 def autocast_check():
  result=torch.tensor([int(torch.is_autocast_enabled('npu'))],device=device)
  torch.get_autocast_dtype('npu');return result
 run('autocast_query_API',autocast_check)
 def sdpa():
  q=torch.randn(1,4,8,128,device=device,dtype=torch.bfloat16)
  k=torch.randn(1,4,12,128,device=device,dtype=torch.bfloat16);v=torch.randn_like(k)
  mask=torch.arange(12,device=device)[None,:]<=torch.arange(8,device=device)[:,None]
  return torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
 run('BF16_SDPA_variable_QK_lengths_and_mask',sdpa)
 def solve():
  x=torch.randn(4,128,128,device=device,dtype=torch.float32)
  A=x@x.transpose(-1,-2)+torch.eye(128,device=device)*128
  L=torch.linalg.cholesky(A)
  return torch.linalg.solve_triangular(L,torch.eye(128,device=device).expand(4,-1,-1),upper=False)
 run('FP32_Cholesky_and_triangular_solve',solve)
 run('depthwise_Conv2d',lambda:torch.nn.Conv2d(8,8,3,padding=1,groups=8).to(device)(torch.randn(2,8,16,16,device=device)))
 run('RMSNorm',lambda:torch.nn.RMSNorm(128).to(device)(torch.randn(8,128,device=device)))
 def vae_precision():
  layer=torch.nn.Linear(128,128).to(device) # explicit FP32 parameters
  with torch.autocast('npu',dtype=torch.float16):return layer(torch.randn(8,128,device=device))
 run('FP32_weights_FP16_autocast_primitive',vae_precision)
 signal.alarm(0)
 report={'scope':'tiny primitive probe only; not full H3/VAE or pretrained quality',
 'torch':torch.__version__,'torch_npu':torch_npu.__version__,'device':device,'deadline_seconds':a.timeout_seconds,
 'maximum_explicit_test_tensor_bytes':'well below64MiB; runtime/workspace separately measured',
 'results':results,'model_or_checkpoint_loaded':False,
 'passed':all(r['passed'] for r in results)}
 print(json.dumps(report,indent=2),flush=True)
 if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
