"""CPU-only vendor dependency and exact staged-source import guards.
No torch/model/device work. Preserve env.sh's CANN search paths verbatim in order.
"""
import hashlib,importlib,importlib.util,os,sys
from pathlib import Path
TBE_ORIGIN=Path('/path/to/artifact-storage/SHARED_LOCK_OWNER/envs/minimax-h3-npu/Ascend/cann-9.0.1/python/site-packages/tbe/__init__.py')
TBE_SHA='edea56738f5239851bf84b2f0ff88aced003bf14a2ec1b1eafa90118ccaa64cc'

def require(ok,message):
 if not ok:raise ValueError(message)

def child_pythonpath(stage,inherited):
 require(Path(stage).is_absolute(),'Require absolute staged source directory')
 require(isinstance(inherited,str) and bool(inherited),'Source frozen env.sh before launching; no inherited Python search path')
 parts=inherited.split(os.pathsep)
 require(all(p and Path(p).is_absolute() for p in parts),'No empty/current-directory or relative inherited Python search roots')
 # Stage is first, then ALL environment-derived paths in their original order.
 return os.pathsep.join([str(stage)]+[p for p in parts if p!=str(stage)])

def staged_model_origins(plan,stage):
 stage=Path(stage).resolve();records={}
 for name,r in plan['files'].items():
  if not name.endswith('.py'):continue
  expected=stage/name;require(Path(r['path'])==expected,'Source path outside frozen stage')
  spec=importlib.util.find_spec(name[:-3]);require(spec is not None and spec.origin is not None and Path(spec.origin).resolve()==expected,'Staged source import origin differs: '+name)
  require(hashlib.sha256(expected.read_bytes()).hexdigest()==r['sha256'],'Staged import source SHA differs: '+name)
  records[name]={'origin':str(expected),'sha256':r['sha256']}
 return records

def vendor_check(*,import_tbe):
 spec=importlib.util.find_spec('tbe');require(spec is not None and spec.origin is not None and Path(spec.origin).resolve()==TBE_ORIGIN,'CANN9.0.1 tbe origin missing/wrong; preserve env.sh vendor PYTHONPATH')
 require(hashlib.sha256(TBE_ORIGIN.read_bytes()).hexdigest()==TBE_SHA,'Installed TBE package entry changed')
 r={'find_spec_origin':str(TBE_ORIGIN),'tbe_init_sha256':TBE_SHA,'import_executed':False}
 if import_tbe:
  tbe=importlib.import_module('tbe');require(Path(tbe.__file__).resolve()==TBE_ORIGIN,'Imported TBE origin differs');r['import_executed']=True;r['imported_origin']=str(Path(tbe.__file__).resolve())
 return r

def cpu_preflight(plan,stage):
 require('torch' not in sys.modules and 'torch_npu' not in sys.modules,'Dependency CPU preflight must run before Torch imports')
 keys=('ASCEND_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','TORCH_DEVICE_BACKEND_AUTOLOAD');old={k:os.environ.get(k) for k in keys}
 try:
  for k in keys:os.environ[k]='0' if k=='TORCH_DEVICE_BACKEND_AUTOLOAD' else ''
  result={'staged_sources':staged_model_origins(plan,stage),'vendor':vendor_check(import_tbe=True),'environment':{k:os.environ[k] for k in keys},'torch_imported':'torch' in sys.modules,'torch_npu_imported':'torch_npu' in sys.modules}
  require(not result['torch_imported'] and not result['torch_npu_imported'],'Unexpected Torch/NPU import in TBE dependency preflight')
  return result
 finally:
  for k,v in old.items():
   if v is None:os.environ.pop(k,None)
   else:os.environ[k]=v
