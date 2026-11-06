"""Post-hoc CPU-only lag diagnostic. No training, checkpoint selection or NPU execution."""
from pathlib import Path
import hashlib,json,os,datetime
os.environ['OMP_NUM_THREADS']='1'
import torch
torch.set_num_threads(1)
r=Path('${EXPERIMENT_ROOT}');run=r/'upscaler_v5_motion_20261106';out=run/'diagnostics/temporal_alignment_v1.json'
assert not out.exists()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
ids={'dev':['VIDGEN5S_0421364','VIDGEN5S_0665341','VIDGEN5S_0589299','VIDGEN5S_0760711'],'test':['VIDGEN5S_0864318','VIDGEN5S_0128747','VIDGEN5S_0086441','VIDGEN5S_0829322']}
results={};sources={}
for split,clips in ids.items():
 for ident in clips:
  pairpath=r/'upscaler_ultraflash/pairs/samples'/(ident+'.pt');pair=load(pairpath);ref=pair['reference'].float();del pair
  sources[str(pairpath.relative_to(r))]=sha(pairpath)
  assert ref.shape==(24,37,24,42)
  variants={a:run/'evaluation/predictions'/split/a/ident/'latent.pt' for a in ['v5','control','motion']}
  teacher=r/'upscaler_hybrid/teacher_cache'/(ident+'.pt')
  if teacher.exists():variants['teacher']=teacher
  result={}
  # Fixed predicted t=6..29 and identical pixels for every lag; GT t remains in 0..35.
  for label,p in variants.items():
   v=load(p);pred=v['latent'].float();del v
   assert pred.shape==ref.shape
   sources[str(p.relative_to(r))]=sha(p);rows=[]
   for lag in range(-6,7):
    a=pred[:,6:30];b=ref[:,6+lag:30+lag]
    rows.append({'reference_offset':lag,'mse':float((a-b).square().mean()),'difference_mse':float(((a[:,1:]-a[:,:-1])-(b[:,1:]-b[:,:-1])).square().mean())})
   result[label]={'scan':rows,'best_mse_offset':min(rows,key=lambda x:x['mse'])['reference_offset'],'best_difference_offset':min(rows,key=lambda x:x['difference_mse'])['reference_offset']}
  results[ident]={'split':split,'variants':result}
  print(json.dumps({'id':ident,'best_offsets':{k:[v['best_mse_offset'],v['best_difference_offset']] for k,v in result.items()}}),flush=True)
payload={'status':'completed','diagnostic_only':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_code_sha256':sha(Path(__file__)),'device':'CPU','torch_threads':1,'scope':'8 deliberately selected already-inspected clips; not blind, no new training or selection. Fixed predicted latent indices6..29, GT offset-6..6, no interpolation. Negative offset means a prediction resembles an earlier reference. A nonzero optimum can indicate a lag but does not prove its mechanism; zero optimum cannot exclude mixtures/blur. No NPU operations or RGB decoding.','sources':sources,'results':results}
out.parent.mkdir(exist_ok=True)
with out.open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
print(str(out))
