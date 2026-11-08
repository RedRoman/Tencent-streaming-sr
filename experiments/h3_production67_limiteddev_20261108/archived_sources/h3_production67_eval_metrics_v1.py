"""Full226 metrics with the unchanged teacher-dev inequalities, no test unlock.
Bounded-memory accumulator derives from diagnose_h3_720to1080_v3.rgb_agreement;
only ROI source is now the same detached decodedHR for every target/model.
Loss coefficients/ROI threshold and latent raw/residual formulas remain frozen.
"""
import hashlib,math,statistics
import torch
FRAMES=226
EPS=1e-8
MODELS=('v5','control','teacher_student')
TARGETS=('teacher','decoded_hr','native_hr')
DERIVED_FROM={'surrogate':'1dfc47a50b85fc67119629b9e1be2b3907a890eeaef7ebc97f46010e376ad269','teacher_eval':'c22410d662605a8bbb070ad0788972f00c00bddd1db1f487573a19e56b49c7a4'}
def require(x,m):
 if not x:raise ValueError(m)
def frame_regions226():
 blend=[i for boundary in range(1,13) for i in range(17*boundary,17*boundary+5)]
 return dict(full226=list(range(226)),prefix208=list(range(208)),tail18=list(range(208,226)),blend=blend,nonblend=[i for i in range(226) if i not in set(blend)])
def transition_pairs226():
 blend=set(frame_regions226()['blend']);pairs=[(i,i+1) for i in range(225)]
 return dict(blend_touching_pairs=[[a,b] for a,b in pairs if a in blend or b in blend],nonblend_only_pairs=[[a,b] for a,b in pairs if a not in blend and b not in blend])
def rgb_metrics(output,target,shared_decoded_hr,perceptual,device):
 require(output.shape==target.shape==shared_decoded_hr.shape and len(output)==FRAMES and output.ndim==4 and output.shape[-1]==3 and output.dtype==target.dtype==shared_decoded_hr.dtype==torch.uint8,'RGB metric geometry differs')
 require(output.device.type==target.device.type==shared_decoded_hr.device.type=='cpu','Lossless RGB metrics require CPU uint8')
 frames=[];pairs=[]
 for i in range(FRAMES):
  a=output[i].float()/255.;b=target[i].float()/255.;d=a-b
  with torch.no_grad():lp=perceptual((a.permute(2,0,1)[None]*2-1).to(device),(b.permute(2,0,1)[None]*2-1).to(device))
  frames.append(dict(l1=float(d.abs().mean()),mse=float(d.square().mean()),lpips=float(lp.mean())))
  if i:
   pa=output[i-1].float()/255.;pb=target[i-1].float()/255.;td=b-pb
   # Same decodedHR uint8/255 delta mask for teacher, decodedHR and nativeHR.
   fixed=shared_decoded_hr[i].float()/255.-shared_decoded_hr[i-1].float()/255.;mask=fixed.abs().mean(-1)>=.05;residual=(a-pa)-td
   pairs.append(dict(left=i-1,right=i,mask_pixels=int(mask.sum()),pair_pixels=mask.numel(),roi_endpoint_sum=float((((a-b).abs()+(pa-pb).abs())*mask[...,None]).sum(dtype=torch.float64)),delta_abs_sum=float(residual.abs().sum(dtype=torch.float64)),delta_square_sum=float(residual.square().sum(dtype=torch.float64)),scalar_count=residual.numel(),mask_sha256=hashlib.sha256(mask.contiguous().numpy().tobytes()).hexdigest()))
 def group(indices):
  rows=[pairs[i] for i in indices];mask=sum(x['mask_pixels'] for x in rows);n=sum(x['scalar_count'] for x in rows)
  return dict(adjacent_pairs=len(rows),motion_roi_fraction=mask/max(1,sum(x['pair_pixels'] for x in rows)),motion_l1=sum(x['roi_endpoint_sum'] for x in rows)/max(1,mask*6),delta_l1=sum(x['delta_abs_sum'] for x in rows)/max(1,n),temporal_difference_mse=sum(x['delta_square_sum'] for x in rows)/max(1,n),motion_mask_sha256=hashlib.sha256(''.join(x['mask_sha256'] for x in rows).encode()).hexdigest())
 regions={}
 for label,ids in frame_regions226().items():
  chosen=set(ids);ps=[i for i in range(FRAMES-1) if i in chosen and i+1 in chosen];mse=statistics.mean(frames[i]['mse'] for i in ids);l1=statistics.mean(frames[i]['l1'] for i in ids);lp=statistics.mean(frames[i]['lpips'] for i in ids)
  rec=dict(frame_count=len(ids),pixel_l1=l1,mse=mse,psnr_db=-10*math.log10(max(mse,1e-12)),lpips=lp,global_score=l1+.2*lp,**group(ps));require(all(math.isfinite(v) for v in rec.values() if isinstance(v,(int,float))),'Nonfinite RGB metric');regions[label]=rec
 return dict(regions=regions,transitions={k:group([a for a,b in ids]) for k,ids in transition_pairs226().items()},lpips_per_frame=[f['lpips'] for f in frames],frame_metrics=frames,frames=FRAMES,adjacent_pairs=225)
def engineering_conditions(targets):
 require(set(targets)==set(TARGETS) and all(set(targets[t])==set(MODELS) for t in TARGETS),'Exact target/model coverage required')
 primary={m:targets['teacher'][m]['regions']['full226'] for m in MODELS};checks={}
 for baseline in ('v5','control'):
  for key in ('global_score','motion_l1','delta_l1'):checks['teacher_'+key+'_below_'+baseline]=primary['teacher_student'][key]<primary[baseline][key]-EPS
 guards={}
 for domain in ('decoded_hr','native_hr'):
  values={m:targets[domain][m]['regions']['full226'] for m in MODELS}
  guards[domain]={key:values['teacher_student'][key]<=values['v5'][key]*factor for key,factor in (('global_score',1.01),('pixel_l1',1.02),('lpips',1.02))}
  require(all(math.isfinite(float(v)) for row in values.values() for v in row.values() if isinstance(v,(int,float))),'Nonfinite guard input')
 require(all(math.isfinite(float(v)) for row in primary.values() for v in row.values() if isinstance(v,(int,float))),'Nonfinite primary input')
 return dict(teacher_conditions=checks,quality_guards=guards,all_engineering_conditions=all(checks.values()) and all(v for d in guards.values() for v in d.values()),strict_absolute_improvement=EPS,guard_factors={'global_score':1.01,'pixel_l1':1.02,'lpips':1.02},formal16dev_gate_passed=False,test_unlocked=False,checkpoint_reselection=False,scope='Fixed singleWing engineering dev, full226. Original teacher inequalities and decodedHR guard preserved; independent nativeHR guard added and separately reported. Neither condition result unlocks test or proves visual quality.')
def regressions(targets):
 keys=('pixel_l1','mse','lpips','global_score','motion_l1','delta_l1','temporal_difference_mse');out={}
 for t in TARGETS:
  out[t]={}
  for reg in frame_regions226():
   out[t][reg]={}
   for baseline in ('v5','control'):
    d={k:targets[t]['teacher_student']['regions'][reg][k]-targets[t][baseline]['regions'][reg][k] for k in keys};out[t][reg][baseline]={'differences':d,'worse_keys':[k for k,v in d.items() if v>EPS]}
 return out
