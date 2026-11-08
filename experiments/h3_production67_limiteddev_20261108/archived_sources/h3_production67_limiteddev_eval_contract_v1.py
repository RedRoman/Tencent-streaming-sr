"""Pure CPU identity and equal-clip aggregation for fixed limited-dev full226.
No weights, arrays, model imports, device calls or filesystem writes on import.
"""
import copy, hashlib, json, math
SCHEMA='h3-production67-limiteddev-full226-protocol-v1'
TRAIN_PROTOCOL='cbf2bdc35d854fd427169dda1cede22aea118a5cbaa5e0903a54d3ca2cf011e2'
TERMINAL_SOURCE='a9036ec0ae9d4763080aa69b772ebcbab658db9f7e5e600662f2c22c0e3b8078'
TERMINAL_POLICY='9f047f243634ffdc4db569a23ffb98c5a107198387c3424a603e23427f74b055'
V5='c7aaaa396d877a84c3fed2a5536189b7ed5e61226a82ef1a9e6be1e831f7f914'
REGISTRY='729586146a7f329feed9fe40731651ca1d1a4c3b5a8dfe453b4e29d662c351ac'
DEV=('WingIt_dev_3586_3811','ElephantsDream_dev_13070_13295','WingIt_dev_300_525')
MODELS=('v5','control','teacher_student');TARGETS=('teacher','decoded_hr','native_hr')
FAMILIES=('wing_it','elephants_dream','wing_it')
COHORTS=('original_generation_mode_unbound','strictdet_generation','strictdet_generation')
MODE={'deterministic_algorithms':True,'warn_only':False}
PRECISION={'dtype':'float32','conv_hf32':False,'matmul_hf32':False,'float32_matmul_precision':'highest'}
REGIONS={'full226':226,'prefix208':208,'tail18':18,'blend':60,'nonblend':166}
TRANSITIONS={'blend_touching_pairs':72,'nonblend_only_pairs':153}
SCALARS=('pixel_l1','mse','psnr_db','lpips','global_score','motion_roi_fraction','motion_l1','delta_l1','temporal_difference_mse')
TEMPORAL=('motion_roi_fraction','motion_l1','delta_l1','temporal_difference_mse')
SCIENCE={'original_generation_mode_unbound':'d0af8222ed095516b01bd5897a66c4b4ec9b898c97fc7ddf4d79034ca6afe2db','strictdet_generation':'59743f52860e421d6c7037cd6508e135553bd5a74f9917617e8d75f50722fc2a'}
def require(v,m):
 if not v:raise ValueError(m)
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def terminal_identity(t):
 require(t['schema']=='h3-production67-limiteddev-terminal-metadata-binding-v1' and t['status']=='terminal_metadata_verified_pending_selected_payload_and_full226_eval','Actual reviewed terminal binder result required')
 require(t['binding_source_sha256']==TERMINAL_SOURCE and t['policy_sha256']==TERMINAL_POLICY and t['protocol_sha256']==TRAIN_PROTOCOL and t['registry_sha256']==REGISTRY,'Foreign binder/source/training protocol')
 require(t['scope']=='limiteddev_train2_dev3_test0' and t['actual_updates_reported_and_metadata_cross_checked']==4000 and t['dev_order']==list(DEV) and t['precision']==PRECISION and t['mode']==MODE and t['cohort_science_sha256']==SCIENCE,'Terminal scope/order/precision changed')
 require(all(t[k] is False for k in ('test_access','formal16dev_claim','full226_dev_quality_gate_passed','visual_quality_passed')),'Terminal cannot unlock test or claim quality')
 require(t['metadata_before_after_equal'] is True and [r['arm'] for r in t['arms']]==['control','teacher'] and all(r['completed_updates']==2000 for r in t['arms']),'Incomplete terminal arms')
 require([r['sample_id'] for r in t['dev_input_identities']]==list(DEV),'Dev identities changed')
 for arm in t['arms']:
  d=arm['selected_model_descriptor'];require(d['arm']==arm['arm'] and d['step']==arm['selected_step'] and d['protocol_sha256']==TRAIN_PROTOCOL and d['initializer_checkpoint_sha256']==V5,'Selected descriptor crossed arm/step/protocol')
  require(type(d['step']) is int and d['step'] in range(0,2001,200),'Selected step outside frozen cadence')
 return True

def selected_header(saved,arm_record):
 d=arm_record['selected_model_descriptor']
 require(set(saved)=={'schema','arm','step','protocol_sha256','initializer_checkpoint_sha256','metrics','model'},'Model-only payload must exclude optimizer/RNG/extra state')
 require(saved['schema']=='h3-production67-limiteddev-selected-weights-v1' and saved['arm']==d['arm'] and saved['step']==d['step'] and saved['protocol_sha256']==TRAIN_PROTOCOL and saved['initializer_checkpoint_sha256']==V5 and saved['metrics']==arm_record['selected_metrics'],'Selected payload lineage/score differs')
 return True

def tensor_facts(facts,schema):
 require(len(facts)==len(schema)==322 and set(facts)==set(schema),'322 exact tensor names required')
 total=0
 for name,f in facts.items():
  require(f['dtype']=='torch.float32' and f['device']=='cpu' and f['shape']==schema[name]['shape'] and f['finite'] is True,'Selected tensor shape/dtype/device/finite differs')
  n=math.prod(f['shape']);require(type(f['numel']) is int and f['numel']==n,'Selected tensor numel differs');total+=n
 require(total==345280216,'Selected parameter count differs');return total

def validate_clip(r):
 require(r['sample_id'] in DEV and r['split']=='dev' and r['frames']==226 and r['fps']==[24,1] and r['duration_seconds']==226/24,'Wrong fixed full226 dev clip')
 require(r['optimizer_updates']==0 and r['checkpoint_reselection'] is False and r['test_unlocked'] is False,'Eval cannot train/reselect/test')
 targets=r['targets'];require(set(targets)==set(TARGETS),'Exact three target domains required')
 for domain in TARGETS:
  require(set(targets[domain])==set(MODELS),'Exact three model variants required')
  for model in MODELS:
   v=targets[domain][model];require(v['frames']==226 and v['adjacent_pairs']==225 and len(v['frame_metrics'])==len(v['lpips_per_frame'])==226,'Missing complete frame metrics')
   require(set(v['regions'])==set(REGIONS) and set(v['transitions'])==set(TRANSITIONS),'Missing blend/prefix/tail domains')
   for name,count in REGIONS.items():
    x=v['regions'][name];require(x['frame_count']==count and all(math.isfinite(float(x[k])) for k in SCALARS),'Bad region geometry/metrics')
   for name,count in TRANSITIONS.items():
    x=v['transitions'][name];require(x['adjacent_pairs']==count and all(math.isfinite(float(x[k])) for k in TEMPORAL),'Bad transition metrics')
 for kind,names in (('regions',REGIONS),('transitions',TRANSITIONS)):
  for name in names:
   require(len({targets[t][m][kind][name]['motion_mask_sha256'] for t in TARGETS for m in MODELS})==1,'DecodedHR ROI changed across model/target')
 require(set(r['latent_metrics'])==set(MODELS),'Latent model coverage missing')
 for values in r['latent_metrics'].values():require(all(math.isfinite(float(v)) for v in values.values()),'Nonfinite latent metric')
 return True

def equal_clip_targets(rows):
 require(rows,'Empty aggregation')
 for r in rows:validate_clip(r)
 out={d:{} for d in TARGETS}
 for d in TARGETS:
  for m in MODELS:
   regions={};trans={}
   for name,count in REGIONS.items():
    vals=[r['targets'][d][m]['regions'][name] for r in rows]
    require(len({v['adjacent_pairs'] for v in vals})==1,'Clip transition-count differs')
    regions[name]={k:sum(float(v[k]) for v in vals)/len(vals) for k in SCALARS}
    regions[name].update(frame_count=count,adjacent_pairs=vals[0]['adjacent_pairs'],per_clip_motion_mask_sha256=[v['motion_mask_sha256'] for v in vals])
   for name,count in TRANSITIONS.items():
    vals=[r['targets'][d][m]['transitions'][name] for r in rows]
    trans[name]={k:sum(float(v[k]) for v in vals)/len(vals) for k in TEMPORAL};trans[name].update(adjacent_pairs=count,per_clip_motion_mask_sha256=[v['motion_mask_sha256'] for v in vals])
   out[d][m]={'regions':regions,'transitions':trans}
 return out

def condition_result(targets,metrics,scope):
 # The original function implements every numeric inequality. Only scope text changes.
 v=metrics.engineering_conditions(targets);v['scope']=scope
 require(v['strict_absolute_improvement']==1e-8 and v['guard_factors']=={'global_score':1.01,'pixel_l1':1.02,'lpips':1.02} and v['test_unlocked'] is False and v['checkpoint_reselection'] is False,'Frozen guard contract changed')
 return v

def aggregate(rows,metrics):
 require([r['sample_id'] for r in rows]==list(DEV),'Fixed ordered3dev only; no duplicated/test/replaced windows')
 targets=equal_clip_targets(rows);latent={}
 for m in MODELS:
  vals=[r['latent_metrics'][m] for r in rows];keys=set(vals[0]);require(all(set(v)==keys for v in vals),'Latent metric key mismatch')
  latent[m]={k:sum(float(v[k]) for v in vals)/len(vals) for k in keys}
 require('legacy_ewarp_l1' in latent['v5'],'Original raw latent eWarp must remain separate')
 differences={b:{k:latent['teacher_student'][k]-latent[b][k] for k in latent[b]} for b in ('v5','control')}
 groups={}
 for field,names in (('film_family',FAMILIES),('generation_cohort',COHORTS)):
  groups[field]={}
  for name in dict.fromkeys(names):
   subset=[r for r,n in zip(rows,names) if n==name]
   groups[field][name]={'sample_ids':[r['sample_id'] for r in subset],'count':len(subset),'targets':equal_clip_targets(subset),'diagnostic_only_not_new_gate':True}
 return {'scope':'fixed_limiteddev3_full226_not_formal16dev_or_test','sample_ids':list(DEV),'clip_count':3,
  'aggregation':'Arithmetic equal-clip mean of each reported metric; PSNR is mean per-clip PSNR, ROI/delta are means of per-clip metrics, never pooled pixels. Overall gate weights each Wing window equally; two windows share one film.',
  'targets':targets,'conditions':condition_result(targets,metrics,'Fixed limited-dev3 equal-clip aggregate full226; no reselection, test unlock or visual claim.'),
  'regressions':metrics.regressions(targets),'latent_metrics':latent,'latent_differences_teacher_student_minus_baseline':differences,'groups':groups,
  'per_clip_conditions':{r['sample_id']:condition_result(r['targets'],metrics,'Individual limited-dev clip diagnostic; does not replace aggregate gate.') for r in rows},
  'formal16dev_gate_passed':False,'test_unlocked':False,'checkpoint_reselection':False,'visual_quality_passed':False}
