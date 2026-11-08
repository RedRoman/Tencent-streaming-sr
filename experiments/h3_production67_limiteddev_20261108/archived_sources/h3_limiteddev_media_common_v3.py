"""CPU metadata/layout helpers. No Torch, media, arrays or output on import."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,os,re,sys
STAGE=Path('/path/to/staging/h3_production67_limiteddev_media_20261107_v3')
EVAL=Path('/path/to/h3-project/upscaler_production67_limiteddev_evaluation_20261107_v1')
REMOTE_OUTPUT=Path('/path/to/h3-project/upscaler_production67_limiteddev_media_20261107_v3')
LOCAL_OUTPUT=Path('/path/to/h3-project/upscaler_v5_motion_20261106/analysis/production67_limiteddev_training_20261107/media_h264_v3')
SCIENCE_SHA='d031e4383c689bc03615a954bdb9b37fe7970b8425b24775a93b7f7262edb737'
EXEC_SHA='f2c12d031e9f8138554aa8048497c58217dd537471be527ac90087ce78131f0a'
DEV=('WingIt_dev_3586_3811','ElephantsDream_dev_13070_13295','WingIt_dev_300_525')
LABELS=('native_hr','teacher','v5','control','teacher_student')
MAIN=('teacher_student','teacher','native_hr')
ROUND='limiteddev_20261107'
HEADER=112
CROP_WH=[1536,624]
RETAINED=(*LABELS,'decoded_hr')
TEMPORAL_CROPS=({'sample_id':'WingIt_dev_3586_3811','route':'fixed_dog_face_full226','xywh':[560,48,512,512],'source_definition_index':0,'source_region':'dog_face'}, {'sample_id':'WingIt_dev_3586_3811','route':'fixed_cat_face_hand_full226','xywh':[1008,560,512,512],'source_definition_index':1,'source_region':'cat_face_hand'}, {'sample_id':'WingIt_dev_300_525','route':'fixed_camera_pullback_full226','xywh':[704,284,512,512],'source_definition_index':None,'source_region':'fixed_central_camera_pullback_region'})
SHAPE=[226,1080,1920,3]
PREVIEW=[2880,652]
FPS=24;FRAMES=226

def require(v,m):
 if not v:raise ValueError(m)
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):
 p=Path(p);require(p.is_file() and not p.is_symlink(),'Regular file required');h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def record(p):
 p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def exact(p):
 p=Path(p);r=p.resolve();mac_tmp=(sys.platform=='darwin' and str(p).startswith('/path/to/staging/') and r==Path('/private')/p.relative_to('/'));require(p.is_absolute() and (r==p or mac_tmp) and not p.is_symlink(),'Canonical absolute path without symlink required');return r
def read(p):
 p=exact(p);require(p.suffix=='.json' and p.stat().st_size<=64<<20,'Bounded JSON required')
 def pairs(v):
  d={}
  for k,x in v:require(k not in d,'Duplicate JSON key');d[k]=x
  return d
 return json.loads(p.read_text(),object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def same_content(a,b):return a['bytes']==b['bytes'] and a['sha256']==b['sha256']
def hidden_cpu():
 require('torch' not in sys.modules and 'torch_npu' not in sys.modules,'No model imports permitted')
 for k in ('ASCEND_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES'):os.environ[k]=''
 os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0';os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
 return {k:os.environ[k] for k in ('ASCEND_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','TORCH_DEVICE_BACKEND_AUTOLOAD')}
def verify_plan(path,digest):
 require(Path(__file__).resolve()==(STAGE/'h3_limiteddev_media_common_v3.py').resolve(),'Exact staged common-module origin required')
 require(sha(path)==digest,'Exact media plan required');p=read(path)
 require(p['schema']=='h3-limiteddev-retainedRGB-media-plan-v3' and p['stage_directory']==str(STAGE) and p['dev_order']==list(DEV) and p['native_routes']==list(LABELS) and p['main_routes']==list(MAIN),'Fixed media scope differs')
 require(p['evaluation_plan_sha256']==SCIENCE_SHA and p['evaluation_execution_manifest_sha256']==EXEC_SHA and p['shape']==SHAPE and p['fps']==24 and p['preview_wh']==PREVIEW,'Frozen display geometry differs')
 require(p['counts']=={'comparison_videos':3,'native1080_videos':15,'temporal_crop_videos':3,'native_crop_contacts':84,'scientific_decodes':0,'model_forwards':0,'metric_evaluations':0,'optimizer_updates':0},'Fixed call/output scope differs')
 for n,d in p['files'].items():require(d['path']==str(STAGE/n) and record(d['path'])==d,'Media source changed '+n)
 require(record(p['source_crop_authority']['path'])==p['source_crop_authority'],'Source crop definition changed')
 require(record(p['motion_source_authority']['path'])==p['motion_source_authority'],'Source-only motion reference changed')
 require(record(p['frozen_evaluation_plan']['path'])==p['frozen_evaluation_plan'] and p['frozen_evaluation_plan']['sha256']==SCIENCE_SHA,'Scientific plan identity differs')
 require(p['primary_causal_selection']=='frozen_teacher_arm_best_checkpoint_no_cross_arm_ranking' and p['round']==ROUND and p['label_header_pixels']==HEADER and p['native_crop_contact_wh']==CROP_WH,'Display binding changed')
 for key in ('display_identity_policy','display_identity_review','display_requirement'):
  require(record(p[key]['path'])==p[key],'Display identity/requirement evidence changed')
 require(p['display_requirement']['sha256']=='6c445b05c484a837a48a7d21dfd5ab8a9352d6b6cfce2943970b939251de1bf2','User display requirement differs')
 temporal_crops(p)
 return p

def verify_entry(name,actual,plan):
 require(Path(actual).resolve()==(STAGE/name).resolve() and sha(actual)==plan['files'][name]['sha256'],'Exact staged entry source required')

def completed_child(cleanup):
 require(cleanup is not None and cleanup['reaped'] is True and cleanup['returncode']==0 and cleanup['live_group_members_at_end']==[],'Owned child not successfully reaped')
 return dict(cleanup)

def labels(selected,display,sid):
 require(set(selected)=={'control','teacher'} and all(type(x)is int and x in range(0,2001,200) for x in selected.values()),'Actual final selected steps required')
 require(display['round']==ROUND and display['primary_route']=='teacher_student' and display['cross_arm_ranking'] is False and display['checkpoint_reselection'] is False,'Frozen teacher-arm presentation binding required')
 d=display['selected_models']['teacher'];require(d['arm']=='teacher' and d['step']==selected['teacher'] and re.fullmatch('[0-9a-f]{64}',d['sha256']) is not None,'Actual teacher-arm selected weight SHA required')
 origin=display['clips'][sid];up=origin['upstream_weight_sha256'];rgb=origin['native_RGB_sha256']
 require(all(re.fullmatch('[0-9a-f]{64}',x) is not None for x in (up,rgb)),'Original teacher/native identity SHA missing')
 status='FULL226 PASS; visual review pending' if display['full226_conditions_passed'] else 'PROVISIONAL: FULL226 CONDITIONS FAILED'
 names={'teacher_student':['Best causal candidate (teacher arm)',ROUND+' | sparse best step '+str(d['step']),status,'Weight SHA256: '+d['sha256'][:32],d['sha256'][32:],'FP32; HF32 off; latent causal'],
 'teacher':['Open H3 original bidirectional upsampler','Original upstream weight SHA256:',up[:32],up[32:],origin['teacher_generation_precision_label'],'Decode: FP32 HF32 off; strictdet'],
 'native_hr':['GT reference | verified native HR',origin['native_source_label'],'Aligned native RGB SHA256:',rgb[:32],rgb[32:],'Native 1920x1080; same frame indices'],
 'control':['Control frozen best | step '+str(selected['control'])], 'v5':['Initial v5 | step 1800']}
 require(all(isinstance(x,str) and len(x)<=74 for rows in names.values() for x in rows),'Bounded readable display labels required')
 return names

def display_binding(t,summary,loaded,clips,plan):
 require(record(plan['display_identity_policy']['path'])==plan['display_identity_policy'] and record(plan['display_identity_review']['path'])==plan['display_identity_review'],'Reviewed display source changed')
 policy=read(plan['display_identity_policy']['path'])
 require(policy['schema']=='h3-limiteddev-original-teacher-native-display-policy-v1' and policy['status']=='metadata_verified_no_new_science' and policy['dev_order']==list(DEV),'Reviewed original teacher/native identities required')
 require(policy['same_LR_and_explicit_target_verified'] is True and policy['original_bidirectional_path_verified'] is True and policy['native_source_alignment_verified'] is True,'Missing original bidirectional/native evidence')
 require(policy['new_model_calls']==0 and policy['test_unlocked'] is False and policy['review_sha256']==plan['display_identity_review']['sha256'],'Display provenance review identity differs')
 models={a['arm']:a['selected_model_descriptor'] for a in t['arms']}
 for arm,route in (('control','control'),('teacher','teacher_student')):
  proof=loaded['infer'][2]['model_proofs'][route]
  require(proof['selected_model']==models[arm] and proof['all_shapes_FP32_finite_verified'] is True and proof['optimizer_or_rng_loaded'] is False,'Selected payload proof differs from frozen terminal selection')
 conditions=summary['equal_clip_summary']['conditions']
 require(conditions['strict_absolute_improvement']==1e-8 and conditions['guard_factors']=={'global_score':1.01,'pixel_l1':1.02,'lpips':1.02} and conditions['test_unlocked'] is False and conditions['checkpoint_reselection'] is False,'Existing full226 result contract differs')
 booleans=list(conditions['teacher_conditions'].values())+[v for domain in conditions['quality_guards'].values() for v in domain.values()]
 require(set(conditions['teacher_conditions'])=={'teacher_'+k+'_below_'+b for b in ('v5','control') for k in ('global_score','motion_l1','delta_l1')} and all(set(v)=={'global_score','pixel_l1','lpips'} for v in conditions['quality_guards'].values()) and set(conditions['quality_guards'])=={'decoded_hr','native_hr'} and len(booleans)==12 and all(type(v)is bool for v in booleans),'Complete existing full226 condition booleans required')
 require(type(conditions['all_engineering_conditions']) is bool and conditions['all_engineering_conditions']==all(booleans),'Full226 aggregate condition differs')
 identities={}
 for sid in DEV:
  p=clips[sid]['payload_proof'];x=policy['clips'][sid]
  require(all(p[k]==x[k] for k in ('pair_record_sha256','teacher_record_sha256','child_contract_sha256','native_array')) and canonical(p['source_identity'])==x['source_identity_sha256'],'Teacher/native source identity changed')
  require(clips[sid]['retained_RGB']['native_hr']['visible_RGB_sha256']==x['native_RGB_sha256'],'Native RGB sequence differs from accepted source')
  require(x['teacher_variant']=='original_upstream_bidirectional' and x['causal_aligned_or_student_substitution'] is False and x['native_is_teacher_or_decoded_or_upscaled_LR'] is False,'Teacher or GT role substitution forbidden')
  require(p['full_payload_SHA_shape_finite_verified_by_loader2'] is True,'Actual original payload acceptance required')
  for descriptor in ('same_LR_draft_descriptor','original_teacher_latent_descriptor'):
   v=x[descriptor];observed=p['payload_snapshot'][v['path']]
   require(all(observed[k]==v[k] for k in ('bytes','sha256')),'Teacher must use exact same frozen LR and original bidirectional output')
  require(x['upstream_weight_sha256']=='4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6','Original upstream weight differs')
  identities[sid]=x
 return {'round':ROUND,'primary_route':'teacher_student','selection_scope':'best checkpoint within teacher arm under frozen sparse protocol; no cross-arm or cross-experiment ranking','cross_arm_ranking':False,'checkpoint_reselection':False,'selected_models':models,'full226_conditions_passed':conditions['all_engineering_conditions'],'full226_conditions':conditions,'continuous_visual_review_passed':False,'clips':identities,'original_teacher_and_GT_identity_policy':plan['display_identity_policy'],'original_teacher_and_GT_review':plan['display_identity_review']}

def draw_header(draw,x,lines,footer):
 require(len(lines)<=6 and all(isinstance(t,str) and len(t)<=74 for t in lines),'Bounded identity header required')
 for i,line in enumerate(lines):draw.text((x+7,5+15*i),line,fill='white')
 draw.text((x+7,96),footer,fill='white')

def stage_chain(phase,outer,complete,report,outer_desc,report_desc,terminal_sha):
 require(outer['status']=='completed' and outer['stage']==phase and outer['exit_code']==0 and outer['plan_sha256']==SCIENCE_SHA and outer['execution_manifest_sha256']==EXEC_SHA and outer['terminal_binding_sha256']==terminal_sha and outer['optimizer_updates']==0,'Actual successful evaluation outer required')
 require(outer['cleanup']['reaped'] is True and outer['cleanup']['returncode']==0 and outer['cleanup']['live_group_members_at_end']==[],'Evaluation child/group not reaped')
 require(outer['report']==report_desc==complete['report'] and report['status']=='completed' and report['stage']==phase and complete['status']=='completed_after_all_checks' and complete['stage']==phase,'Report/complete chain differs')
 for d in (complete,report):require(d['plan_sha256']==SCIENCE_SHA and d['terminal_binding_sha256']==terminal_sha and d['optimizer_updates']==0,'Evaluation receipt lineage differs')
 if phase=='infer':require(report['actual_new_upscaler_forwards']==9 and report['actual_new_decoder_forwards']==0,'Expected9 completed predictions')
 else:require(report['actual_decoder_forwards']==15,'Expected15 completed decodes')

def retained_descriptor(sid,label,r,proof):
 require(label in RETAINED and r['visible_shape']==SHAPE and r['frames']==226 and r['fps']==[24,1] and r['duration_seconds']==226/24 and r['time_crop'] is None and r['expected_presentation_indices']==[0,225],'Complete already-cropped RGB required')
 require(r['video_encoded'] is False and r['video_PTS_audited'] is False and r['native_display_pixels'] is True and re.fullmatch('[0-9a-f]{64}',r['visible_RGB_sha256']) is not None,'Original RGB identity differs')
 d=r['payload'];exact(d['path'])
 if label=='native_hr':require(d==proof['native_array'] and r['kind']=='existing_verified_native_array' and r['spatial_crop'] is None and r['raw_shape'] is None,'Native source must remain separately bound')
 else:require(d['path']==str(EVAL/'pixels'/sid/(label+'_RGB.npy')) and r['kind']=='retained_same_metric_lossless_RGB' and r['spatial_crop']==[4,1084,0,1920] and r['raw_shape']==[226,1088,1920,3],'Retained metric RGB path/crop differs')
 return d

def inputs(args,plan):
 # Metadata-only admission. It does not recompute metrics or load any model/latent.
 require(exact(args.terminal_binding)!=EVAL/'pixels/summary.json' and sha(args.terminal_binding)==args.terminal_binding_sha256,'Actual terminal binding required')
 C=module('frozen_full226_identity',STAGE/'h3_production67_limiteddev_eval_contract_v1.py');t=read(args.terminal_binding);C.terminal_identity(t)
 evidence={'terminal_binding':record(args.terminal_binding)};loaded={}
 for phase in ('infer','pixels'):
  paths={'outer':EVAL/'launches'/f'{phase}_attempt1.json','complete':EVAL/phase/'complete.json','report':EVAL/phase/'report.json'}
  for k,p in paths.items():evidence[phase+'_'+k]=record(p)
  if phase=='pixels':require(evidence[phase+'_outer']['sha256']==args.pixels_outer_sha256,'Actual pixels outer SHA required')
  o,c,r=[read(paths[k]) for k in ('outer','complete','report')];stage_chain(phase,o,c,r,evidence[phase+'_outer'],evidence[phase+'_report'],args.terminal_binding_sha256)
  require(o['complete']==evidence[phase+'_complete'],'Outer complete descriptor differs');loaded[phase]=(o,c,r)
 require(loaded['pixels'][0]['infer_outer_sha256']==evidence['infer_outer']['sha256'],'Pixels not bound to this actual infer outer')
 sp=EVAL/'pixels/summary.json';evidence['summary']=record(sp);require(loaded['pixels'][2]['summary']==evidence['summary'],'Exact metric summary required');s=read(sp)
 require(s['status']=='completed_limiteddev3_full226_evaluation_not_visual_acceptance' and s['plan_sha256']==SCIENCE_SHA and s['terminal_binding_sha256']==args.terminal_binding_sha256 and s['actual_decoder_forwards']==15 and s['actual_new_upscaler_forwards_this_stage']==0 and s['optimizer_updates']==0,'Complete full226 evaluation required')
 require(s['precision']==C.PRECISION and s['mode']==C.MODE and s['cohort_science_sha256']==t['cohort_science_sha256'] and s['generation_mode_semantics']==t['generation_mode_semantics'],'Science/cohort/mode changed')
 require(all(s[k] is False for k in ('formal16dev_gate_passed','test_unlocked','checkpoint_reselection','video_encoded','video_PTS_audited','visual_quality_passed')),'Media must not unlock test/reselection/quality claims')
 selected={a['arm']:a['selected_step'] for a in t['arms']};require(s['selected_steps']==selected,'Final selection differs');clips={}
 require(len(s['clip_summaries'])==3,'Exactly3 originaldev summaries required')
 for sid,d in zip(DEV,s['clip_summaries']):
  require(d['path']==str(EVAL/'pixels'/sid/'summary.json') and record(d['path'])==d,'Fixed ordered clip summary differs');evidence[sid]=d;r=read(d['path']);C.validate_clip(r)
  require(r['sample_id']==sid and r['plan_sha256']==SCIENCE_SHA and r['terminal_binding_sha256']==args.terminal_binding_sha256 and r['actual_decoder_forwards']==5 and set(r['retained_RGB'])==set(RETAINED),'Clip identity or six retainedroutes differ')
  proof=r['payload_proof'];expected=next(x for x in t['dev_input_identities'] if x['sample_id']==sid)
  require(proof['sample_id']==sid and proof['split']=='dev' and all(proof[k]==expected[k] for k in ('pair_record_sha256','teacher_record_sha256','child_contract_sha256')),'Fixed dev source input differs')
  for label in RETAINED:retained_descriptor(sid,label,r['retained_RGB'][label],proof)
  require(r['retained_RGB']['teacher']['origin']['teacher_record_sha256']==proof['teacher_record_sha256'] and r['retained_RGB']['native_hr']['origin']==proof['source_identity'],'Teacher/native target origin changed')
  for label in ('v5','control','teacher_student'):
   pr=loaded['infer'][2]['predictions'][sid][label];require(record(pr['path'])==pr,'Prediction receipt changed');evidence[sid+'_'+label]=pr;x=read(pr['path'])
   require(x['sample_id']==sid and x['variant']==label and x['terminal_binding_sha256']==args.terminal_binding_sha256 and x['plan_sha256']==SCIENCE_SHA and x['latent']==r['retained_RGB'][label]['origin'],'RetainedRGB does not match evaluated prediction')
  clips[sid]=r
 display=display_binding(t,s,loaded,clips,plan);names={sid:labels(selected,display,sid) for sid in DEV}
 evidence['display_identity_policy']=plan['display_identity_policy'];evidence['display_identity_review']=plan['display_identity_review'];evidence['display_requirement']=plan['display_requirement']
 return {'evidence':evidence,'selected_steps':selected,'labels':names,'display_binding':display,'terminal':t,'summary':s,'clips':clips}

def unchanged(evidence):
 for d in evidence.values():require(record(d['path'])==d,'Immutable source metadata changed')

def crop_specs(plan,sid):
 result=plan['crops'][sid];require(len(result)==(6 if sid==DEV[0] else 36 if sid==DEV[1] else 42),'Fixed source-only crop budget differs')
 for s in result:
  require(type(s['frame'])is int and 0<=s['frame']<226 and s['selected_from_model_outputs'] is False,'Crop frame/model selection forbidden')
  x,y,w,h=s['xywh'];require(all(type(v)is int for v in (x,y,w,h)) and min(x,y)>=0 and w==h==512 and x+w<=1920 and y+h<=1080,'No-resize native512 bounds differ')
 return result

def temporal_crops(plan):
 require(plan['temporal_crops']==list(TEMPORAL_CROPS),'Fixed source-only temporal crop definitions differ')
 source=read(plan['source_crop_authority']['path'])['crops']
 for spec in TEMPORAL_CROPS[:2]:
  ref=source[spec['source_definition_index']]
  require(ref['frame']==0 and ref['source_frame']==3586 and ref['region']==spec['source_region'] and ref['xywh']==spec['xywh'] and ref['selected_from_native_source_only'] is True,'Temporal ROI not exact existing source definition')
 motion=TEMPORAL_CROPS[2];refs=plan['crops'][DEV[2]][-6:]
 require([s['frame'] for s in refs]==list(range(138,144)) and all(s['xywh']==motion['xywh'] and s['selected_from_model_outputs'] is False for s in refs),'Full-time motion crop must reuse predeclared source-only central region')
 return TEMPORAL_CROPS

def video_specs():
 result=[]
 for sid in DEV:
  for route in ('comparison',*LABELS):result.append({'sample_id':sid,'route':route,'relative_path':sid+'/'+('comparison_3column.mp4' if route=='comparison' else route+'_native1080.mp4'),'width':PREVIEW[0] if route=='comparison' else 1920,'height':PREVIEW[1] if route=='comparison' else 1080})
  for spec in TEMPORAL_CROPS:
   if spec['sample_id']!=sid:continue
   result.append({'sample_id':sid,'route':spec['route'],'relative_path':sid+'/'+spec['route']+'_3column.mp4','width':CROP_WH[0],'height':CROP_WH[1],'fixed_xywh':spec['xywh'],'source_definition_index':spec['source_definition_index'],'source_region':spec['source_region'],'all_frames':[0,225],'tracking':False,'spatial_resize':False})
 return result

def temporal_crop_frame(arrays,names,spec,index):
 import numpy as np
 from PIL import Image,ImageDraw
 require(spec in TEMPORAL_CROPS and 0<=index<226 and set(arrays)==set(LABELS),'Fixed complete temporal crop inputs required')
 image=Image.new('RGB',tuple(CROP_WH),'#111111');draw=ImageDraw.Draw(image)
 for i,k in enumerate(MAIN):
  a=native_crop(arrays[k],{'frame':index,'xywh':spec['xywh']});image.paste(Image.fromarray(a),(512*i,HEADER));draw_header(draw,512*i,names[k],f'Frame {index}/225 | PTS {index}/24 | 1x crop')
 return np.asarray(image)

def avg2(frame):
 # Integer uint8 sum/4 is exactly representable in FP32; rint matches the old
 # Torch avg_pool2d(...,2,2).round().byte() nearest-even display-only preview.
 import numpy as np
 require(frame.ndim==3 and frame.shape[-1]==3 and frame.dtype==np.uint8 and frame.shape[0]%2==frame.shape[1]%2==0,'Even uint8 RGB required')
 h,w,_=frame.shape;v=frame.reshape(h//2,2,w//2,2,3).astype(np.uint16).sum(axis=(1,3));return np.rint(v/4).astype(np.uint8)

def comparison_frame(frames,names,index,selected):
 import numpy as np
 from PIL import Image,ImageDraw
 require(set(frames)==set(LABELS) and 0<=index<226,'Exact five baseline frame inputs required; three main views')
 image=Image.new('RGB',tuple(PREVIEW),'#111111');draw=ImageDraw.Draw(image)
 for i,label in enumerate(MAIN):
  a=frames[label];require(a.shape==(1080,1920,3) and a.dtype==np.uint8,'Native input geometry differs')
  x=i*960;image.paste(Image.fromarray(avg2(a)),(x,HEADER));draw_header(draw,x,names[label],f'Frame {index:03d}/225 | PTS {index}/24 s | 24 fps | display 0.5x box average')
 return np.asarray(image)

def native_crop(array,spec):
 import numpy as np
 frame=spec['frame'];x,y,w,h=spec['xywh']
 require(type(frame) is int and 0<=frame<len(array) and all(type(z)is int for z in (x,y,w,h)) and min(x,y)>=0 and w==h==512 and x+w<=array.shape[2] and y+h<=array.shape[1],'Invalid fixed native-scale crop')
 a=np.asarray(array[frame,y:y+h,x:x+w]);require(a.shape==(512,512,3) and a.dtype==np.uint8,'Crop geometry/dtype differs');return a.copy()

def crop_contact(cells,names,path,spec):
 from PIL import Image,ImageDraw
 require(set(cells)==set(LABELS),'Five exact baseline native crops required; three displayed');image=Image.new('RGB',tuple(CROP_WH),'#111111');draw=ImageDraw.Draw(image)
 for i,k in enumerate(MAIN):
  a=cells[k];require(a.shape==(512,512,3),'No crop resize allowed');image.paste(Image.fromarray(a),(512*i,HEADER));draw_header(draw,512*i,names[k],f"Frame {spec['frame']}/225 | static contact | 1x crop")
 require(not Path(path).exists(),'Crop output already exists');image.save(path)

def parse_ffmpeg_audit(stderr,path,width,height):
 duration=re.findall(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)',stderr);require(len(duration)==1,'Container duration absent');h,m,s=duration[0];seconds=3600*int(h)+60*int(m)+float(s);require(abs(seconds-226/24)<.011,'Container duration differs')
 pattern=r'\[Parsed_showinfo[^\]]*\].*?\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:([-+0-9.eE]+).*?\bs:(\d+)x(\d+)'
 rows=[{'index':int(n),'pts':int(p),'pts_seconds':float(t),'width':int(w),'height':int(h)} for n,p,t,w,h in re.findall(pattern,stderr)]
 require(len(rows)==226 and [x['index'] for x in rows]==list(range(226)),'Not all226 frames decoded')
 require(all((x['width'],x['height'])==(width,height) and abs(x['pts_seconds']-x['index']/24)<1e-5 for x in rows),'Video dimensions/PTS differ')
 require(all(rows[i]['pts']<rows[i+1]['pts'] for i in range(225)),'Nonmonotonic/duplicate PTS')
 input_headers=stderr.split('Stream mapping:')[0].split('Output #0')[0]
 codecs=re.findall(r'Stream #0:\d+[^\n]*?Video:\s*([^,\s]+)',input_headers);require(len(codecs)==1 and codecs==['mpeg4'],'Original CPU MPEG4 codec differs; not H264')
 return {'passed':True,'decoded_frames':226,'fps':24,'duration_seconds':seconds,'exact_duration_seconds':226/24,'width':width,'height':height,'codec':'mpeg4','PTS':rows,'PTS_sha256':canonical(rows),'file':record(path),'subjective_visual_acceptance':False}

def validate_h264(a,output,width,height):
 require(a['status']=='passed' and a['video_count']==len(a['videos'])==1 and a['expected_frames']==226 and a['expected_fps']==24 and abs(a['expected_duration_seconds']-226/24)<1e-9,'Audit coverage failure')
 v=a['videos'][0];require(v['path']==Path(output).name and v['passed'] and v['decoded_frames']==226 and v['nominal_fps']==24 and v['reader_status']=='completed' and not v['errors'],'Decode failed')
 require(v['codecs'] in (['avc1'],['avc3']) and [v['width'],v['height']]==[width,height] and v['decoded_sizes']==[str(width)+'x'+str(height)],'Codec/size differs')
 require(v['cadence_errors']==v['invalid_pts']==v['nonmonotonic_pts']==v['missing_image_buffers']==0,'PTS errors')
 require(abs(v['asset_duration']['seconds']-226/24)<=.001 and abs(v['track_time_range']['duration']['seconds']-226/24)<=.001 and abs(v['first_pts']['seconds'])<=1e-6 and abs(v['last_pts']['seconds']-225/24)<=1e-6 and abs(v['pts_span_seconds']-225/24)<=1e-6,'PTS timing differs')
 require(v['sha256']==sha(output) and v['bytes']==Path(output).stat().st_size,'Output hash differs')

def new_attempt(root,number,prior_failure_sha):
 require(type(number)is int and number>=1,'Positive explicit attempt required');exact(root)
 allowed={'.writer.lock'}|{'attempt_'+str(i) for i in range(1,number)}
 require({p.name for p in root.iterdir()}<=allowed,'Unknown/existing attempt output retained; no clobber')
 if number==1:require(prior_failure_sha is None and not any(p.name.startswith('attempt_') for p in root.iterdir()),'No retry implicit')
 else:
  require(prior_failure_sha is not None,'Explicit actual prior failure SHA required')
  for i in range(1,number):
   f=root/('attempt_'+str(i))/'failure.json';require(f.is_file() and not (f.parent/'render_receipt.json').exists() and not (f.parent/'conversion_receipt.json').exists(),'Prior attempt not failed')
  f=root/('attempt_'+str(number-1))/'failure.json';require(sha(f)==prior_failure_sha and read(f)['status']=='failed_no_auto_retry','Prior failure identity differs')
 out=root/('attempt_'+str(number));out.mkdir(exist_ok=False);return out
