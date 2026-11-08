"""Subset CPU H264 adapter. Frozen v3 rendering, science and originals are unchanged."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,re,shutil,sys
sys.path.insert(0,'/path/to/staging/h3_production67_limiteddev_media_20261107_v3')
import h3_limiteddev_media_common_v3 as C

OUTPUT=Path('/path/to/h3-project/upscaler_v5_motion_20261106/analysis/production67_limiteddev_training_20261107/media_h264_v4')
RENDER_SHA='0c871145247fd9b142ce1b3a7d24b09dae28a3bdf3eb478e8b67cceefb3dcead'
V3_PLAN_SHA='77f4d052ae16a042ff9f1df0bf15788e0615a1c17c0462c4d70a971516a38d62'

def subset_specs(kind):
 C.require(kind in ('priority','native'),'Unknown subset')
 return [s for s in C.video_specs() if (s['route']=='comparison' or s['route'].startswith('fixed_'))==(kind=='priority')]

def metadata(r,plan):
 C.require(r['schema']=='h3-limiteddev-retainedRGB-render-v3' and r['status']=='completed_CPU_originals_full226_technical_audited' and r['plan_sha256']==V3_PLAN_SHA and r['dev_order']==list(C.DEV) and r['native_routes']==list(C.LABELS) and r['main_routes']==list(C.MAIN),'Actual complete fixed3dev render metadata required')
 C.require(r['labels']=={sid:C.labels(r['selected_steps'],r['display_binding'],sid) for sid in C.DEV} and r['frames']==226 and r['fps']==24 and r['duration_seconds']==226/24 and r['codec']=='mpeg4' and r['H264_completed'] is False,'Original labels or geometry differs')
 C.require(all(r[k]==0 for k in ('scientific_decodes','model_forwards','metric_evaluations','optimizer_updates')) and all(r[k] is False for k in ('test_unlocked','formal16dev_claim','continuous_playback_reviewed','visual_quality_passed')) and r['original_pixels_unchanged'] is True,'Original scope changed')
 C.require(r['display_binding']['original_teacher_and_GT_identity_policy']==plan['display_identity_policy'] and r['display_binding']['original_teacher_and_GT_review']==plan['display_identity_review'] and r['display_binding']['clips']==C.read(plan['display_identity_policy']['path'])['clips'],'Original teacher/native GT identity differs')
 C.require(len(r['videos'])==21 and len(r['crops'])==84,'Complete21+84 metadata required')
 for n in r['files']:
  p=Path(n);C.require(not p.is_absolute() and '..' not in p.parts and p.suffix in ('.mp4','.png','.json','.log'),'Unsafe original whitelist')
 for v,s in zip(r['videos'],C.video_specs()):
  C.require(all(v[k]==x for k,x in s.items()) and v['codec']=='mpeg4' and v['is_H264'] is False and v['technical_full226_passed'] is True,'Full video declaration changed')
  C.require(C.same_content(v['video'],r['files'][s['relative_path']]) and C.same_content(v['audit'],r['files'][str(Path(s['relative_path']).with_suffix('.audit.json'))]),'Video not in complete render whitelist')
 expected=[(sid,i,s) for sid in C.DEV for i,s in enumerate(C.crop_specs(plan,sid))]
 for item,(sid,i,s) in zip(r['crops'],expected):
  rel=sid+'/native_crop_'+str(i+1).zfill(3)+'.png'
  C.require(item['sample_id']==sid and item['spec']==s and item['relative_path']==rel and item['width']==1536 and item['height']==624 and item['source_selection_only'] is True and item['subjective_acceptance'] is False and C.same_content(item['image'],r['files'][rel]),'Full84 contact declaration changed')

def original_audit(a,v,s):
 C.require(a['passed'] is True and a['decoded_frames']==226 and a['fps']==24 and a['codec']=='mpeg4' and a['exact_duration_seconds']==226/24 and [a['width'],a['height']]==[s['width'],s['height']] and C.same_content(a['file'],v['video']),'Original full226 audit differs')
 rows=a['PTS'];C.require(len(rows)==226 and [x['index'] for x in rows]==list(range(226)) and C.canonical(rows)==a['PTS_sha256'],'Original PTS coverage/hash differs')
 C.require(all(x['width']==s['width'] and x['height']==s['height'] and abs(x['pts_seconds']-x['index']/24)<1e-5 for x in rows) and all(rows[i]['pts']<rows[i+1]['pts'] for i in range(225)),'Original PTS timing/dimensions differs')

def validate_input_file(root,r,s):
 v=next(x for x in r['videos'] if x['relative_path']==s['relative_path']);p=C.exact(root/s['relative_path']);ap=p.with_suffix('.audit.json')
 C.require(C.same_content(C.record(p),v['video']) and C.same_content(C.record(ap),v['audit']),'Selected complete input SHA differs')
 original_audit(C.read(ap),v,s)
 return v

def inputs(root,plan,kind):
 root=C.exact(root);p=root/'render_receipt.json';C.require(C.sha(p)==RENDER_SHA,'Exact actual completed render receipt required');r=C.read(p);metadata(r,plan)
 actual=set()
 for p in root.rglob('*'):
  C.require(not p.is_symlink() and (p.is_file() or p.is_dir()),'Nonregular input')
  if p.is_file():actual.add(str(p.relative_to(root)))
 C.require(actual<=set(r['files'])|{'render_receipt.json'},'Unlisted/partial file in final source root')
 for s in subset_specs(kind):validate_input_file(root,r,s)
 return r

def verify_adapter(path,digest):
 C.require(C.sha(path)==digest,'Adapter plan SHA differs');p=C.read(path)
 C.require(p['schema']=='h3-limiteddev-H264-subset-adapter-plan-v4' and p['render_sha256']==RENDER_SHA and p['frozen_v3_plan']['sha256']==V3_PLAN_SHA and p['output_root']==str(OUTPUT),'Adapter scope changed')
 C.require(C.record(__file__)==p['adapter_source'],'Adapter source changed')
 for k in ('priority','native'):C.require(p['subsets'][k]==[s['relative_path'] for s in subset_specs(k)],'Subset changed')
 plan=C.verify_plan(p['frozen_v3_plan']['path'],V3_PLAN_SHA)
 for d in plan['local_H264_tools'].values():C.require(C.record(d['path'])==d,'Frozen codec/auditor changed')
 return p,plan

def completed_file(rec,r,s,plan,adapter_sha):
 C.require(rec['schema']=='h3-limiteddev-H264-file-v4' and rec['status']=='completed_one_H264_full226_technical_audit' and rec['adapter_plan_sha256']==adapter_sha and rec['render_sha256']==RENDER_SHA and rec['spec']==s,'Reusable file receipt identity differs')
 src=next(v for v in r['videos'] if v['relative_path']==s['relative_path'])
 C.require(C.same_content(rec['source'],src['video']) and C.same_content(rec['source_audit'],src['audit']) and rec['selected_steps']==r['selected_steps'] and rec['labels']==r['labels'][s['sample_id']],'Reusable source/label differs')
 out=C.exact(rec['output']['path']);audit=C.exact(rec['audit']['path']);C.require(out.is_relative_to(OUTPUT) and audit==out.with_suffix('.audit.json'),'Reusable output escaped fixed root')
 C.require(C.record(out)==rec['output'] and C.record(audit)==rec['audit'] and rec['tools']==plan['local_H264_tools'],'Reusable output/tool SHA differs')
 C.validate_h264(C.read(audit),out,s['width'],s['height'])
 for name in ('encode_cleanup','audit_cleanup'):C.completed_child(rec[name])
 C.require(rec['continuous_playback_reviewed'] is False and rec['visual_quality_passed'] is False,'Technical receipt cannot claim subjective pass')

def history(r,plan,adapter_sha):
 found={};specs={s['relative_path']:s for s in C.video_specs()}
 for p in sorted(OUTPUT.glob('batch_attempt_*/*/*_conversion_receipt.json')):
  rec=C.read(p);n=rec['spec']['relative_path'];C.require(n in specs and n not in found,'Duplicate completed output')
  C.require(Path(rec['output']['path'])==p.parent/Path(n).name,'Reusable receipt/output location mismatch')
  completed_file(rec,r,specs[n],plan,adapter_sha);found[n]={'receipt':C.record(p),'record':rec}
 return found

def new_batch(number,previous_failure):
 C.require(type(number)is int and number>=1,'Positive explicit batch attempt required')
 allowed={'.writer.lock'}|{'batch_attempt_'+str(i) for i in range(1,number)}
 C.require({p.name for p in OUTPUT.iterdir()}<=allowed,'Unknown/existing batch output; no overwrite')
 for i in range(1,number):
  p=OUTPUT/('batch_attempt_'+str(i));C.require(p.is_dir() and (p/'failure.json').is_file()!=(p/'batch_receipt.json').is_file(),'Every prior batch requires one retained terminal receipt')
 if number==1:C.require(previous_failure is None,'No implicit retry')
 else:
  previous=OUTPUT/('batch_attempt_'+str(number-1));f=previous/'failure.json';done=previous/'batch_receipt.json'
  C.require(f.is_file()!=done.is_file(),'Prior batch must have one terminal receipt')
  if f.is_file():C.require(previous_failure is not None and C.sha(f)==previous_failure and C.read(f)['status']=='failed_no_auto_retry','Exact failed previous batch required')
  else:C.require(previous_failure is None and C.read(done)['status']=='completed_subset','Prior success must be complete subset')
 out=OUTPUT/('batch_attempt_'+str(number));out.mkdir(exist_ok=False);return out

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for n in ('adapter-plan','adapter-plan-sha256','source-root','subset'):ap.add_argument('--'+n,required=True)
 ap.add_argument('--attempt',type=int,required=True);ap.add_argument('--previous-failure-sha256');a=ap.parse_args();env=C.hidden_cpu();adapter,plan=verify_adapter(a.adapter_plan,a.adapter_plan_sha256);r=inputs(Path(a.source_root),plan,a.subset)
 C.require(shutil.disk_usage(OUTPUT.parent).free>=plan['minimum_H264_free_bytes'],'Insufficient output disk');OUTPUT.mkdir(exist_ok=True);C.require(not (OUTPUT/'.writer.lock').is_symlink(),'Writer lock symlink')
 U=C.module('frozen_v4_owned_utils',C.STAGE/'run_h3_surrogate67_npu4_v1.py');cancel=U.Cancellation();runner=U.ChildRunner(cancel);out=None
 with (OUTPUT/'.writer.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   previous=history(r,plan,a.adapter_plan_sha256);selected=subset_specs(a.subset);out=new_batch(a.attempt,a.previous_failure_sha256);cancel.install();results=[]
   C.write(out/'started.json',{'adapter_plan_sha256':a.adapter_plan_sha256,'render_sha256':RENDER_SHA,'parent_pid':os.getpid(),'environment':env,'subset':a.subset,'selected_paths':[s['relative_path'] for s in selected],'attempt':a.attempt,'previous_failure_sha256':a.previous_failure_sha256})
   for s in selected:
    cancel.check();n=s['relative_path'];source=validate_input_file(Path(a.source_root),r,s)
    if n in previous:results.append({'relative_path':n,'reused_completed':True,'receipt':previous[n]['receipt']});continue
    src=Path(a.source_root)/n;dest=out/n;dest.parent.mkdir(exist_ok=True);C.require(not dest.exists(),'Never overwrite H264')
    cmd=[plan['local_H264_tools']['avconvert']['path'],'--source',str(src),'--output',str(dest),'--preset','PresetHighestQuality']
    with dest.with_suffix('.convert.log').open('x') as f:code=runner.run(cmd,dict(os.environ),f,timeout=1800)
    C.require(code==0 and dest.is_file(),'Conversion failed; preserve partial');encode=C.completed_child(runner.cleanup);audit=dest.with_suffix('.audit.json');cmd2=[plan['local_H264_tools']['auditor']['path'],'--root',str(dest.parent),'--output',str(audit),'--expect-frames','226','--expect-fps','24','--expect-duration',str(226/24),str(dest)]
    with dest.with_suffix('.audit.log').open('x') as f:code=runner.run(cmd2,dict(os.environ),f,timeout=600)
    C.require(code==0,'Full H264 decode/PTS failed');cleanup=C.completed_child(runner.cleanup);C.validate_h264(C.read(audit),dest,s['width'],s['height']);validate_input_file(Path(a.source_root),r,s)
    rec={'schema':'h3-limiteddev-H264-file-v4','status':'completed_one_H264_full226_technical_audit','adapter_plan_sha256':a.adapter_plan_sha256,'render_sha256':RENDER_SHA,'spec':s,'source':C.record(src),'source_audit':source['audit'],'output':C.record(dest),'audit':C.record(audit),'tools':plan['local_H264_tools'],'selected_steps':r['selected_steps'],'labels':r['labels'][s['sample_id']],'encode_cleanup':encode,'audit_cleanup':cleanup,'encode_command':cmd,'audit_command':cmd2,'codec':'H264','no_resize':True,'scientific_decodes':0,'model_forwards':0,'optimizer_updates':0,'continuous_playback_reviewed':False,'visual_quality_passed':False}
    rp=dest.with_name(dest.stem+'_conversion_receipt.json');C.write(rp,rec);results.append({'relative_path':n,'reused_completed':False,'receipt':C.record(rp)})
   C.require(inputs(Path(a.source_root),plan,a.subset)==r,'Selected originals changed');verify_adapter(a.adapter_plan,a.adapter_plan_sha256);all_done=history(r,plan,a.adapter_plan_sha256)
   batch={'schema':'h3-limiteddev-H264-subset-v4','status':'completed_subset','adapter_plan_sha256':a.adapter_plan_sha256,'render_sha256':RENDER_SHA,'subset':a.subset,'paths':[s['relative_path'] for s in selected],'results':results,'actual_H264_completed_count':len(all_done),'full21_video_technical_complete':len(all_done)==21,'contacts_delivery_complete_not_claimed':True,'continuous_playback_reviewed':False,'visual_quality_passed':False}
   C.write(out/'batch_receipt.json',batch)
   if len(all_done)==21:C.write(OUTPUT/'full21_video_receipt.json',{'status':'completed_21_H264_full226_technical_audits','adapter_plan_sha256':a.adapter_plan_sha256,'render_sha256':RENDER_SHA,'outputs':{n:x['receipt'] for n,x in all_done.items()},'continuous_playback_reviewed':False,'visual_quality_passed':False,'contacts_delivery_complete_not_claimed':True})
   print(json.dumps({'status':'completed_subset','batch_receipt':C.record(out/'batch_receipt.json'),'count':len(selected),'full21_video_technical_complete':len(all_done)==21}),flush=True)
  except BaseException as e:
   if out is not None:C.write(out/'failure.json',{'status':'failed_no_auto_retry','error_type':type(e).__name__,'error':str(e),'partial_outputs_retained':True,'cleanup':runner.cleanup,'scientific_decodes':0,'optimizer_updates':0})
   raise
  finally:
   if runner.active is not None:U.terminate_owned(runner.active);U.await_owned_quiescence(runner.active)
   cancel.restore_handlers()

if __name__=='__main__':main()
