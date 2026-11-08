"""Separate local H264 copies of complete verified CPU originals; no arrays/model."""
from pathlib import Path
import argparse,fcntl,json,os,shutil
import h3_limiteddev_media_common_v3 as C

def inputs(args,plan):
 root=C.exact(args.source_root);p=root/'render_receipt.json';C.require(C.sha(p)==args.render_receipt_sha256,'Actual complete original render receipt required');r=C.read(p)
 C.require(r['schema']=='h3-limiteddev-retainedRGB-render-v3' and r['status']=='completed_CPU_originals_full226_technical_audited' and r['plan_sha256']==args.plan_sha256 and r['dev_order']==list(C.DEV) and r['native_routes']==list(C.LABELS) and r['main_routes']==list(C.MAIN),'Complete fixed3dev originals required')
 C.require(r['labels']=={sid:C.labels(r['selected_steps'],r['display_binding'],sid) for sid in C.DEV} and r['frames']==226 and r['fps']==24 and r['duration_seconds']==226/24 and r['codec']=='mpeg4' and r['H264_completed'] is False,'Original geometry/selected labels/codec differs')
 C.require(all(r[k]==0 for k in ('scientific_decodes','model_forwards','metric_evaluations','optimizer_updates')) and all(r[k] is False for k in ('test_unlocked','formal16dev_claim','continuous_playback_reviewed','visual_quality_passed')) and r['original_pixels_unchanged'] is True,'Original scope changed')
 C.require(r['display_binding']['original_teacher_and_GT_identity_policy']==plan['display_identity_policy'] and r['display_binding']['original_teacher_and_GT_review']==plan['display_identity_review'] and r['display_binding']['clips']==C.read(plan['display_identity_policy']['path'])['clips'],'Portable display identity differs from frozen original provenance')
 C.require(len(r['videos'])==21 and len(r['crops'])==84,'Complete fixed videos/crops required')
 C.require(not any(Path(n).suffix in ('.npy','.pt','.rgb24') for n in r['files']),'No arrays/weights/rawscratch delivery')
 for entry in root.rglob('*'):C.require(not entry.is_symlink() and (entry.is_file() or entry.is_dir()),'Nonregular delivered entry')
 actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()};C.require(actual==set(r['files'])|{'render_receipt.json'},'Exact delivered member whitelist differs')
 for name,d in r['files'].items():
  rel=Path(name);C.require(not rel.is_absolute() and '..' not in rel.parts and C.exact(root/rel).is_file() and C.same_content(C.record(root/rel),d),'Delivered original SHA/path differs')
 for v,spec in zip(r['videos'],C.video_specs()):
  C.require(all(v[k]==value for k,value in spec.items()) and v['codec']=='mpeg4' and v['is_H264'] is False and v['technical_full226_passed'] is True,'Video identity/order/geometry differs')
  src=root/spec['relative_path'];C.require(C.same_content(C.record(src),v['video']),'Original video differs')
  ap=src.with_suffix('.audit.json');C.require(C.same_content(C.record(ap),v['audit']),'Original audit SHA differs');a=C.read(ap)
  C.require(a['passed'] is True and a['decoded_frames']==226 and a['fps']==24 and a['codec']=='mpeg4' and a['exact_duration_seconds']==226/24 and [a['width'],a['height']]==[spec['width'],spec['height']] and C.same_content(a['file'],v['video']),'Original full226 audit differs')
 expected_crops=[(sid,i,s) for sid in C.DEV for i,s in enumerate(C.crop_specs(plan,sid))]
 for item,(sid,i,spec) in zip(r['crops'],expected_crops):
  rel=sid+'/native_crop_'+str(i+1).zfill(3)+'.png';C.require(item['sample_id']==sid and item['spec']==spec and item['relative_path']==rel and item['width']==1536 and item['height']==624 and item['source_selection_only'] is True and item['subjective_acceptance'] is False and C.same_content(C.record(root/rel),item['image']),'Native source-only crop identity differs')
 return r

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for n in ('plan','plan-sha256','source-root','render-receipt-sha256'):ap.add_argument('--'+n,required=True)
 ap.add_argument('--attempt',type=int,required=True);ap.add_argument('--previous-failure-sha256');ap.add_argument('--check-only',action='store_true');args=ap.parse_args();env=C.hidden_cpu();plan=C.verify_plan(args.plan,args.plan_sha256);C.verify_entry(Path(__file__).name,__file__,plan)
 for d in plan['local_H264_tools'].values():C.require(C.record(d['path'])==d,'Frozen local converter/auditor differs')
 r=inputs(args,plan)
 if args.check_only:print(json.dumps({'status':'complete_originals_verified_only','new_conversions':0}));return
 C.require(shutil.disk_usage(C.LOCAL_OUTPUT.parent).free>=plan['minimum_H264_free_bytes'],'Insufficient conversion disk');C.LOCAL_OUTPUT.mkdir(exist_ok=True);out=None;owned=False
 U=C.module('frozen_H264_owned_utils',C.STAGE/'run_h3_surrogate67_npu4_v1.py');cancel=U.Cancellation();runner=U.ChildRunner(cancel)
 C.require(not (C.LOCAL_OUTPUT/'.writer.lock').is_symlink(),'Writer lock cannot be symlink')
 with (C.LOCAL_OUTPUT/'.writer.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   out=C.new_attempt(C.LOCAL_OUTPUT,args.attempt,args.previous_failure_sha256);owned=True;cancel.install();results=[]
   C.write(out/'started.json',{'plan_sha256':args.plan_sha256,'original_receipt':C.record(Path(args.source_root)/'render_receipt.json'),'parent_pid':os.getpid(),'environment':env,'attempt':args.attempt})
   for spec in C.video_specs():
    cancel.check();src=Path(args.source_root)/spec['relative_path'];dest=out/spec['relative_path'];dest.parent.mkdir(exist_ok=True);C.require(not dest.exists(),'No overwrite H264')
    cmd=[plan['local_H264_tools']['avconvert']['path'],'--source',str(src),'--output',str(dest),'--preset','PresetHighestQuality'];log=dest.with_suffix('.convert.log')
    with log.open('x') as f:code=runner.run(cmd,dict(os.environ),f,timeout=1800)
    C.require(code==0 and dest.is_file(),'H264 conversion failed; partial retained');encode_cleanup=C.completed_child(runner.cleanup);audit=dest.with_suffix('.audit.json');cmd2=[plan['local_H264_tools']['auditor']['path'],'--root',str(dest.parent),'--output',str(audit),'--expect-frames','226','--expect-fps','24','--expect-duration',str(226/24),str(dest)]
    with dest.with_suffix('.audit.log').open('x') as f:code=runner.run(cmd2,dict(os.environ),f,timeout=600)
    C.require(code==0,'Full H264 decode/PTS audit failed');audit_cleanup=C.completed_child(runner.cleanup);C.validate_h264(C.read(audit),dest,spec['width'],spec['height']);results.append({**spec,'source':C.record(src),'output':C.record(dest),'audit':C.record(audit),'codec':'H264','no_resize':True,'technical_only':True,'encode_cleanup':encode_cleanup,'audit_cleanup':audit_cleanup})
   C.require(inputs(args,plan)==r,'Original media changed');C.verify_plan(args.plan,args.plan_sha256)
   for d in plan['local_H264_tools'].values():C.require(C.record(d['path'])==d,'Converter/auditor changed')
   result={'schema':'h3-limiteddev-retainedRGB-H264-v3','status':'completed_21_H264_full226_technical_audits','plan_sha256':args.plan_sha256,'render_receipt':C.record(Path(args.source_root)/'render_receipt.json'),'selected_steps':r['selected_steps'],'labels':r['labels'],'main_routes':list(C.MAIN),'display_binding':r['display_binding'],'outputs':results,'video_count':21,'frames_each':226,'fps':24,'duration_seconds':226/24,'native_single_route_shape':[1080,1920],'comparison_shape':[652,2880],'fixed_temporal_crop_shape':[624,1536],'fixed_temporal_crop_count':3,'originals_unchanged':True,'source_NPY_downloaded':False,'scientific_decodes':0,'model_forwards':0,'optimizer_updates':0,'test_unlocked':False,'formal16dev_claim':False,'continuous_playback_reviewed':False,'visual_quality_passed':False,'lossless_metric_RGB_remains_authoritative':True,'color_limit':'H264 is lossy inspection output; AVFoundation RGB-to-video color conversion is not a proof of original colorimetric equivalence.'}
   C.write(out/'conversion_receipt.json',result);print(json.dumps({'status':result['status'],'receipt':C.record(out/'conversion_receipt.json')}))
  except BaseException as e:
   if owned:C.write(out/'failure.json',{'status':'failed_no_auto_retry','error_type':type(e).__name__,'error':str(e),'partial_outputs_retained':True,'cleanup':runner.cleanup,'scientific_decodes':0,'optimizer_updates':0})
   raise
  finally:
   if runner.active is not None:U.terminate_owned(runner.active);U.await_owned_quiescence(runner.active)
   cancel.restore_handlers()
if __name__=='__main__':main()
