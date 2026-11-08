"""CPU retained-RGB export only. No model/latent/metric calls and no downloads."""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,json,os,shutil,sys
import h3_limiteddev_media_common_v3 as C

def array_open(desc):
 import numpy as np
 p=Path(desc['payload']['path']);C.require(C.record(p)==desc['payload'],'Original retained/native NPY whole SHA differs')
 a=np.load(p,mmap_mode='r',allow_pickle=False);C.require(a.dtype==np.uint8 and list(a.shape)==C.SHAPE and a.flags.c_contiguous,'Exact226x1080x1920 C-order uint8 NPY required')
 h=hashlib.sha256()
 for frame in a:h.update(memoryview(frame).cast('B'))
 C.require(h.hexdigest()==desc['visible_RGB_sha256'],'Raw RGB SHA differs from actual metric pixels')
 return a

def raw_file(path,frames,width,height,cancel):
 import numpy as np
 C.require(not path.exists(),'No overwrite scratch');h=hashlib.sha256();count=0
 with path.open('xb') as f:
  for a in frames:
   cancel.check();C.require(count<226 and a.shape==(height,width,3) and a.dtype==np.uint8,'Exactly226 bounded frames required');b=np.ascontiguousarray(a);h.update(memoryview(b).cast('B'));f.write(memoryview(b).cast('B'));count+=1
  f.flush();os.fsync(f.fileno())
 C.require(count==226 and path.stat().st_size==226*height*width*3,'Raw stream frame count differs');return {'frames':count,'width':width,'height':height,'raw_RGB_sha256':h.hexdigest(),'bytes':path.stat().st_size}

def render_video(out,spec,frames,plan,runner,cancel,env):
 ff=plan['ffmpeg']['path'];path=out/spec['relative_path'];path.parent.mkdir(exist_ok=True);raw=path.with_suffix('.scratch.rgb24');rawproof=raw_file(raw,frames,spec['width'],spec['height'],cancel)
 command=[ff,'-hide_banner','-nostdin','-n','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',str(spec['width'])+'x'+str(spec['height']),'-r','24','-i',str(raw),'-an','-c:v','mpeg4','-q:v','2','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
 log=path.with_suffix('.encode.log')
 with log.open('x') as f:code=runner.run(command,env,f,timeout=1800)
 C.require(code==0 and path.is_file(),'CPU MPEG4 render failed; partial retained');encode_cleanup=C.completed_child(runner.cleanup)
 command2=[ff,'-hide_banner','-nostdin','-v','info','-xerror','-i',str(path),'-map','0:v:0','-an','-sn','-dn','-vf','showinfo','-vsync','0','-f','null','-'];auditlog=path.with_suffix('.audit.log')
 with auditlog.open('x') as f:code=runner.run(command2,env,f,timeout=600)
 C.require(code==0,'Full226 technical decode failed');audit_cleanup=C.completed_child(runner.cleanup);audit=C.parse_ffmpeg_audit(auditlog.read_text(),path,spec['width'],spec['height']);ap=path.with_suffix('.audit.json');C.write(ap,audit)
 raw.unlink() # Only this attempt's generated scratch, after success; authoritative NPY untouched.
 return {**spec,'video':C.record(path),'audit':C.record(ap),'encode_log':C.record(log),'decode_log':C.record(auditlog),'raw_input':rawproof,'encode_command':command,'audit_command':command2,'encode_cleanup':encode_cleanup,'audit_cleanup':audit_cleanup,'codec':'mpeg4','is_H264':False,'technical_full226_passed':True,'subjective_visual_acceptance':False}

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for n in ('plan','plan-sha256','terminal-binding','terminal-binding-sha256','pixels-outer-sha256'):ap.add_argument('--'+n,required=True)
 ap.add_argument('--attempt',type=int,required=True);ap.add_argument('--previous-failure-sha256');ap.add_argument('--check-only',action='store_true');args=ap.parse_args()
 hidden=C.hidden_cpu();plan=C.verify_plan(args.plan,args.plan_sha256);C.verify_entry(Path(__file__).name,__file__,plan);C.require(C.record(plan['ffmpeg']['path'])==plan['ffmpeg'],'Exact previously used CPU FFmpeg required; no codec fallback');ctx=C.inputs(args,plan)
 for sid in C.DEV:C.crop_specs(plan,sid)
 if args.check_only:
  print(json.dumps({'status':'metadata_and_tools_checked_only','payloads_read':False,'media_encoded':0,'model_calls':0}));return
 C.require(shutil.disk_usage(C.REMOTE_OUTPUT.parent).free>=plan['minimum_render_free_bytes'],'Insufficient bounded rendering disk')
 C.REMOTE_OUTPUT.mkdir(exist_ok=True);owned=False;out=None
 U=C.module('frozen_media_owned_utils',C.STAGE/'run_h3_surrogate67_npu4_v1.py');cancel=U.Cancellation();runner=U.ChildRunner(cancel);env=dict(os.environ)
 C.require(not (C.REMOTE_OUTPUT/'.writer.lock').is_symlink(),'Writer lock cannot be symlink')
 with (C.REMOTE_OUTPUT/'.writer.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   out=C.new_attempt(C.REMOTE_OUTPUT,args.attempt,args.previous_failure_sha256);owned=True;cancel.install();videos=[];crops=[];sources={};runtime={}
   import numpy as np
   import PIL
   runtime={'python':sys.version,'numpy':np.__version__,'Pillow':PIL.__version__,'environment':hidden}
   C.write(out/'started.json',{'utc':U.now(),'plan_sha256':args.plan_sha256,'attempt':args.attempt,'previous_failure_sha256':args.previous_failure_sha256,'parent_pid':os.getpid(),'metadata':ctx['evidence'],'runtime':runtime,'scientific_decodes':0,'optimizer_updates':0})
   for sid in C.DEV:
    clip=ctx['clips'][sid];arrays={k:array_open(clip['retained_RGB'][k]) for k in C.LABELS};sources[sid]={k:clip['retained_RGB'][k] for k in C.LABELS};sub=out/sid;sub.mkdir(exist_ok=False)
    for spec in [s for s in C.video_specs() if s['sample_id']==sid]:
     if spec['route']=='comparison':frames=(C.comparison_frame({k:arrays[k][i] for k in C.LABELS},ctx['labels'][sid],i,ctx['selected_steps']) for i in range(226))
     elif spec['route'] in C.LABELS:frames=(arrays[spec['route']][i] for i in range(226))
     else:
      roi=next(x for x in C.temporal_crops(plan) if x['route']==spec['route'] and x['sample_id']==sid);frames=(C.temporal_crop_frame(arrays,ctx['labels'][sid],roi,i) for i in range(226))
     v=render_video(out,spec,frames,plan,runner,cancel,env)
     if spec['route'] in C.LABELS:C.require(v['raw_input']['raw_RGB_sha256']==clip['retained_RGB'][spec['route']]['visible_RGB_sha256'],'Native single-route input was modified')
     videos.append(v);C.write(sub/(spec['route']+'_video_receipt.json'),v)
    for i,spec in enumerate(C.crop_specs(plan,sid)):
     cancel.check();cells={k:C.native_crop(arrays[k],spec) for k in C.LABELS};p=sub/('native_crop_'+str(i+1).zfill(3)+'.png');C.crop_contact(cells,ctx['labels'][sid],p,spec)
     crops.append({'sample_id':sid,'spec':spec,'relative_path':str(p.relative_to(out)),'image':C.record(p),'width':1536,'height':624,'unscaled512_RGB_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in cells.items()},'source_selection_only':True,'subjective_acceptance':False})
    for k in C.LABELS:C.require(C.record(clip['retained_RGB'][k]['payload']['path'])==clip['retained_RGB'][k]['payload'],'Original RGB payload changed during export')
    del arrays;print(json.dumps({'completed_CPU_media_sample':sid,'videos':8 if sid==C.DEV[0] else 7 if sid==C.DEV[2] else 6,'scientific_decodes':0}),flush=True)
   C.require(len(videos)==21 and len(crops)==84,'Exact bounded deliverable coverage differs');C.unchanged(ctx['evidence']);C.verify_plan(args.plan,args.plan_sha256);C.require(C.record(plan['ffmpeg']['path'])==plan['ffmpeg'],'FFmpeg source changed')
   # Portable allowlist excludes scratch, authoritative arrays and any model/checkpoint.
   members={str(p.relative_to(out)):C.record(p) for p in out.rglob('*') if p.is_file()};C.require(not any(k.endswith('.rgb24') or k.endswith('.npy') or k.endswith('.pt') for k in members),'Unexpected cache/rawscratch in deliverables')
   receipt={'schema':'h3-limiteddev-retainedRGB-render-v3','status':'completed_CPU_originals_full226_technical_audited','plan_sha256':args.plan_sha256,'utc':U.now(),'attempt':args.attempt,'dev_order':list(C.DEV),'native_routes':list(C.LABELS),'main_routes':list(C.MAIN),'display_binding':ctx['display_binding'],'selected_steps':ctx['selected_steps'],'labels':ctx['labels'],'original_evaluation_evidence':ctx['evidence'],'metric_RGB_sources':sources,'videos':videos,'crops':crops,'files':members,'runtime':runtime,'ffmpeg':plan['ffmpeg'],'frames':226,'fps':24,'duration_seconds':226/24,'codec':'mpeg4','H264_completed':False,'scientific_decodes':0,'model_forwards':0,'metric_evaluations':0,'optimizer_updates':0,'test_unlocked':False,'formal16dev_claim':False,'continuous_playback_reviewed':False,'visual_quality_passed':False,'original_pixels_unchanged':True,'limits':plan['limits']}
   C.write(out/'render_receipt.json',receipt);print(json.dumps({'status':receipt['status'],'receipt':C.record(out/'render_receipt.json')}))
  except BaseException as e:
   if owned:C.write(out/'failure.json',{'status':'failed_no_auto_retry','error_type':type(e).__name__,'error':str(e),'partial_outputs_retained':True,'cleanup':runner.cleanup,'scientific_decodes':0,'optimizer_updates':0})
   raise
  finally:
   if runner.active is not None:U.terminate_owned(runner.active);U.await_owned_quiescence(runner.active)
   cancel.restore_handlers()
if __name__=='__main__':main()
