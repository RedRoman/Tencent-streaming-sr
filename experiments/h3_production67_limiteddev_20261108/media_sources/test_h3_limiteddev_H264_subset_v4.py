"""New subset admission/PTS/reuse boundaries, synthetic bytes only; no codecs."""
import copy,hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path[:0]=['/tmp','/path/to/staging/h3_production67_limiteddev_media_20261107_v3']
import convert_h3_limiteddev_retained_media_v4 as A
import test_h3_limiteddev_media_v3 as F
C=A.C

def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d));return C.record(p)

class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='h3_v4_subset_synthetic_');self.root=Path(self.tmp.name).resolve();self.plan=C.read('/path/to/staging/h3_production67_limiteddev_media_20261107_v3/h3_limiteddev_retained_rgb_media_plan_v3.json');videos=[];crops=[]
  for s in C.video_specs():
   p=self.root/s['relative_path'];p.parent.mkdir(exist_ok=True);p.write_bytes(b'synthetic-not-media');rows=[{'index':i,'pts':i,'pts_seconds':i/24,'width':s['width'],'height':s['height']} for i in range(226)]
   d={'passed':True,'decoded_frames':226,'fps':24,'codec':'mpeg4','exact_duration_seconds':226/24,'width':s['width'],'height':s['height'],'file':C.record(p),'PTS':rows,'PTS_sha256':C.canonical(rows)};ad=save(p.with_suffix('.audit.json'),d);videos.append({**s,'video':C.record(p),'audit':ad,'codec':'mpeg4','is_H264':False,'technical_full226_passed':True})
  for sid in C.DEV:
   for i,s in enumerate(C.crop_specs(self.plan,sid)):
    rel=sid+'/native_crop_'+str(i+1).zfill(3)+'.png';p=self.root/rel;p.write_bytes(b'synthetic-not-PNG');crops.append({'sample_id':sid,'spec':s,'relative_path':rel,'image':C.record(p),'width':1536,'height':624,'source_selection_only':True,'subjective_acceptance':False})
  display=C.display_binding(*F.fixture(self.plan,False),self.plan);selected={'control':0,'teacher':800}
  self.r={'schema':'h3-limiteddev-retainedRGB-render-v3','status':'completed_CPU_originals_full226_technical_audited','plan_sha256':A.V3_PLAN_SHA,'dev_order':list(C.DEV),'native_routes':list(C.LABELS),'main_routes':list(C.MAIN),'display_binding':display,'labels':{sid:C.labels(selected,display,sid) for sid in C.DEV},'selected_steps':selected,'frames':226,'fps':24,'duration_seconds':226/24,'codec':'mpeg4','H264_completed':False,'original_pixels_unchanged':True,'videos':videos,'crops':crops,'files':{str(p.relative_to(self.root)):C.record(p) for p in self.root.rglob('*') if p.is_file()},**{k:0 for k in ('scientific_decodes','model_forwards','metric_evaluations','optimizer_updates')},**{k:False for k in ('test_unlocked','formal16dev_claim','continuous_playback_reviewed','visual_quality_passed')}}
  save(self.root/'render_receipt.json',self.r)
 def tearDown(self):self.tmp.cleanup()
 def input(self,kind='priority'):
  with patch.object(A,'RENDER_SHA',C.sha(self.root/'render_receipt.json')):return A.inputs(self.root,self.plan,kind)
 def test_01_exact_disjoint_six_fifteen(self):
  p={x['relative_path'] for x in A.subset_specs('priority')};n={x['relative_path'] for x in A.subset_specs('native')};self.assertEqual((len(p),len(n),len(p&n)),(6,15,0));self.assertEqual(p|n,{x['relative_path'] for x in C.video_specs()})
  with self.assertRaises(ValueError):A.subset_specs('latest')
 def test_02_priority_does_not_require_native_or_contacts(self):
  keep={x['relative_path'] for x in A.subset_specs('priority')}
  for p in list(self.root.rglob('*')):
   if p.is_file() and (p.suffix=='.png' or (p.suffix=='.mp4' and str(p.relative_to(self.root)) not in keep)):p.unlink()
  self.assertEqual(len(self.input()['videos']),21);self.assertFalse(self.input()['display_binding']['full226_conditions_passed'])
 def test_03_missing_selected_rejected(self):
  (self.root/A.subset_specs('priority')[0]['relative_path']).unlink()
  with self.assertRaises((ValueError,FileNotFoundError)):self.input()
 def test_04_corrupt_selected_rejected(self):
  (self.root/A.subset_specs('priority')[0]['relative_path']).write_bytes(b'partial')
  with self.assertRaises(ValueError):self.input()
 def test_05_actual_render_hash_required(self):
  with self.assertRaises(ValueError):A.inputs(self.root,self.plan,'priority')
 def test_06_full_metadata_retained_and_labels_fail_closed(self):
  for key in ('crop','route','selected','teacher'):
   r=copy.deepcopy(self.r)
   if key=='crop':r['crops'].pop()
   elif key=='route':r['main_routes'].reverse()
   elif key=='selected':r['selected_steps']['teacher']=2000
   else:r['display_binding']['clips'][C.DEV[0]]['teacher_generation_precision_label']='BF16 parity'
   with self.assertRaises(ValueError):A.metadata(r,self.plan)
 def test_07_whitelist_partial_and_symlink_rejected(self):
  p=self.root/'partial.download';p.write_bytes(b'partial')
  with self.assertRaises(ValueError):self.input()
  p.unlink();p.symlink_to(self.root/'render_receipt.json')
  with self.assertRaises(ValueError):self.input()
 def test_08_pts_missing_duplicate_and_wrong_size_rejected(self):
  s=A.subset_specs('priority')[0];v=next(x for x in self.r['videos'] if x['relative_path']==s['relative_path']);original=C.read(v['audit']['path'])
  for kind in ('short','duplicate','size'):
   a=copy.deepcopy(original)
   if kind=='short':a['PTS'].pop()
   elif kind=='duplicate':a['PTS'][20]['pts']=a['PTS'][19]['pts']
   else:a['PTS'][20]['width']-=1
   a['PTS_sha256']=C.canonical(a['PTS'])
   with self.assertRaises(ValueError):A.original_audit(a,v,s)
 def test_09_batch_no_overwrite_or_unbound_retry(self):
  out=self.root/'out';out.mkdir()
  with patch.object(A,'OUTPUT',out):
   first=A.new_batch(1,None)
   with self.assertRaises(ValueError):A.new_batch(1,None)
   with self.assertRaises(ValueError):A.new_batch(2,None)
   save(first/'failure.json',{'status':'failed_no_auto_retry'})
   with self.assertRaises(ValueError):A.new_batch(2,None)
   self.assertEqual(A.new_batch(2,C.sha(first/'failure.json')).name,'batch_attempt_2')
 def test_10_completed_subset_can_advance_without_retry(self):
  out=self.root/'out';out.mkdir()
  with patch.object(A,'OUTPUT',out):
   first=A.new_batch(1,None);save(first/'batch_receipt.json',{'status':'completed_subset'});self.assertEqual(A.new_batch(2,None).name,'batch_attempt_2')
 def test_11_reused_source_audit_tamper_rejected(self):
  s=A.subset_specs('priority')[0];v=next(x for x in self.r['videos'] if x['relative_path']==s['relative_path']);rec={'schema':'h3-limiteddev-H264-file-v4','status':'completed_one_H264_full226_technical_audit','adapter_plan_sha256':'a'*64,'render_sha256':A.RENDER_SHA,'spec':s,'source':v['video'],'source_audit':{**v['audit'],'sha256':'f'*64},'selected_steps':self.r['selected_steps'],'labels':self.r['labels'][s['sample_id']]}
  with self.assertRaises(ValueError):A.completed_file(rec,self.r,s,self.plan,'a'*64)
 def test_12_completed_reuse_requires_unchanged_output_and_unique_path(self):
  s=A.subset_specs('priority')[0];v=next(x for x in self.r['videos'] if x['relative_path']==s['relative_path']);out=self.root/'out';p=out/'batch_attempt_1'/s['relative_path'];p.parent.mkdir(parents=True);p.write_bytes(b'synthetic-not-H264');d=C.record(p)
  video={'path':p.name,'passed':True,'decoded_frames':226,'nominal_fps':24,'reader_status':'completed','errors':[],'codecs':['avc1'],'width':s['width'],'height':s['height'],'decoded_sizes':[str(s['width'])+'x'+str(s['height'])],'cadence_errors':0,'invalid_pts':0,'nonmonotonic_pts':0,'missing_image_buffers':0,'asset_duration':{'seconds':226/24},'track_time_range':{'duration':{'seconds':226/24}},'first_pts':{'seconds':0},'last_pts':{'seconds':225/24},'pts_span_seconds':225/24,'sha256':d['sha256'],'bytes':d['bytes']}
  audit=save(p.with_suffix('.audit.json'),{'status':'passed','video_count':1,'expected_frames':226,'expected_fps':24,'expected_duration_seconds':226/24,'videos':[video]});cleanup={'reaped':True,'returncode':0,'live_group_members_at_end':[]}
  rec={'schema':'h3-limiteddev-H264-file-v4','status':'completed_one_H264_full226_technical_audit','adapter_plan_sha256':'a'*64,'render_sha256':A.RENDER_SHA,'spec':s,'source':v['video'],'source_audit':v['audit'],'selected_steps':self.r['selected_steps'],'labels':self.r['labels'][s['sample_id']],'output':d,'audit':audit,'tools':self.plan['local_H264_tools'],'encode_cleanup':cleanup,'audit_cleanup':cleanup,'continuous_playback_reviewed':False,'visual_quality_passed':False}
  with patch.object(A,'OUTPUT',out):
   A.completed_file(rec,self.r,s,self.plan,'a'*64);save(p.with_name(p.stem+'_conversion_receipt.json'),rec);self.assertEqual(len(A.history(self.r,self.plan,'a'*64)),1)
   q=out/'batch_attempt_2'/s['relative_path'];save(q.with_name(q.stem+'_conversion_receipt.json'),{**rec,'output':{**d,'path':str(q)}})
   with self.assertRaises(ValueError):A.history(self.r,self.plan,'a'*64)
   p.write_bytes(b'changed')
   with self.assertRaises(ValueError):A.completed_file(rec,self.r,s,self.plan,'a'*64)

if __name__=='__main__':unittest.main(verbosity=2)
