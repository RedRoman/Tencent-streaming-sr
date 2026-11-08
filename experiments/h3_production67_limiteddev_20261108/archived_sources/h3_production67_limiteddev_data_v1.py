"""Limited-dev immutable payload admission over the unchanged typed router.
No registry/cache mutation. Each first-process audit loads all five real payloads
through the strict original child loader. Test remains inaccessible.
"""
import copy, hashlib, json
from pathlib import Path
import h3_production67_cohort_router_v1 as router
ROUTER_SHA='4dd024b749c7134d3aaf2c4686da3f913909edc9b858fa8081c4f6396a4de2aa'
AUDIT_SCHEMA='h3-production67-limiteddev-five-payload-audit-v1'
require=router.require
rows_for=router.rows_for
read_router=router.read_router
load_bundles=router.load_bundles

def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def check_sources():
 require(file_sha(Path(__file__).with_name('h3_production67_cohort_router_v1.py'))==ROUTER_SHA,'Frozen router changed')

def current_stat(p):
 p=Path(p);require(p.is_file() and not p.is_symlink(),'Immutable regular payload required')
 s=p.stat();return {'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns,'inode':s.st_ino,'device':s.st_dev}

def audit_payloads(dataset):
 check_sources();require({s:sum(r['split']==s for r in dataset['rows']) for s in ('train','dev','test')}=={'train':2,'dev':3,'test':0},'Exact limited dataset required')
 records=[];snapshot={}
 for row in dataset['rows']:
  pair,teacher,contract=load_bundles(dataset,row['sample_id'],split=row['split'])
  for base,descs in ((Path(dataset['root'])/row['pair_root'],list(pair['record']['arrays'].values())),(Path(dataset['root'])/row['teacher_root'],[teacher['record']['latent']])):
   for desc in descs:
    p=base/desc['path'];snapshot[str(p)]={'sha256':desc['sha256'],**current_stat(p)}
  records.append({'sample_id':row['sample_id'],'split':row['split'],'cohort_id':dataset['cohort_for_sample'][row['sample_id']],'pair_record_sha256':pair['pair_record_sha256'],'teacher_record_sha256':teacher['teacher_record_sha256'],'data_contract_sha256':pair['record']['data_contract_sha256'],'actual_shape_finite_full_SHA_checked':True})
  del pair,teacher,contract
 current=read_router(dataset['root'],dataset['descriptor']);require(current['metadata_snapshot']==dataset['metadata_snapshot'],'Small metadata changed during actual load')
 return {'schema':AUDIT_SCHEMA,'status':'actual_five_child_payloads_verified_limiteddev_only','checker_sha256':file_sha(__file__),'registry_sha256':dataset['descriptor']['sha256'],'cohort_science_sha256':copy.deepcopy(dataset['cohort_science_sha256']),'counts':{'train':2,'dev':3,'test':0},'sample_order':[r['sample_id'] for r in dataset['rows']],'records':records,'payload_file_stat_and_verified_sha256':snapshot,'metadata_snapshot':current['metadata_snapshot'],'formal_training_allowed':False,'test_access':False,'optimizer_updates':0}

def validate_audit(dataset,audit):
 check_sources();require(audit['schema']==AUDIT_SCHEMA and audit['status']=='actual_five_child_payloads_verified_limiteddev_only' and audit['checker_sha256']==file_sha(__file__),'Actual strict payload audit missing')
 require(audit['registry_sha256']==dataset['descriptor']['sha256'] and audit['cohort_science_sha256']==dataset['cohort_science_sha256'] and audit['counts']=={'train':2,'dev':3,'test':0} and audit['sample_order']==[r['sample_id'] for r in dataset['rows']],'Audit dataset identity differs')
 require(len(audit['records'])==5 and all(x['actual_shape_finite_full_SHA_checked'] is True for x in audit['records']),'Incomplete actual loads')
 current=read_router(dataset['root'],dataset['descriptor']);require(current['metadata_snapshot']==audit['metadata_snapshot'],'Metadata drift after audit')
 require(len(audit['payload_file_stat_and_verified_sha256'])==35,'Missing payload inventory')
 for p,v in audit['payload_file_stat_and_verified_sha256'].items():require(current_stat(p)=={k:v[k] for k in current_stat(p)},'Verified payload changed since audit')
 return True
