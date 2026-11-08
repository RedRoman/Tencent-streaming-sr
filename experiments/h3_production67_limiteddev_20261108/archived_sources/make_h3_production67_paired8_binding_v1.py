"""CPU-only binding of one completed native Charge pair; never launches training."""
import argparse,hashlib,json
from pathlib import Path

def record(p):
 p=Path(p);assert p.is_file() and not p.is_symlink();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':h.hexdigest()}

def build(root):
 root=Path(root).resolve();report=root/'npu_report.json';d=json.loads(report.read_text())
 assert d['status']=='completed' and d['stage']=='npu' and d['actual_encodes']==2 and d['actual_teacher_forwards']==1 and d['actual_decoder_forwards']==0 and d['optimizer_updates']==0 and d['training_ready'] is False
 pair=root/'pair/pair.json';teacher=root/'teacher/teacher.json';contract=root/'data_contract.json'
 assert record(pair)==d['pair'] and record(teacher)==d['teacher']
 pr=json.loads(pair.read_text());tr=json.loads(teacher.read_text());assert pr['sample_id']==tr['sample_id'] and pr['split']==tr['split']=='train'
 assert (root/'pair/data_contract.json').read_bytes()==contract.read_bytes()
 return {'schema':'h3-production67-paired8-real-data-binding-v1','scope':'single_source_engineering_preflight_not_formal_dataset','formal_training_allowed':False,'cache_root':str(root),'contract':record(contract),'acceptance':record(report),'accepted_status':'completed','samples':[{'sample_id':pr['sample_id'],'source_group':pr['source_group'],'split':'train','pair_root':'pair','pair_record_path':'pair.json','pair_record_sha256':record(pair)['sha256'],'teacher_root':'teacher','teacher_record_path':'teacher.json','teacher_record_sha256':record(teacher)['sha256']}]}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=build(a.data_root)
 with a.output.open('x') as f:json.dump(b,f,indent=2);f.write('\n')
 print(json.dumps(record(a.output)))
