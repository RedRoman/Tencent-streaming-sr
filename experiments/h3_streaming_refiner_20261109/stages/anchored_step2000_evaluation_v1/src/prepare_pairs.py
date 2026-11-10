"""Reuse authorized strict native-HR pairs; freeze exact accepted upsampler."""
from pathlib import Path
import sys,json,hashlib,gc
import numpy as np
import torch

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def prepare(root,output,log,prior_stage):
    root=Path(root);output=Path(output);output.mkdir(exist_ok=False)
    stage=Path(prior_stage)
    sys.path.insert(0,str(stage))
    import h3_production67_cohort_router_v1 as R
    import check_h3_production67_explicit_hf32off_npu_v1 as factory
    from data_utils import encode_silence
    plan=json.loads((stage/'h3_production67_limiteddev_full226_plan_v1.json').read_text())
    for name in ['check_h3_production67_explicit_hf32off_npu_v1.py','h3_production67_cohort_router_v1.py']:
        assert sha(stage/name)==plan['files'][name]['sha256']
    dataset=R.read_router(root,plan['router'])
    weight=root/'upscaler_production67_limiteddev_20261107/teacher/updates/best_step_2000.pt'
    expected='7b9f43c17462fb2d0554645bc7889dc7abdebc052ee9f18f6780624de912e0b5'
    assert sha(weight)==expected
    checkpoint=torch.load(str(weight),map_location='cpu',weights_only=True,mmap=True)
    student,_,_,_,_,_=factory.build(plan,torch)
    student.load_state_dict(checkpoint['model'],strict=True)
    assert all(torch.equal(v,checkpoint['model'][k]) for k,v in student.original_state_dict().items())
    student.to('npu:0').eval().requires_grad_(False)
    records=[]
    for row in R.rows_for(dataset,'train'):
        sid=row['sample_id'];log('validating_native_pair',sample=sid)
        pair,teacher,contract=R.load_bundles(dataset,sid,split='train')
        record=pair['record'];base=root/row['pair_root']
        source=torch.from_numpy(np.load(base/record['arrays']['draft']['path'],allow_pickle=False).copy())
        hr=torch.from_numpy(np.load(base/record['arrays']['reference_hr']['path'],allow_pickle=False).copy())
        assert source.shape==(24,67,44,80) and hr.shape==(24,67,68,120)
        with torch.no_grad():up=student.forward_full(source[None].to('npu:0'),target_size=(67,68,120))[0].float().cpu()
        assert up.shape==hr.shape and torch.isfinite(up).all()
        path=output/(sid+'.pt');torch.save({'upscaled':up,'reference':hr,'sample_id':sid},path)
        records.append({'sample_id':sid,'split':'train','path':str(path),'sha256':sha(path),'pair_record_sha256':pair['pair_record_sha256'],'native_hr_identity':record['native_hr_identity'],'normalization_binding_sha256':record['normalization_binding_sha256'],'cohort':dataset['cohort_for_sample'][sid]})
        log('frozen_upsampler_cached',sample=sid,sha256=records[-1]['sha256'])
        del pair,teacher,source,hr,up
    del student,checkpoint;gc.collect();torch.npu.empty_cache()
    audio,meta=encode_silence(plan['base_model'],frames=226,fps=24,device=torch.device('npu:0'))
    torch.save({'audio_latents':audio,'prompt_embeds':torch.empty(0,5120,dtype=torch.bfloat16),'text_token_tags':torch.empty(0,dtype=torch.long),'audio_metadata':meta,'text_mode':'explicit_no_text_engineering_pilot'},output/'conditioning.pt')
    manifest={'upsampler_sha256':expected,'samples':records,'condition_sha256':sha(output/'conditioning.pt'),'condition_mode':'no_text_known_silence_full226','upstream_full_payload_verification':True,'formal_training_gate_claimed':False,'scope':'user-authorized train2 pilot; strict original router/payload validation retained','geometry':[24,67,68,120]}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest
