"""Verify and reuse accepted frozen-upsampled dev predictions; no accelerator.

The original strict router still verifies the full paired payloads. Prediction
and lossless pixel receipts are read-only; new manifests live in this task.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from prepare_pairs import sha

ACCEPTED='7b9f43c17462fb2d0554645bc7889dc7abdebc052ee9f18f6780624de912e0b5'

def checked(desc):
    path=Path(desc['path'])
    assert path.is_file() and path.stat().st_size==desc['bytes']
    assert sha(path)==desc['sha256'],str(path)
    return path

def main():
    p=argparse.ArgumentParser()
    for arg in ('root','prior-stage','prior-evaluation','protocol','condition','output'):
        p.add_argument('--'+arg,required=True)
    a=p.parse_args();root=Path(a.root);stage=Path(a.prior_stage);old=Path(a.prior_evaluation)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(stage))
    import h3_production67_cohort_router_v1 as router
    plan=json.loads((stage/'h3_production67_limiteddev_full226_plan_v1.json').read_text())
    for name,d in plan['files'].items():assert sha(stage/name)==d['sha256']
    protocol=json.loads(Path(a.protocol).read_text())
    ds=router.read_router(root,plan['router'])
    assert [v['sample_id'] for v in router.rows_for(ds,'dev')]==protocol['dev']
    infer=json.loads((old/'infer/report.json').read_text())
    assert infer['status']=='completed' and infer['dev_order']==protocol['dev']
    assert infer['model_proofs']['teacher_student']['selected_model']['sha256']==ACCEPTED
    checked(infer['model_proofs']['teacher_student']['selected_model'])
    records=[]
    for sid in protocol['dev']:
        pair,teacher,contract=router.load_bundles(ds,sid,split='dev')
        rec=pair['record'];row=next(v for v in ds['rows'] if v['sample_id']==sid)
        pred=json.loads(checked(infer['predictions'][sid]['teacher_student']).read_text())
        assert pred['input_identity']['pair_record_sha256']==pair['pair_record_sha256']
        assert pred['evaluated_model']['selected_model']['sha256']==ACCEPTED
        up=np.load(checked(pred['latent']),allow_pickle=False)
        hr=np.load(root/row['pair_root']/rec['arrays']['reference_hr']['path'],allow_pickle=False)
        assert up.shape==hr.shape==(24,67,68,120) and up.dtype==np.float32
        assert np.isfinite(up).all() and np.isfinite(hr).all()
        pixels=json.loads((old/'pixels'/sid/'summary.json').read_text())
        assert pixels['sample_id']==sid and pixels['payload_proof']['pair_record_sha256']==pair['pair_record_sha256']
        retained={}
        for name in ('native_hr','decoded_hr','teacher_student'):
            desc=pixels['retained_RGB'][name]['payload'];z=np.load(checked(desc),allow_pickle=False,mmap_mode='r')
            assert z.shape==(226,1080,1920,3) and z.dtype==np.uint8
            retained[name]=desc
        assert retained['native_hr']['sha256']==rec['arrays']['native_hr_rgb']['sha256']
        path=out/(sid+'.pt')
        torch.save({'sample_id':sid,'upscaled':torch.from_numpy(up.copy()),'reference':torch.from_numpy(hr.copy())},path)
        records.append({'sample_id':sid,'path':str(path),'sha256':sha(path),
                        'pair_record_sha256':pair['pair_record_sha256'],'rgb':retained,
                        'prior_prediction':infer['predictions'][sid]['teacher_student'],
                        'no_refiner_native_metrics':pixels['targets']['native_hr']['teacher_student']})
        print(json.dumps({'event':'verified_dev_payloads','sample_id':sid}),flush=True)
    result={'status':'complete_CPU_verified_reuse','new_accelerator_forwards':0,'optimizer_updates':0,
            'accepted_upsampler_sha256':ACCEPTED,'protocol_sha256':sha(a.protocol),
            'condition':{'path':a.condition,'sha256':sha(a.condition)},'samples':records,
            'nativeHR_is_source_RGB':True,'test_unlocked':False}
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
