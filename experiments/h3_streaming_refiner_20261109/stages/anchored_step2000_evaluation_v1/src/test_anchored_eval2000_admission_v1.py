"""Admission and label controls, not model-quality evidence."""
import json, os, sys, tempfile
from pathlib import Path
import anchored_evaluation_admission2000 as adm
from encode_anchored_stage2000_cpu_v1 import validate_receipts

def rejected(fn):
    try: fn()
    except (AssertionError,FileNotFoundError): return
    raise AssertionError('Invalid evidence was accepted')

def main():
    original={k:getattr(adm,k) for k in ['ENDPOINT_ADAPTER_SHA','ENDPOINT_PRIVATE_SHA','ENDPOINT_RECEIPT_SHA']}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for n in ['adapter_step2000.pt','adapter_stage1000.pt','adapter_stage3000.pt']:
                (p/n).write_bytes(b'not the exact endpoint')
            assert adm.checkpoint_receipt(p) is None
            ap=p/'adapter_stage2000.pt';op=p/'private_resume_stage2000.pt';cp=p/'checkpoint_stage2000.json'
            ap.write_bytes(b'fixture adapter');op.write_bytes(b'fixture private')
            row={'stage_step':2000,'total_optimizer_updates':4500,'adapter_sha256':adm.sha(ap),
                'private_optimizer_sha256':adm.sha(op),'adapter_bytes':ap.stat().st_size,'private_state_bytes':op.stat().st_size}
            adm.write(cp,row)
            # Fixture digests are temporary; production constants are restored below.
            adm.ENDPOINT_ADAPTER_SHA=adm.sha(ap);adm.ENDPOINT_PRIVATE_SHA=adm.sha(op);adm.ENDPOINT_RECEIPT_SHA=adm.sha(cp)
            assert adm.checkpoint_receipt(p)['record']==row
            bad=dict(row,total_optimizer_updates=2000);adm.write(cp,bad)
            rejected(lambda:adm.checkpoint_receipt(p))
            adm.ENDPOINT_RECEIPT_SHA=adm.sha(cp)
            rejected(lambda:adm.checkpoint_receipt(p))
            adm.write(cp,row);adm.ENDPOINT_RECEIPT_SHA=adm.sha(cp)
            op.write_bytes(b'tampered private');rejected(lambda:adm.checkpoint_receipt(p))
            op.unlink();rejected(lambda:adm.checkpoint_receipt(p))
    finally:
        for k,v in original.items():setattr(adm,k,v)
    launch={'physical_npu':7,'parent_pid':os.getpid(),'child_pid':os.getpid(),'child_reaped':False}
    assert adm.guard_training_device(launch,4)['training_process_present']
    rejected(lambda:adm.guard_training_device(launch,7))
    common={'status':'full226_nativeHR_metrics_complete_visual_review_pending','sample_id':'fixture',
            'data_manifest_sha256':'data','raw_decode_shape':[226,1088,1920,3],'crop_y':[4,1084],'fps':24,
            'decoder':'frozen_official_overlap_provisional_for_quality_comparison','optimizer_updates':0}
    candidate=dict(common,arm='anchored_stage2000_candidate',training_step=4500,checkpoint_sha256='adapter')
    dense=dict(common,arm='released_dense_refiner',training_step=None,checkpoint_sha256=None)
    validate_receipts(candidate,dense,'fixture','data','adapter')
    for changes in [{'arm':'anchored_stage1000_candidate'},{'training_step':3500},
                    {'checkpoint_sha256':'different'},{'crop_y':[0,1080]},{'fps':30}]:
        rejected(lambda changes=changes:validate_receipts(dict(candidate,**changes),dense,'fixture','data','adapter'))
    rejected(lambda:validate_receipts(candidate,dict(dense,checkpoint_sha256='trained'),'fixture','data','adapter'))
    reference={'status':'anchored_stage1000_full226_video_ready_visual_review_pending','sample_id':'fixture',
               'stage_step':1000,'total_training_step':3500,'checkpoint':{'record':{'adapter_sha256':adm.REFERENCE_ADAPTER_SHA}},
               'optimizer_updates':0,'training_mutated':False}
    adm.validate_reference_metadata(reference,'fixture')
    for changes in [{'sample_id':'another'},{'stage_step':2000},{'total_training_step':4500},
                    {'optimizer_updates':1},{'checkpoint':{'record':{'adapter_sha256':'latest'}}}]:
        rejected(lambda changes=changes:adm.validate_reference_metadata(dict(reference,**changes),'fixture'))
    assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
    names=['anchored_evaluation_admission2000.py','run_anchored_step2000_evaluation_v1.py',
           'evaluate_anchored_stage2000_v1.py','encode_anchored_stage2000_cpu_v1.py',
           'test_anchored_eval2000_admission_v1.py','encode_preview_cpu.py']
    print(json.dumps({'status':'passed','scope':'CPU admission/label controls on synthetic metadata, not real33B quality',
        'exact2000_only':True,'old_TF_stage1000_latest_and_tamper_rejected':True,
        'active_training_device_rejected':True,'wrong_reference_clip_step_checkpoint_rejected':True,
        'wrong_candidate_step_geometry_checkpoint_rejected':True,'production_pin_constants_restored':True,
        'tested_sources':{'src/'+n:adm.sha(Path(__file__).with_name(n)) for n in names},
        'torch_imported':False,'accelerator_imported':False,'optimizer_updates':0}))

if __name__=='__main__': main()
