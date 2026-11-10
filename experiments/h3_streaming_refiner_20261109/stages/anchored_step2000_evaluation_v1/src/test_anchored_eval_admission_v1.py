"""CPU-only admission/label controls; fixtures are not model-quality evidence."""
import copy
import json
import os
from pathlib import Path
import tempfile
import sys
from anchored_evaluation_admission import checkpoint_receipt, guard_training_device, sha, write
from encode_anchored_comparison_cpu_v1 import validate_receipts

def rejected(fn):
    try:fn()
    except (AssertionError,FileNotFoundError):return
    raise AssertionError('Invalid evidence was accepted')

def main():
    with tempfile.TemporaryDirectory() as folder:
        p=Path(folder)
        # Old TF names and arbitrary newer-stage files are not fallback targets.
        (p/'adapter_step1000.pt').write_bytes(b'old-TF fixture')
        (p/'adapter_stage2000.pt').write_bytes(b'newer fixture')
        assert checkpoint_receipt(p) is None
        ap=p/'adapter_stage1000.pt';op=p/'private_resume_stage1000.pt'
        ap.write_bytes(b'adapter fixture');op.write_bytes(b'private fixture')
        row={'stage_step':1000,'total_optimizer_updates':3500,'adapter_sha256':sha(ap),
             'private_optimizer_sha256':sha(op),'adapter_bytes':ap.stat().st_size,'private_state_bytes':op.stat().st_size}
        write(p/'checkpoint_stage1000.json',row)
        assert checkpoint_receipt(p)['record']==row
        bad=dict(row,total_optimizer_updates=1000);write(p/'checkpoint_stage1000.json',bad)
        rejected(lambda:checkpoint_receipt(p));write(p/'checkpoint_stage1000.json',row)
        op.write_bytes(b'private changed');rejected(lambda:checkpoint_receipt(p))
        op.unlink();rejected(lambda:checkpoint_receipt(p))
    launch={'physical_npu':7,'parent_pid':os.getpid(),'child_pid':os.getpid(),'child_reaped':False}
    assert guard_training_device(launch,4)['training_process_present']
    rejected(lambda:guard_training_device(launch,7))
    common={'status':'full226_nativeHR_metrics_complete_visual_review_pending','sample_id':'fixture',
            'data_manifest_sha256':'data','raw_decode_shape':[226,1088,1920,3],'crop_y':[4,1084],'fps':24,
            'decoder':'frozen_official_overlap_provisional_for_quality_comparison','optimizer_updates':0}
    candidate=dict(common,arm='anchored_stage1000_candidate',training_step=3500,checkpoint_sha256='adapter')
    dense=dict(common,arm='released_dense_refiner',training_step=None,checkpoint_sha256=None)
    validate_receipts(candidate,dense,'fixture','data','adapter')
    for changes in [{'arm':'strict_streaming_step1000'},{'training_step':1000},
                    {'checkpoint_sha256':'different'},{'crop_y':[0,1080]},{'fps':30}]:
        rejected(lambda changes=changes:validate_receipts(dict(candidate,**changes),dense,'fixture','data','adapter'))
    rejected(lambda:validate_receipts(candidate,dict(dense,checkpoint_sha256='trained'),'fixture','data','adapter'))
    assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
    names=['anchored_evaluation_admission.py','run_anchored_step1000_evaluation_v1.py',
           'evaluate_anchored_streaming_v1.py','encode_anchored_comparison_cpu_v1.py',
           'test_anchored_eval_admission_v1.py','encode_preview_cpu.py']
    print(json.dumps({'status':'passed','scope':'CPU admission and label controls using synthetic metadata; not real33B inference',
          'old_TF_and_latest_substitution_rejected':True,'incomplete_or_tampered_checkpoint_rejected':True,
          'active_training_device_rejected':True,'wrong_arm_step_checkpoint_geometry_rejected':True,
          'tested_sources':{'src/'+n:sha(Path(__file__).with_name(n)) for n in names},
          'torch_imported':False,'accelerator_imported':False,'optimizer_updates':0}))

if __name__=='__main__':main()
