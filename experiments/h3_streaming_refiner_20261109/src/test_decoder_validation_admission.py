"""CPU-only negative tests for integrity and active-predecessor rejection."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from decoder_validation_admission import checked,sha,predecessor_state,admit,verify_parent_locks


def rejects(fn,kind=ValueError):
    try:fn()
    except kind:return
    raise AssertionError('Invalid admission was accepted')


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate-stage',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);assert not out.exists()
    with tempfile.TemporaryDirectory(prefix='h3-decoder-admission-') as tmp:
        tmp=Path(tmp);payload=tmp/'payload';payload.write_bytes(b'verified payload')
        desc={'path':str(payload),'sha256':sha(payload),'bytes':payload.stat().st_size}
        assert checked(desc)==payload
        payload.write_bytes(b'changed payload!');rejects(lambda:checked(desc))
        launch=tmp/'launch.json'
        launch.write_text(json.dumps({'status':'running','parent_pid':os.getpid(),'child_pid':os.getpid(),'child_reaped':False}))
        assert predecessor_state(launch)['ready'] is False
        launch.write_text(json.dumps({'status':'complete','parent_pid':os.getpid(),'child_pid':os.getpid(),'child_reaped':True}))
        assert predecessor_state(launch)['ready'] is False
        launch.write_text(json.dumps({'status':'complete','child_reaped':True}));rejects(lambda:predecessor_state(launch))
        launch.write_text(json.dumps({'status':'complete','parent_pid':2**62,'child_pid':2**62+1,'child_reaped':True}))
        assert predecessor_state(launch)['ready'] is True
        launch.write_text(json.dumps({'status':'failed','parent_pid':2**62,'child_pid':2**62+1,'child_reaped':False}))
        assert predecessor_state(launch)['ready'] is False
        rejects(lambda:verify_parent_locks(-1))
        rejects(lambda:admit(SimpleNamespace(candidate_stage=args.candidate_stage,validation_plan_sha='0'*64)))
    assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
    result={'status':'passed','scope':'CPU admission negative tests only','payload_corruption_rejected':True,
        'running_predecessor_rejected':True,'live_PID_even_with_reaped_flag_rejected':True,
        'missing_process_identities_rejected':True,'nonreaped_failure_rejected':True,
        'changed_predeclared_plan_rejected':True,'invalid_physical_device_rejected':True,
        'torch_imported':False,'torch_npu_imported':False,'optimizer_updates':0,'accelerator_forwards':0,
        'source_sha256':sha(__file__),
        'admission_source_sha256':sha(Path(__file__).with_name('decoder_validation_admission.py'))}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
