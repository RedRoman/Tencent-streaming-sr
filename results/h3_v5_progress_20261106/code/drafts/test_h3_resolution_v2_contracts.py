"""Stdlib-only synthetic receipt checks, not model/media/data validation.

Extracts only five pure contract helpers from the draft AST. It never imports
the diagnostic module, torch, model weights, a source video, or an NPU runtime.
"""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

SOURCE=Path('${SCRATCH}/diagnose_h3_v5_resolution_chain_v2.py')
RESULT=Path('${SCRATCH}/h3_resolution_v2_cpu_structure.json')
IDS=('VIDGEN5S_0421364','VIDGEN5S_0665341','VIDGEN5S_0589299','VIDGEN5S_0760711')


def require(condition,message):
    if not condition:raise ValueError(message)


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def file_record(path):return dict(sha256=sha256(path),bytes=Path(path).stat().st_size)
def read_json(path):return json.loads(Path(path).read_text())
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))


tree=ast.parse(SOURCE.read_text())
compile(tree,str(SOURCE),'exec')
names={'record','check_prepared','artifact','latent_receipts','verify_latent_summary'}
functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
assert len(functions)==len(names)
ns=dict(Path=Path,require=require,sha256=sha256,file_record=file_record,read_json=read_json,IDS=IDS)
exec(compile(ast.Module(body=functions,type_ignores=[]),str(SOURCE),'exec'),ns)
checks=[]


def rejects(name,path,mutate,call):
    original=path.read_bytes();value=read_json(path);mutate(value);write_json(path,value)
    try:
        call()
    except (ValueError,KeyError,FileNotFoundError) as exc:
        checks.append(dict(name=name,passed=True,rejection=str(exc)))
    else:raise AssertionError('Expected rejection: '+name)
    finally:path.write_bytes(original)


with tempfile.TemporaryDirectory(prefix='h3_resolution_v2_contract_test_') as temp:
    output=Path(temp);args=SimpleNamespace(output=output)
    mode={'synthetic_precision':'fp32'}
    contract=dict(execution_precision=mode,posterior='sample',posterior_seed=42,
        new_encodings={'native768':list(IDS),'information_matched384':list(IDS[:2])},
        new_predictions={'teacher_native':list(IDS),'motion_native':list(IDS),'teacher_information_matched':list(IDS[:2])},
        motion_checkpoint={'sha256':'motion-sha'},original_upscaler_fingerprint={'base_weights_sha256':'teacher-sha'},
        pairs={ident:{'pair':{'sha256':'pair-'+ident},'source':{'sha256':'source-'+ident}} for ident in IDS})
    write_json(output/'contract.json',contract);digest=sha256(output/'contract.json')
    reader_records={}
    for ident in IDS:
        folder=output/'readers'/ident;folder.mkdir(parents=True)
        (folder/'reader.log').write_text('synthetic native reader evidence; no actual video was read')
        legacy=folder/'legacy_reader';legacy.mkdir()
        (legacy/f'{ident}.read.log').write_text('synthetic legacy reader evidence')
        proof={'exact_120_frames':True,'fps_generated_duplicate_frames':0,'fps_filter':{'input_frames':124}}
        write_json(legacy/f'{ident}.reader.json',proof)
        rgb={'sha256':'raw-'+ident};native_rgb={'sha256':'native-'+ident};info_rgb={'sha256':'info-'+ident}
        value=dict(sample_id=ident,contract_sha256=digest,reader={'source':contract['pairs'][ident]['source'],'rgb':native_rgb},
            reader_log=file_record(folder/'reader.log'),cached_source384=rgb,information_matched384=info_rgb,
            legacy384_replay={'reader':proof,'rgb':rgb,'cached_source384_bitexact':True,
                'files':{str(p.relative_to(folder)):file_record(p) for p in legacy.iterdir()}})
        write_json(folder/'proof.json',value);reader_records[ident]=ns['record'](folder/'proof.json')
    write_json(output/'prepare.json',dict(status='prepared',contract_sha256=digest,sample_ids=list(IDS),readers=reader_records))
    ns['check_prepared'](args,contract,digest)
    checks.append(dict(name='complete_four_reader_synthetic_chain',passed=True))
    for group,key,kind in (('encodings','new_encodings','encoding'),('predictions','new_predictions','prediction')):
        for label,idents in contract[key].items():
            for ident in idents:
                folder=output/group/label/ident;folder.mkdir(parents=True)
                (folder/'latent.pt').write_bytes(b'synthetic non-tensor sentinel')
                rec=dict(status='completed',contract_sha256=digest,kind=kind,sample_id=ident,variant=label,
                    files={'latent.pt':file_record(folder/'latent.pt')},execution_precision_start=mode,execution_precision_end=mode)
                if kind=='encoding':
                    proof=read_json(output/'readers'/ident/'proof.json')
                    rec.update(frozen_vae_unchanged=True,posterior='sample',posterior_seed=42,
                        input_rgb=proof['reader']['rgb'] if label=='native768' else proof['information_matched384'])
                else:
                    model='motion' if label=='motion_native' else 'teacher'
                    rec.update(frozen_upscaler_unchanged=True,model=model,checkpoint_sha256=model+'-sha',
                        input_origin=ns['record'](output/'encodings/information_matched384'/ident/'done.json')
                            if label=='teacher_information_matched' else contract['pairs'][ident]['pair'])
                write_json(folder/'done.json',rec)
    receipts=ns['latent_receipts'](args,contract,digest);assert len(receipts)==16
    write_json(output/'latent_summary.json',dict(status='latent_completed',contract_sha256=digest,
        no_pixel_decoder_invoked=True,frozen_inputs_and_sources_unchanged=True,receipts=receipts))
    ns['verify_latent_summary'](args,contract,digest)
    checks.append(dict(name='six_encodings_ten_predictions_complete_synthetic_chain',passed=True))
    verify=lambda:ns['verify_latent_summary'](args,contract,digest)
    pred=output/'predictions/teacher_native'/IDS[0]/'done.json'
    info=output/'encodings/information_matched384'/IDS[0]/'done.json'
    rejects('wrong_checkpoint',pred,lambda v:v.update(checkpoint_sha256='other'),verify)
    rejects('wrong_prediction_input',pred,lambda v:v.update(input_origin={'sha256':'other'}),verify)
    rejects('wrong_model',pred,lambda v:v.update(model='motion'),verify)
    rejects('wrong_variant',pred,lambda v:v.update(variant='motion_native'),verify)
    rejects('wrong_execution_precision',pred,lambda v:v.update(execution_precision_end={'synthetic_precision':'other'}),verify)
    rejects('extra_unexpected_artifact_file',pred,lambda v:v['files'].update(extra={'sha256':'x'}),verify)
    rejects('changed_information_encoding_receipt',info,lambda v:v.update(seconds=123),verify)
    rejects('wrong_encoder_input_pixels',info,lambda v:v.update(input_rgb={'sha256':'other'}),verify)
    rejects('wrong_posterior_seed',info,lambda v:v.update(posterior_seed=43),verify)
    rejects('missing_information_encoding_summary_receipt',output/'latent_summary.json',
        lambda v:v['receipts'].pop('information_matched384/'+IDS[0]),verify)
    rejects('missing_prepared_reader',output/'prepare.json',lambda v:v['readers'].pop(IDS[0]),
        lambda:ns['check_prepared'](args,contract,digest))
    rejects('changed_contract_digest',output/'contract.json',lambda v:v.update(extra='changed'),verify)
    assert sha256(output/'contract.json')==digest

result=dict(status='passed',script=str(SOURCE),script_sha256=sha256(SOURCE),test_script_sha256=sha256(__file__),
    scope='Stdlib AST/compile and synthetic receipt rejection checks only. No module import, torch, actual source video, model, media decode, upload or NPU execution.',
    helper_names=sorted(names),checks=checks,check_count=len(checks),real_asset_runtime_validation=False)
write_json(RESULT,result)
print(json.dumps(dict(status='passed',check_count=len(checks),result=str(RESULT),script_sha256=result['script_sha256'])))
