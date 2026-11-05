"""Extend fixed-step1800 H3 pixel evaluation to all16 reused test clips.

No training, checkpoint selection, source modification or legacy-result rewrite.
Recompute13 previously undecoded tests plus one old test anchor. Reuse the old
three test pixel records exactly once in the16-test aggregate; keep legacy1
separate. Seven raw MP4s per recomputed clip are decoded completely (98 total).
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import time
import torch

from data_utils import decode_latent, empty_cache, load_video_vae, select_device, synchronize, tensor_load, write_mp4
from decode_results import comparisons
import evaluate_h3_conv3d_hybrid as previous
from evaluate_h3_cmn import file_record, module_source, parity, read_json, require, vae_fingerprint
from evaluate_causal_upscaler_pixels import metric
from h3_conv3d_hybrid import build_student
from prepare_data import load_author_upscaler
from train_h3_cmn import objective
from train_h3_conv3d_hybrid import execution_precision
from train_upscaler_trial import atomic_json, atomic_torch_save, sha256, wrapper_forward
from train_upscaler_ultraflash import load_pairs, tensors

LABELS = ('pretrained', 'old_causal_ewarp', 'cmn', 'cmn_hybrid', 'conv3d_hybrid', 'reference')
MODELS = LABELS[:-1]
HISTORY_LABEL = dict(pretrained='pretrained', old_causal_ewarp='long_ewarp', cmn='cmn', cmn_hybrid='cmn_hybrid', conv3d_hybrid='conv3d_hybrid')
KEYS = previous.PIXEL_KEYS
TARGETS = ('vs_decoded_reference', 'vs_source_rgb_secondary')
SELECTED_SHA = 'c7aaaa396d877a84c3fed2a5536189b7ed5e61226a82ef1a9e6be1e831f7f914'
SOURCE_NAMES = tuple(dict.fromkeys((Path(__file__).name, *previous.SOURCE_FILES, 'prepare_data.py', 'train_h3_conv3d_hybrid.py')))


def frozen_versions(*models):
    return tuple((id(p), p._version) for model in models for p in model.parameters())


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        require(read_json(path) == value, f'Existing immutable evidence differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.partial_{os.getpid()}_{time.time_ns()}')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path)  # Atomic publication, refusing an existing destination.
    temporary.unlink()


def archive_incomplete(directory, output, reason):
    """Move only this extension's incomplete attempt; never overwrite evidence."""
    if directory.exists():
        require(directory.resolve().is_relative_to(output.resolve()), 'Unsafe incomplete-attempt path')
        parent = output/'failures'; parent.mkdir(exist_ok=True)
        target = parent/f'{directory.name}_{time.time_ns()}'
        directory.rename(target)
        immutable_json(target/'failure_context.json', dict(reason=reason, original_path=str(directory), archived_path=str(target)))


def context(args):
    """Recompute the complete immutable input contract at every phase boundary."""
    rows, run, old_hybrid, prior_fingerprints = previous.inputs(args)
    base = read_json(args.base_evaluation/'summary.json')
    snapshot = read_json(args.base_evaluation/'latent_summary.json')
    require(base['status']=='completed' and base['decode_complete'], 'Original pixel evaluation incomplete')
    require(base['latent_summary_sha256']==sha256(args.base_evaluation/'latent_summary.json'), 'Base latent snapshot changed')
    require(snapshot['status']=='latents_completed' and snapshot['input_fingerprints']==prior_fingerprints, 'Base source/data chain changed')
    for key in ('input_fingerprints','variants','selected_videos','latent_files'):
        require(base[key]==snapshot[key], f'Base final/snapshot mismatch: {key}')
    require(run['best_step']==1800 and run['best_checkpoint_sha256']==SELECTED_SHA, 'This extension is fixed to the existing step1800 checkpoint')
    require(base['train_run']['best_step']==1800 and base['train_run']['best_checkpoint_sha256']==SELECTED_SHA, 'Base selected checkpoint differs')
    test_ids=sorted(i for i,r in rows.items() if r['split']=='test')
    reused=sorted(i for i in base['selected_videos'] if rows[i]['split']=='test')
    legacy=[i for i in base['selected_videos'] if rows[i]['split']=='legacy']
    require(len(test_ids)==16 and len(reused)==3 and legacy==['VIDGEN5S_0380588'], 'Unexpected fixed split')
    added=sorted(set(test_ids)-set(reused)); anchor=reused[0]
    require(len(added)==13 and anchor=='VIDGEN5S_0086441', 'Extension selection differs')
    plan=dict(test_ids=test_ids,new_test_ids=added,reused_test_ids=reused,anchor_id=anchor,
              recomputed_ids=[anchor,*added],legacy_ids=legacy,frames=120,fps=24,
              comparison_order=list(LABELS),raw_mp4_count=98,
              selection='All13 remaining retained tests plus lexicographically first old test as parity anchor; fixed model, no selection',
              test_reuse='Previously inspected/reused test set, not a blind test; the anchor is recomputed but counted only once via its original record')
    evaluations={'conv3d_hybrid':(args.base_evaluation,base),'cmn_hybrid':(args.hybrid_evaluation,old_hybrid),
                 'cmn':(args.cmn_evaluation,read_json(args.cmn_evaluation/'summary.json'))}
    run_paths=dict(old_causal_ewarp=args.old_run,cmn=args.cmn_run,cmn_hybrid=args.hybrid_run,conv3d_hybrid=args.run)
    runs={label:read_json(path/'summary.json') for label,path in run_paths.items()}
    checkpoints={}; predictions={}
    for label,path in run_paths.items():
        summary=runs[label]
        require(summary['status']=='completed', f'Incomplete source training: {label}')
        checkpoints[label]=dict(path=str(path/'best.pt'),**file_record(path/'best.pt'))
        require(checkpoints[label]['sha256']==summary['best_checkpoint_sha256'], f'Checkpoint changed: {label}')
        if label=='old_causal_ewarp':
            require(checkpoints[label]['sha256']==base['historical_conv3d_runs']['ewarp']['best_checkpoint_sha256'], 'Old causal checkpoint changed')
        else:
            ev=evaluations[label][1]
            require(sha256(path/'summary.json')==ev['input_fingerprints']['train_summary_sha256'], f'Source summary changed: {label}')
            require(checkpoints[label]['sha256']==ev['input_fingerprints']['checkpoint_sha256'], f'Source checkpoint chain differs: {label}')
        predictions[label]={}
        for ident in test_ids:
            pred_path=path/'predictions/test'/f'{ident}.pt'
            rec=dict(path=str(pred_path),**file_record(pred_path))
            if label!='old_causal_ewarp':
                require(rec['sha256']==evaluations[label][1]['saved_prediction_sha256'][ident], f'Historical prediction SHA differs: {label}/{ident}')
            rec['historical_sha_bound']=label!='old_causal_ewarp'
            predictions[label][ident]=rec
    require(checkpoints['old_causal_ewarp']['sha256']==run['contract']['initial_sha256'], 'Warm-start chain changed')
    files={}
    def bind(path):
        path=Path(path);files[str(path)]=file_record(path)
    for path in (args.base_evaluation/'summary.json',args.base_evaluation/'latent_summary.json',
                 args.hybrid_evaluation/'summary.json',args.hybrid_evaluation/'latent_summary.json',
                 args.cmn_evaluation/'summary.json',args.cmn_evaluation/'latent_summary.json',args.pairs/'READY.json',args.pairs/'plan.json'):
        bind(path)
    for path in run_paths.values(): bind(path/'summary.json')
    for split in ('train','dev','test','legacy'): bind(args.pairs/f'{split}.jsonl')
    for ident in test_ids:
        path=Path(rows[ident]['path']);path=path if path.is_absolute() else args.pairs/path
        require(sha256(path)==rows[ident]['sha256'], f'Pair changed: {ident}')
        bind(path);bind(args.pairs/'flow'/path.name)
    # Bind all referenced original anchor tensors and reused media by actual SHA.
    for ident in base['selected_videos']:
        for label,rec in base['latent_files'][ident].items():
            path=args.base_evaluation/'latents'/ident/f'{label}.pt'
            require(file_record(path)==rec, 'Original saved latent changed');bind(path)
        for name,rec in base['media_audit'][ident].items():
            path=args.base_evaluation/'decoded'/ident/name
            require(file_record(path)=={k:rec[k] for k in ('sha256','bytes')}, 'Original media changed');bind(path)
    bind(args.weights);bind(args.upstream/'open_h3/upscaler.py');bind(args.upstream/'open_h3/upscaler_node.py')
    # Bind the actual decoder/LPIPS assets, not only their historical filenames.
    for name,expected in base['decoder_fingerprint']['weights'].items():
        path=Path(args.base_model)/'vae'/name
        require(file_record(path)==expected,'Decoder asset changed');files[str(path)]=expected
    import lpips
    for path,expected in ((Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth',base['lpips_fingerprint']['backbone']),
                          (Path(lpips.__file__).resolve().parent/'weights/v0.1/alex.pth',base['lpips_fingerprint']['linear_weights'])):
        require(file_record(path)==expected,'LPIPS asset changed');files[str(path)]=expected
    bind(Path(inspect.getfile(lpips.LPIPS)))
    require(sha256(Path(inspect.getfile(lpips.LPIPS)))==base['lpips_fingerprint']['implementation_sha256'],'LPIPS implementation changed')
    original=run['contract']['original_upscaler_fingerprint']
    require(sha256(args.weights)==original['base_weights_sha256'] and sha256(args.upstream/'open_h3/upscaler_node.py')==original['loaded_node_sha256'], 'Original teacher/upscaler changed')
    sources={name:sha256(module_source(name)) for name in SOURCE_NAMES}
    sources[Path(__file__).name]=sha256(__file__)
    sources['run_h3_conv3d_test16.sh']=sha256(Path(__file__).with_name('run_h3_conv3d_test16.sh'))
    contract=dict(schema_version=1,plan=plan,plan_sha256=prior_fingerprints['plan_sha256'],
        selected_step=1800,selected_checkpoint_sha256=SELECTED_SHA,
        base_input_fingerprints=prior_fingerprints,files=files,checkpoints=checkpoints,predictions=predictions,
        source_sha256=sources,decoder_fingerprint=base['decoder_fingerprint'],lpips_fingerprint=base['lpips_fingerprint'],
        training_execution_precision=run['contract']['execution_precision'],
        expected_execution_precision='Read actual flags in inference/decode; never change them. Historical pixel equivalence is independently checked on the anchor.',
        metric_scope='Complete120-frame raw decoded uint8 RGB; fixed latents only, original decoder remains noncausal; no new training or causal claim',
        output=str(args.output),lpips_batch_size=4)
    return contract,rows,base,run


def verify_contract(args):
    current,rows,base,run=context(args)
    require(read_json(args.output/'contract.json')==current, 'Extension inputs/source changed')
    require(read_json(args.output/'plan.json')==current['plan'], 'Extension plan changed')
    return current,rows,base,run,sha256(args.output/'contract.json')


def prepare(args):
    contract,_,_,_=context(args)
    args.output.mkdir(parents=True,exist_ok=True)
    immutable_json(args.output/'contract.json',contract)
    immutable_json(args.output/'plan.json',contract['plan'])
    print(json.dumps(dict(stage='prepare',status='ready',new_test=13,recomputed=14,selected_step=1800)),flush=True)


def valid_receipt(directory, digest, media=False):
    done=directory/'done.json'
    if not done.exists():return None
    receipt=read_json(done)
    require(receipt['status']=='completed' and receipt['contract_sha256']==digest,'Receipt contract differs')
    require(receipt['sample_id']==directory.name,'Receipt sample identity differs')
    kind=directory.parent.name
    if kind in ('pretrained','old_causal_ewarp'):
        require(receipt['variant']==kind and set(receipt['files'])=={'latent.pt'},'Inference receipt variant/files differ')
    elif kind=='latents':
        require(set(receipt['files'])=={f'{label}.pt' for label in LABELS},'Latent receipt variants differ')
    elif kind=='decoded':
        require(set(receipt['files'])=={*[f'{label}.mp4' for label in LABELS],'comparison.mp4','comparison.png','metrics.json'},'Decoded receipt file set differs')
        entry=read_json(directory/'metrics.json')
        require(entry['sample_id']==directory.name and set(entry['variants'])==set(MODELS),'Decoded metrics identity/variant set differs')
    for name,expected in receipt['files'].items():
        require(file_record(directory/name)==expected,f'Completed output changed: {directory/name}')
    if media:
        require(previous.audit_media(directory)==receipt['media_audit'],'Completed raw media failed full replay')
    return receipt


def require_precision(device, contract):
    actual=execution_precision(device)
    require(actual==contract['training_execution_precision'], 'Execution precision differs from original run; do not silently change flags')
    return actual


def metric_parity(actual, expected, context_name):
    for key in previous.METRIC_KEYS:
        require(math.isclose(float(actual[key]),float(expected[key]),abs_tol=1e-5,rel_tol=1e-5), f'Historical latent metric differs: {context_name}/{key}')


def load_saved_prediction(args, contract, pair, label):
    value=tensor_load(Path(contract['predictions'][label][pair['sample_id']]['path']))
    require(value.get('sample_id')==pair['sample_id'], 'Prediction identity differs')
    if label!='old_causal_ewarp':
        require(value.get('split')=='test' and value.get('pair_sha256')==pair['pair_sha256'], 'Prediction pair/split differs')
    latent=value['latent'].float()
    require(tuple(latent.shape)==(24,37,24,42) and bool(torch.isfinite(latent).all()), 'Invalid saved prediction')
    return latent


@torch.no_grad()
def infer(args):
    contract,_,base,_,digest=verify_contract(args)
    device=select_device(args.device);start_precision=require_precision(device,contract)
    pairs={p['sample_id']:p for p in load_pairs(args.pairs/'test.jsonl')}
    planned=contract['plan']['recomputed_ids'];anchor=contract['plan']['anchor_id']
    for label in ('pretrained','old_causal_ewarp'):
        todo=[]
        for ident in planned:
            dst=args.output/'inference'/label/ident
            if valid_receipt(dst,digest) is None:todo.append(ident)
        if not todo:continue
        if label=='pretrained':
            node=load_author_upscaler(args.upstream,args.weights,device)
            model=node.load_model('upscaler',device,'fp32').eval().requires_grad_(False)
            require(sha256(Path(node.__file__))==contract['files'][str(args.upstream/'open_h3/upscaler_node.py')]['sha256'],'Actual teacher module differs')
        else:
            node=None;model=build_student(args.old_run/'best.pt',args.upstream,args.weights,device).eval().requires_grad_(False)
        versions=frozen_versions(model)
        for ident in todo:
            dst=args.output/'inference'/label/ident
            archive_incomplete(dst,args.output,'Inference interrupted before atomic done.json; retained before retry')
            dst.mkdir(parents=True)
            pair=pairs[ident];draft,ref,flow,valid=tensors(pair,device)
            pred=(wrapper_forward(node,model,draft,ref.shape[-3:]) if label=='pretrained' else model(draft))[0]
            require(bool(torch.isfinite(pred).all()),'Nonfinite inferred latent')
            values={k:float(v) for k,v in objective(pred[None],ref,flow,valid,.1).items()}
            metric_parity(values,base['variants'][HISTORY_LABEL[label]][ident],f'{label}/{ident}')
            checks={}
            if label=='old_causal_ewarp':
                checks['independent_checkpoint_vs_unbound_prediction_cache']=parity(pred.cpu(),load_saved_prediction(args,contract,pair,label),3e-5)
            if ident==anchor:
                historical=tensor_load(args.base_evaluation/'latents'/ident/f'{HISTORY_LABEL[label]}.pt')['latent']
                checks['historical_anchor_tensor']=parity(pred.cpu(),historical,3e-5)
            require(all(c['passed'] and c['max_abs']<=1e-4 for c in checks.values()),'Independent tensor parity failed')
            require(frozen_versions(model)==versions and all(p.grad is None for p in model.parameters()),'Frozen inference model changed before receipt')
            atomic_torch_save(dst/'latent.pt',dict(latent=pred.cpu(),sample_id=ident,split='test',pair_sha256=pair['pair_sha256']))
            end_precision=require_precision(device,contract)
            immutable_json(dst/'done.json',dict(status='completed',contract_sha256=digest,sample_id=ident,variant=label,
                files={'latent.pt':file_record(dst/'latent.pt')},latent_metrics=values,parity=checks,
                execution_precision_start=start_precision,execution_precision_end=end_precision))
            print(json.dumps(dict(stage='inference',variant=label,sample_id=ident,passed=True)),flush=True)
            del draft,ref,flow,valid,pred
        require(frozen_versions(model)==versions and all(p.grad is None for p in model.parameters()),'Frozen inference model changed')
        if node is not None:node.MODEL_CACHE.clear()
        del model,node;empty_cache(device)
    # Three students are bound to original19 prediction SHA records. Only old
    # eWarp lacked that historic binding and therefore was independently rerun.
    for ident in planned:
        dst=args.output/'latents'/ident
        if valid_receipt(dst,digest) is not None:continue
        archive_incomplete(dst,args.output,'Latent assembly incomplete; retained before retry');dst.mkdir(parents=True)
        pair=pairs[ident];_,ref,flow,valid=tensors(pair,device);records={};checks={};files={}
        for label in LABELS:
            if label=='reference':latent=pair['reference']
            elif label in ('pretrained','old_causal_ewarp'):
                source=args.output/'inference'/label/ident
                require(valid_receipt(source,digest) is not None,'Missing completed inference')
                latent=tensor_load(source/'latent.pt')['latent']
            else:latent=load_saved_prediction(args,contract,pair,label)
            if label!='reference':
                values={k:float(v) for k,v in objective(latent[None].to(device),ref,flow,valid,.1).items()}
                metric_parity(values,base['variants'][HISTORY_LABEL[label]][ident],f'{label}/{ident}');records[label]=values
            if ident==anchor:
                old_label='reference' if label=='reference' else HISTORY_LABEL[label]
                old=tensor_load(args.base_evaluation/'latents'/ident/f'{old_label}.pt')['latent']
                checks[label]=parity(latent,old,3e-5)
                require(checks[label]['passed'] and checks[label]['max_abs']<=1e-4,'Anchor latent differs')
            path=dst/f'{label}.pt';atomic_torch_save(path,dict(latent=latent.cpu(),sample_id=ident,split='test',pair_sha256=pair['pair_sha256']))
            files[path.name]=file_record(path)
        immutable_json(dst/'done.json',dict(status='completed',contract_sha256=digest,sample_id=ident,
            files=files,latent_metrics=records,anchor_tensor_parity=checks,execution_precision=require_precision(device,contract)))
    teacher_all16={ident:read_json(args.output/'inference/pretrained'/ident/'done.json')['latent_metrics'] for ident in planned}
    for ident in sorted(set(contract['plan']['test_ids'])-set(planned)):
        _,ref,flow,valid=tensors(pairs[ident],device)
        latent=tensor_load(args.base_evaluation/'latents'/ident/'pretrained.pt')['latent']
        values={k:float(v) for k,v in objective(latent[None].to(device),ref,flow,valid,.1).items()}
        metric_parity(values,base['variants']['pretrained'][ident],f'pretrained/reused/{ident}')
        teacher_all16[ident]=values
    immutable_json(args.output/'teacher_test16_metric_parity.json',dict(status='passed',contract_sha256=digest,
        samples=teacher_all16,fresh_inferences=planned,previous_tensor_reuse=sorted(set(contract['plan']['test_ids'])-set(planned)),
        scope='All16 teacher metric comparisons;14 newly inferred tensors and2 SHA-bound original tensors'))
    verify_contract(args)
    immutable_json(args.output/'inference_complete.json',dict(status='completed',contract_sha256=digest,
        recomputed_ids=planned,teacher_inferences=14,old_causal_independent_prediction_checks=14,
        independent_checkpoint_load_scope='One shared independently loaded old checkpoint per invocation with unfinished predictions;14 output comparisons, not14 checkpoint loads',
        all16_teacher_latent_metrics='14 fresh outputs checked against prior metrics; two un-recomputed old tests retain SHA-bound prior tensors/metrics',
        execution_precision_start=start_precision,execution_precision_end=require_precision(device,contract)))
    empty_cache(device)


def anchor_pixel_parity(entry, old):
    checks={}
    for label in MODELS:
        previous.check_historical_pixels(entry['variants'][label],old['variants'][label])
        for target in TARGETS:
            a=entry['variants'][label]['lpips_per_frame'][target];b=old['variants'][label]['lpips_per_frame'][target]
            require(len(a)==len(b)==120,'Anchor LPIPS coverage differs')
            require(all(math.isclose(x,y,abs_tol=1e-5,rel_tol=1e-5) for x,y in zip(a,b)),'Anchor per-frame LPIPS differs')
        checks[label]=dict(passed=True,regions=['full120','prefix102','tail18'],lpips_frames=120,atol=1e-5,rtol=1e-5)
    for key in KEYS:
        require(math.isclose(entry['reference_vs_source_rgb_secondary'][key],old['reference_vs_source_rgb_secondary'][key],abs_tol=1e-5,rel_tol=1e-5),f'Anchor reference/source differs: {key}')
    return checks


@torch.no_grad()
def decode(args):
    contract,rows,base,_,digest=verify_contract(args)
    require(read_json(args.output/'inference_complete.json')['contract_sha256']==digest,'Inference incomplete')
    planned=contract['plan']['recomputed_ids'];todo=[]
    for ident in planned:
        if valid_receipt(args.output/'decoded'/ident,digest,media=True) is None:todo.append(ident)
    if not todo:return
    device=select_device(args.device);precision_start=require_precision(device,contract)
    perceptual,lpips_identity=previous.load_lpips(device)
    vae=load_video_vae(args.base_model,device).eval().requires_grad_(False)
    decoder_identity=dict(weights=vae_fingerprint(args.base_model),implementation_sha256=sha256(Path(inspect.getfile(type(vae)))),
        recipe='data_utils.load_video_vae + decode_latent, original temporal windows, FP32, spatial tile256/overlap64')
    require(decoder_identity==contract['decoder_fingerprint'],'Decoder changed')
    require(lpips_identity==contract['lpips_fingerprint'],'LPIPS changed')
    before=frozen_versions(vae,perceptual)
    for ident in todo:
        dst=args.output/'decoded'/ident
        archive_incomplete(dst,args.output,'Decode/metrics/media audit interrupted before done.json; retained before retry');dst.mkdir(parents=True)
        source_dir=args.output/'latents'/ident;latent_receipt=valid_receipt(source_dir,digest)
        require(latent_receipt is not None,'Latents incomplete')
        pair_path=Path(rows[ident]['path']);pair_path=pair_path if pair_path.is_absolute() else args.pairs/pair_path
        pair=tensor_load(pair_path);source=pair['source_pixels']
        require(tuple(source.shape)==(120,384,672,3) and source.dtype==torch.uint8,'Source RGB differs')
        pixels={};timings={};codecs={}
        for label in LABELS:
            latent=tensor_load(source_dir/f'{label}.pt')['latent']
            if label=='reference':require(torch.equal(latent,pair['reference'].float()),'Reference differs')
            synchronize(device);tick=time.perf_counter();rgb=decode_latent(vae,latent,device,keep_frames=120);synchronize(device)
            require(tuple(rgb.shape)==(120,384,672,3) and rgb.dtype==torch.uint8,'Decoded RGB differs')
            pixels[label]=rgb;timings[label]=time.perf_counter()-tick;codecs[label]=write_mp4(rgb,dst/f'{label}.mp4',fps=24.)
        entry=dict(sample_id=ident,split='test',frame_count=120,decode_seconds=timings,video_codecs=codecs,variants={},
            metric_domain='Uncompressed uint8 RGB; PSNR/temporal[0,1], LPIPS[-1,1]',historical_pixel_metrics_verified=[])
        for label in MODELS:
            entry['variants'][label]=previous.pixel_metrics(pixels[label],pixels['reference'],source,perceptual,device,4)
        entry['reference_vs_source_rgb_secondary']=metric(pixels['reference'],source)
        entry['reference_vs_source_rgb_secondary']['lpips_alex_v0_1']=statistics.mean(previous.lpips_frames(perceptual,pixels['reference'],source,device,4))
        anchor_checks={}
        if ident==contract['plan']['anchor_id']:
            anchor_checks=anchor_pixel_parity(entry,base['pixel_metrics'][ident])
            entry['historical_pixel_metrics_verified']=list(MODELS)
        comparisons(pixels,dst,fps=24.,crf=16,comparison_video=True)
        media=previous.audit_media(dst)
        require(frozen_versions(vae,perceptual)==before and all(p.grad is None for m in (vae,perceptual) for p in m.parameters()),'Frozen decoder/LPIPS changed')
        precision_end=require_precision(device,contract)
        immutable_json(dst/'metrics.json',entry)
        files={name:{k:record[k] for k in ('sha256','bytes')} for name,record in media.items()}
        files['metrics.json']=file_record(dst/'metrics.json')
        immutable_json(dst/'done.json',dict(status='completed',contract_sha256=digest,sample_id=ident,
            files=files,media_audit=media,anchor_pixel_parity=anchor_checks,
            latent_receipt_sha256=sha256(source_dir/'done.json'),decoder_fingerprint=decoder_identity,lpips_fingerprint=lpips_identity,
            execution_precision_start=precision_start,execution_precision_end=precision_end,frozen_modules_unchanged=True))
        print(json.dumps(dict(stage='pixel_evaluated',sample_id=ident,full_frames=120,anchor=bool(anchor_checks))),flush=True)
        del pixels,pair,source
    require(sha256(Path(inspect.getfile(type(vae))))==decoder_identity['implementation_sha256'],'Decoder implementation changed during decoding')
    verify_contract(args);del vae,perceptual;empty_cache(device)


def validate_entry(entry):
    require(entry['frame_count']==120 and set(entry['variants'])==set(MODELS),'Incomplete pixel entry')
    for value in entry['variants'].values():
        for region in (value,value['prefix102'],value['tail18']):
            for target in TARGETS:
                require(all(math.isfinite(float(region[target][k])) for k in KEYS),'Nonfinite metric')
        for target in TARGETS:
            values=value['lpips_per_frame'][target]
            require(len(values)==120 and all(math.isfinite(float(v)) for v in values),'LPIPS coverage differs')
            for lo,hi,region in ((0,120,value),(0,102,value['prefix102']),(102,120,value['tail18'])):
                require(math.isclose(statistics.mean(values[lo:hi]),region[target]['lpips_alex_v0_1'],abs_tol=1e-12,rel_tol=1e-12),'LPIPS average mismatch')


def aggregate(group):
    def regional(region):
        def metrics(ident,label):
            value=group[ident]['variants'][label]
            return value if region is None else value[region]
        means={label:{target:{key:statistics.mean(float(metrics(i,label)[target][key]) for i in group) for key in KEYS} for target in TARGETS} for label in MODELS}
        deltas={label:{target:{key:means['conv3d_hybrid'][target][key]-means[label][target][key] for key in KEYS} for target in TARGETS} for label in MODELS if label!='conv3d_hybrid'}
        return dict(mean=means,conv3d_hybrid_minus_baseline=deltas)
    result=dict(sample_ids=list(group),count=len(group),**regional(None))
    result['regions']={name:regional(name) for name in ('prefix102','tail18')}
    return result


def finalize(args):
    contract,_,base,_,digest=verify_contract(args);plan=contract['plan'];fresh={};media={};receipts={}
    for ident in plan['recomputed_ids']:
        directory=args.output/'decoded'/ident
        receipt=valid_receipt(directory,digest,media=True);require(receipt is not None,f'Missing decoded clip: {ident}')
        require(receipt['latent_receipt_sha256']==sha256(args.output/'latents'/ident/'done.json'),'Latent receipt changed')
        require(receipt['decoder_fingerprint']==contract['decoder_fingerprint'] and receipt['lpips_fingerprint']==contract['lpips_fingerprint'],'Decode model fingerprints differ')
        entry=read_json(directory/'metrics.json');validate_entry(entry)
        fresh[ident]=entry;media[ident]=receipt['media_audit'];receipts[ident]=file_record(directory/'done.json')
    anchor=plan['anchor_id'];anchor_checks=anchor_pixel_parity(fresh[anchor],base['pixel_metrics'][anchor])
    combined={ident:(fresh[ident] if ident in plan['new_test_ids'] else copy.deepcopy(base['pixel_metrics'][ident])) for ident in plan['test_ids']}
    legacy={ident:copy.deepcopy(base['pixel_metrics'][ident]) for ident in plan['legacy_ids']}
    for entry in [*combined.values(),*legacy.values()]:validate_entry(entry)
    require(len(combined)==16 and len(fresh)==14 and len(media)==14,'Wrong coverage')
    raw_count=sum(name.endswith('.mp4') for files in media.values() for name in files)
    require(raw_count==98,'Expected98 fully audited new raw MP4s')
    groups={'all_16_test':combined,'new_13_test':{i:combined[i] for i in plan['new_test_ids']},
            'previous_3_test':{i:combined[i] for i in plan['reused_test_ids']},'legacy_1':legacy}
    precision={ident:{k:read_json(args.output/'decoded'/ident/'done.json')[k] for k in ('execution_precision_start','execution_precision_end')} for ident in plan['recomputed_ids']}
    summary=dict(status='completed',decode_complete=True,contract_sha256=digest,plan=plan,
        new_test_ids=plan['new_test_ids'],old_test_ids=plan['reused_test_ids'],
        anchor_id=plan['anchor_id'],all_test_ids=plan['test_ids'],legacy_ids=plan['legacy_ids'],
        selected_step=1800,selected_checkpoint_sha256=SELECTED_SHA,
        source_evaluation_sha256=sha256(args.base_evaluation/'summary.json'),comparison_order=list(LABELS),
        frames=120,fps=24,seconds_per_video=5,pixel_metric_scope='All16 reused test videos:13 newly decoded plus3 immutable prior records. Legacy1 is reported separately; anchor recomputation is never double-counted.',
        metric_domain='Uncompressed decoded uint8 RGB before any MP4 compression',
        test_reuse='Previously inspected test set; no new blind test; no training/checkpoint selection in this extension',
        causal_scope='Fixed input latents only; original H3 decoder remains noncausal; no new causality claim',
        pixel_metrics=combined,legacy_pixel_metrics=legacy,anchor_recomputed_pixel_metrics=fresh[anchor],
        anchor_pixel_parity=anchor_checks,anchor_counted_via_original_record=True,
        pixel_comparison=dict(scope='Equal video weighting; PSNR is mean per-video PSNR, not pooled-MSE PSNR',
            groups={name:aggregate(group) for name,group in groups.items()}),
        full_decoded_mp4_files=raw_count,media_audit=media,clip_receipt_fingerprints=receipts,
        decoder_fingerprint=contract['decoder_fingerprint'],lpips_fingerprint=contract['lpips_fingerprint'],execution_precision=precision,
        source_contract_unchanged=True)
    immutable_json(args.output/'summary.json',summary)
    print(json.dumps(dict(status='completed',test_clips=16,new_clips=13,anchor_recomputed=1,legacy_separate=1,full_decoded_mp4_files=98)),flush=True)


def check(args):
    contract,_,_,_=context(args)
    result=dict(status='inputs_verified',selected_step=1800,new_clips=13,anchor=contract['plan']['anchor_id'])
    if (args.output/'contract.json').exists():
        require(read_json(args.output/'contract.json')==contract,'Contract changed')
        digest=sha256(args.output/'contract.json')
        result['completed_decode_clips']=[i for i in contract['plan']['recomputed_ids'] if valid_receipt(args.output/'decoded'/i,digest) is not None]
        result['final_summary_exists']=(args.output/'summary.json').exists()
    print(json.dumps(result),flush=True)


def main():
    root=Path(os.environ.get('EXP_ROOT',Path(__file__).resolve().parents[1]))
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('check','prepare','infer','decode','finalize','all'),default='all')
    p.add_argument('--output',type=Path,default=Path(os.environ.get('H3_CONV3D_TEST16_EVALUATION',root/'upscaler_conv3d_hybrid/evaluation_test16_extension')))
    p.add_argument('--run',type=Path,default=root/'upscaler_conv3d_hybrid/kd_pixel_steps2000')
    p.add_argument('--old-run',type=Path,default=root/'upscaler_ultraflash/ewarp_steps2000')
    p.add_argument('--cmn-run',type=Path,default=root/'upscaler_cmn/ewarp_steps10000')
    p.add_argument('--hybrid-run',type=Path,default=root/'upscaler_hybrid/cmn_kd_pixel_steps2000')
    p.add_argument('--base-evaluation',type=Path,default=root/'upscaler_conv3d_hybrid/evaluation')
    p.add_argument('--cmn-evaluation',type=Path,default=root/'upscaler_cmn/evaluation')
    p.add_argument('--hybrid-evaluation',type=Path,default=root/'upscaler_hybrid/evaluation')
    p.add_argument('--pairs',type=Path,default=root/'upscaler_ultraflash/pairs')
    p.add_argument('--initial',type=Path,default=root/'upscaler_ultraflash/ewarp_steps2000/best.pt')
    p.add_argument('--upstream',type=Path,default=Path(os.environ.get('H3_UPSTREAM_DIR',root/'upstream')))
    p.add_argument('--weights',type=Path,default=Path(os.environ.get('H3_UPSCALER_WEIGHTS',root/'assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors')))
    p.add_argument('--base-model',default=os.environ.get('H3_BASE_MODEL'))
    p.add_argument('--device',default='npu:0')
    args=p.parse_args()
    if not args.base_model:
        p.error("Set H3_BASE_MODEL or pass --base-model")
    torch.set_num_threads(4)
    if args.stage=='check':check(args);return
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.extension.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stages=('prepare','infer','decode','finalize') if args.stage=='all' else (args.stage,)
        for stage in stages:globals()[stage](args)


if __name__=='__main__':main()
