"""Full120 dev-gated evaluation of fixed v5/control/motion checkpoints.

The 16 development clips alone determine the evaluation gate. The checkpoints
are already selected by their training runs; this script never selects another
checkpoint. Motion must pass the dev gate before test16+legacy3 are loaded.

All evaluation RGB metrics use the assembled, uncompressed uint8 decoder output
mapped to[0,1]. Training uses unclipped float RGB on sparse native regions, so
these evaluation numbers are deliberately a different, declared domain. The
original frozen decoder is noncausal. Four columns only: v5/control/motion/HR.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
from PIL import Image

from data_utils import (decode_latent, empty_cache, load_video_vae, select_device,
                        synchronize, tensor_load, write_mp4)
from decode_results import comparisons
from evaluate_h3_cmn import audit_video, file_record, module_source, read_json, require, vae_fingerprint
from evaluate_h3_conv3d_hybrid import load_lpips, lpips_frames
from evaluate_h3_conv3d_test16 import archive_incomplete, frozen_versions, immutable_json
from h3_conv3d_hybrid import load_student_checkpoint
from h3_hybrid_pixel import PixelSupervisor
from h3_v5_motion import rgb_motion_losses, target_residual_ewarp
from train_h3_cmn import causal_checks, objective
from train_h3_v5_motion import execution_precision
from train_upscaler_trial import atomic_torch_save, sha256
from train_upscaler_ultraflash import load_pairs, tensors

V5_SHA = 'c7aaaa396d877a84c3fed2a5536189b7ed5e61226a82ef1a9e6be1e831f7f914'
LABELS = ('v5', 'control', 'motion', 'reference')
ARMS = ('control', 'motion')
THRESHOLD = .05
SOURCE_NAMES = ('evaluate_h3_v5_motion.py', 'train_h3_v5_motion.py', 'h3_v5_motion.py',
    'h3_conv3d_hybrid.py', 'causal_upscaler.py', 'h3_hybrid_pixel.py', 'train_h3_cmn.py',
    'h3_cmn.py', 'train_upscaler_ultraflash.py', 'ultraflash_ewarp.py', 'data_utils.py',
    'prepare_data.py', 'train_upscaler_trial.py', 'evaluate_h3_cmn.py',
    'evaluate_h3_conv3d_hybrid.py', 'evaluate_h3_conv3d_test16.py',
    'evaluate_causal_upscaler_pixels.py', 'decode_results.py')


def frame_regions():
    blend = sorted(i for block in range(1, 7) for i in range(17*block, 17*block+5))
    require(len(blend) == 30 and max(blend) == 106, 'Unexpected H3 overlap geometry')
    return {'full120': list(range(120)), 'prefix102': list(range(102)),
            'tail18': list(range(102, 120)), 'blend': blend,
            'nonblend': [i for i in range(120) if i not in set(blend)]}



def transition_pair_groups():
    """Partition all119 actual RGB transitions, including blend entry/exit."""
    blend=set(frame_regions()['blend'])
    pairs=[(i,i+1) for i in range(119)]
    groups={'blend_touching_pairs':[[a,b] for a,b in pairs if a in blend or b in blend],
            'nonblend_only_pairs':[[a,b] for a,b in pairs if a not in blend and b not in blend]}
    require(len(groups['blend_touching_pairs'])==36 and len(groups['nonblend_only_pairs'])==83,
            'Unexpected transition coverage')
    return groups


def json_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def check_selected_run(path, arm):
    summary = read_json(path/'summary.json'); config = read_json(path/'config.json')
    contract = read_json(path/'contract.json')
    require(summary['status'] == 'completed' and not config['preflight'], f'{arm}: training incomplete/preflight')
    require(summary['config'] == config == contract['config'] and summary['contract'] == contract,
            f'{arm}: inconsistent training contract')
    require(config['arm'] == arm and summary['completed_updates'] == config['steps'] == 2000,
            f'{arm}: wrong arm/update count')
    require(summary['test_opened'] is False, f'{arm}: test must remain unopened before full-dev gate')
    require(summary['train_count'] == 128 and summary['dev_count'] == 16, 'Wrong train/dev scope')
    for flag in ('frozen_modules_unchanged', 'source_contract_unchanged', 'execution_precision_unchanged'):
        require(summary[flag] is True, f'{arm}: {flag} failed')
    require(summary['execution_precision_start'] == summary['execution_precision_end'] == contract['execution_precision'],
            f'{arm}: precision changed')
    require(summary['initial_causality']['passed'] and summary['causality']['passed'], f'{arm}: training causal audit failed')
    require(contract['initial_sha256'] == config['initial_sha256'] == V5_SHA, 'Different v5 starting checkpoint')
    for name, digest in contract['source_sha256'].items():
        require(sha256(module_source(name)) == digest, f'{arm}: frozen training source changed: {name}')
    values = summary['validation']; baseline = values[0]['mean']
    require([v['step'] for v in values] == list(range(0, 2001, config['validate_every'])),
            f'{arm}: unexpected dev checkpoint schedule')
    require(summary['selection_guard_baseline'] == baseline, f'{arm}: changed guard baseline')
    eligible = []
    for v in values:
        mean = v['mean']
        okay = (mean['global_score'] <= 1.01*baseline['global_score'] and
                mean['pixel_l1'] <= 1.02*baseline['pixel_l1'] and mean['lpips'] <= 1.02*baseline['lpips'])
        if v['step']:
            require(v['eligible_global_guard'] == okay, f'{arm}: sparse-dev eligibility differs')
        if okay: eligible.append(v)
    chosen = min(eligible, key=lambda v: v['mean']['selection_score'])
    require(chosen['step'] == summary['best_step'] and chosen['mean']['selection_score'] == summary['best_validation_score'],
            f'{arm}: best checkpoint is not the first eligible training-dev minimum')
    best = path/'best.pt'
    require(sha256(best) == summary['best_checkpoint_sha256'], f'{arm}: checkpoint SHA differs')
    saved = tensor_load(best)
    require(saved['contract'] == contract and saved['step'] == summary['best_step'], f'{arm}: checkpoint contract/step differs')
    del saved
    steps = json_lines(path/'steps.jsonl')
    require([v['step'] for v in steps] == list(range(1, 2001)), f'{arm}: noncontiguous updates')
    require(all(math.isfinite(float(v[k])) for v in steps for k in ('loss','pixel_l1','lpips','motion_l1','delta_l1')), 'Nonfinite training metrics')
    return summary, steps


def context(args):
    """Read metadata for all splits; never load a held-out tensor here."""
    ready = read_json(args.pairs/'READY.json'); plan = read_json(args.pairs/'plan.json')
    require(ready['status'] == 'ready' and ready['plan_sha256'] == sha256(args.pairs/'plan.json'), 'Dataset plan changed')
    rows = {}; manifests = {}
    for split, count in (('train',128), ('dev',16), ('test',16), ('legacy',3)):
        path = args.pairs/f'{split}.jsonl'; entries = json_lines(path)
        require(len(entries) == count and all(r['split'] == split for r in entries), 'Manifest count/split differs')
        require(sha256(path) == ready['manifests'][split]['sha256'], 'Manifest SHA changed')
        manifests[split] = file_record(path)
        for row in entries:
            require(row['sample_id'] not in rows, 'Sample split collision')
            rows[row['sample_id']] = row
    require(len(rows) == 163 and len({r['source_video_sha256'] for r in rows.values()}) == 163,
            'Source SHA split collision (metadata only)')
    plan_rows = {r['sample_id']: r for r in plan['entries']}
    require(set(plan_rows) == set(rows), 'Plan/manifest IDs differ')
    for ident, row in rows.items():
        require(row['split'] == plan_rows[ident]['split'] and row['source_video_sha256'] == plan_rows[ident]['source_video_sha256'], 'Plan row changed')
    runs = {}; schedules = {}
    for arm, path in (('control', args.control_run), ('motion', args.motion_run)):
        runs[arm], schedules[arm] = check_selected_run(path, arm)
        require(runs[arm]['contract']['plan_sha256'] == ready['plan_sha256'], 'Training plan differs')
        for split in ('train','dev'):
            require(runs[arm]['contract'][f'{split}_manifest_sha256'] == manifests[split]['sha256'], 'Train/dev manifest changed')
    for left, right in zip(schedules['control'], schedules['motion']):
        require(all(left[k] == right[k] for k in ('step','sample_id','lr','spec')), 'Matched schedules differ')
        require(rows[left['sample_id']]['split'] == 'train', 'Nontraining update sample')
    for key in ('steps','lr','min_lr','warmup','validate_every','save_every','seed','batch_size','full_every','kd_weight','warp_weight','l1_weight','lpips_weight'):
        require(runs['control']['config'][key] == runs['motion']['config'][key], 'Matched optimizer/recipe field differs: '+key)
    baseline = read_json(args.baseline_run/'summary.json')
    require(baseline['status'] == 'completed' and baseline['best_step'] == 1800 and baseline['best_checkpoint_sha256'] == V5_SHA,
            'Named v5 baseline changed')
    require(sha256(args.baseline_run/'best.pt') == V5_SHA, 'v5 checkpoint SHA changed')
    require(baseline['contract']['plan_sha256'] == ready['plan_sha256'], 'v5 baseline data plan differs')
    for split in ('train','dev'):
        require(baseline['contract'][f'{split}_manifest_sha256'] == manifests[split]['sha256'], 'v5 baseline manifest differs')
    for name, digest in baseline['contract']['source_sha256'].items():
        require(sha256(module_source(name)) == digest, 'v5 original training source changed: '+name)
    precision = runs['control']['contract']['execution_precision']
    require(precision == runs['motion']['contract']['execution_precision'] == baseline['contract']['execution_precision'],
            'Baseline/control/motion precision contracts differ')
    require(precision['conv_allow_hf32'] is True and precision['matmul_allow_hf32'] is False, 'Default evaluation must retain training HF32 mode')
    decoder = baseline['contract']['decoder_fingerprint']; perceptual = baseline['contract']['lpips_fingerprint']
    for run in runs.values():
        for key in ('weights','implementation_sha256'):
            require(run['contract']['decoder_fingerprint'][key] == decoder[key], 'Frozen decoder identity changed')
        require(run['contract']['lpips_fingerprint'] == perceptual, 'LPIPS training identity changed')
        original = run['contract']['original_upscaler_fingerprint']
        require(sha256(args.weights) == original['base_weights_sha256'] and sha256(args.upstream/'open_h3/upscaler.py') == original['loader_sha256'], 'Upscaler assets changed')
        require(sha256(args.upstream/'open_h3/upscaler_node.py') == original['loaded_node_sha256'], 'Actual upscaler node changed')
    require(vae_fingerprint(args.base_model) == decoder['weights'], 'Decoder files changed')
    import lpips
    lpips_files = {'backbone': Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth',
                   'linear': Path(lpips.__file__).resolve().parent/'weights/v0.1/alex.pth',
                   'implementation': Path(inspect.getfile(lpips.LPIPS))}
    for label, path in lpips_files.items():
        key = {'backbone':'backbone_sha256','linear':'linear_sha256','implementation':'implementation_sha256'}[label]
        require(sha256(path) == perceptual[key], 'LPIPS asset changed')
    selected = {'v5': args.baseline_run/'best.pt', 'control': args.control_run/'best.pt', 'motion': args.motion_run/'best.pt'}
    run_paths = {'v5': args.baseline_run, 'control': args.control_run, 'motion': args.motion_run}
    contract = dict(schema_version=1, selected_checkpoints={label:dict(path=str(p),**file_record(p),
        selected_step=1800 if label=='v5' else runs[label]['best_step']) for label,p in selected.items()},
        training_summaries={label:dict(path=str(path/'summary.json'),**file_record(path/'summary.json')) for label,path in run_paths.items()},
        training_step_logs={arm:file_record(path/'steps.jsonl') for arm,path in (('control',args.control_run),('motion',args.motion_run))},
        plan_sha256=ready['plan_sha256'], ready_sha256=sha256(args.pairs/'READY.json'), manifests=manifests,
        source_sha256={name:sha256(Path(__file__) if name==Path(__file__).name else module_source(name)) for name in SOURCE_NAMES},
        decoder_fingerprint=decoder,lpips_fingerprint=perceptual,lpips_files={k:dict(path=str(p),**file_record(p)) for k,p in lpips_files.items()},
        execution_precision=precision,metric_domain='Assembled uncompressed decoder uint8 RGB /255 in[0,1]; LPIPS maps these same pixels to[-1,1]. Training uses unclipped float RGB sparse regions and is not numerically the same domain.',
        frame_regions=frame_regions(),temporal_transition_pair_groups=transition_pair_groups(),motion_threshold=THRESHOLD,
        temporal_transition_scope='Extra diagnostic partition of all119 RGB pairs:36 touching any blend frame, including12 entering/exiting edges, versus83 nonblend-only. Does not change frame-region metrics or the full120 gate.',
        motion_l1_definition='Detached GT mean-channel abs adjacent RGB change>=.05; same ROI on both pair endpoints; divide2*3*selected pair-pixels. Repeated endpoints count repeatedly; average clips equally.',
        delta_l1_definition='Mean abs((pred[t]-pred[t-1])-(GT[t]-GT[t-1])) over every actual-adjacent pair and RGB channel/pixel; no flow alignment.',
        temporal_difference_mse_definition='Mean squared((pred[t]-pred[t-1])-(GT[t]-GT[t-1])) on exactly the same actual-adjacent RGB pairs; full120 uses119 pairs.',
        latent_metric_scope='Legacy objective uses valid_frames36; target-residual helper uses supplied valid-pair masks on full37 latents. Raw and residual eWarp are separately retained and do not gate test.',
        full_dev_gate={'global_score_factor_max':1.01,'pixel_l1_factor_max':1.02,'lpips_factor_max':1.02,
                       'motion_l1_strictly_below_v5':True,'delta_l1_strictly_below_v5':True,'test_unlock_arm':'motion'},
        split_ids={split:sorted(i for i,r in rows.items() if r['split']==split) for split in ('dev','test','legacy')},
        causal_dev_ids=sorted(i for i,r in rows.items() if r['split']=='dev')[:4],
        evaluation_scope='No checkpoint selection; test16+legacy3 tensor loads only after motion passes full120 dev. Control remains a fixed comparator even if its gate fails.',
        causal_scope='Each new selected arm gets4dev x9default-HF32 checks; separate long163 HF32off audit belongs to caller pipeline, not this default math scope. Frozen decoder is not causal.',
        comparison_order=list(LABELS),frames=120,fps=24,pixels=[672,384],output=str(args.output))
    return contract,rows,runs


def verify_contract(args):
    current,rows,runs=context(args)
    require(read_json(args.output/'contract.json') == current, 'Evaluation source/input contract changed')
    return current,rows,runs,sha256(args.output/'contract.json')


def require_precision(device, contract):
    actual=execution_precision(device)
    require(actual == contract['execution_precision'], 'Math mode differs: do not change HF32 during evaluation')
    return actual


def valid_receipt(directory, digest):
    path=directory/'done.json'
    if not path.exists(): return None
    receipt=read_json(path)
    require(receipt['status']=='completed' and receipt['contract_sha256']==digest and receipt['sample_id']==directory.name, 'Receipt contract/status/sample differs')
    if directory.parents[2].name=='predictions':
        expected={'latent.pt'} | ({'causality.json'} if receipt['causality'] is not None else set())
    elif directory.parents[1].name=='baseline_cache':
        expected={'rgb.pt'}
    elif directory.parents[1].name=='decoded':
        expected={name+'.mp4' for name in (*LABELS,'comparison')}|{'comparison.png','metrics.json'}
    else:
        raise AssertionError('Unknown evaluation receipt location')
    require(set(receipt['files'])==expected, 'Receipt required file coverage differs')
    for name,record in receipt['files'].items():
        require(Path(name).name==name and file_record(directory/name)==record, 'Receipt file changed')
    return receipt


def load_split(args, contract, rows, split):
    pairs=load_pairs(args.pairs/f'{split}.jsonl')
    by_id={p['sample_id']:p for p in pairs}
    require(set(by_id)==set(contract['split_ids'][split]), 'Loaded split IDs differ')
    for ident,pair in by_id.items():
        require(pair['split']==split and pair['pair_sha256']==rows[ident]['sha256'], 'Loaded pair differs from manifest')
    return by_id


@torch.no_grad()
def infer_split(args, contract, pairs, split, device, digest):
    """One model per arm, sequentially; every prediction binds fixed checkpoint."""
    for label in LABELS[:-1]:
        todo=[ident for ident in contract['split_ids'][split] if valid_receipt(args.output/'predictions'/split/label/ident,digest) is None]
        if not todo:continue
        checkpoint=Path(contract['selected_checkpoints'][label]['path'])
        model=load_student_checkpoint(checkpoint,args.upstream,args.weights,device).eval().requires_grad_(False)
        before=frozen_versions(model);precision_start=require_precision(device,contract)
        for ident in todo:
            directory=args.output/'predictions'/split/label/ident
            archive_incomplete(directory,args.output,'Prediction/causal check interrupted before done.json; preserved before retry')
            directory.mkdir(parents=True)
            pair=pairs[ident];draft,reference,flow,valid=tensors(pair,device)
            causal=None
            if split=='dev' and label in ARMS and ident in contract['causal_dev_ids']:
                causal=causal_checks(model,draft)
                causal['absolute_max_abs']=max(v['max_abs'] for v in causal['checks'].values())
                causal['passed']=bool(causal['passed'] and causal['absolute_max_abs']<=1e-4)
                immutable_json(directory/'causality.json',causal)
                require(len(causal['checks'])==9 and causal['passed'], f'{label}/{ident}: default production-shape causal audit failed')
                model.reset_stream()
            pred=model(draft)
            require(tuple(pred.shape)==(1,24,37,24,42) and bool(torch.isfinite(pred).all()), 'Invalid prediction')
            latent_metrics={k:float(v) for k,v in objective(pred,reference,flow,valid,.1).items()}
            latent_metrics.update({k:float(v) for k,v in target_residual_ewarp(pred,reference,flow,valid).items()})
            require(all(math.isfinite(v) for v in latent_metrics.values()), 'Nonfinite latent metric')
            atomic_torch_save(directory/'latent.pt',dict(latent=pred[0].cpu(),sample_id=ident,split=split,pair_sha256=pair['pair_sha256']))
            files={'latent.pt':file_record(directory/'latent.pt')}
            if causal is not None:files['causality.json']=file_record(directory/'causality.json')
            require(frozen_versions(model)==before and all(p.grad is None for p in model.parameters()), 'Frozen student changed')
            immutable_json(directory/'done.json',dict(status='completed',contract_sha256=digest,sample_id=ident,split=split,variant=label,
                checkpoint_sha256=contract['selected_checkpoints'][label]['sha256'],pair_sha256=pair['pair_sha256'],files=files,
                latent_metrics=latent_metrics,causality=causal,execution_precision_start=precision_start,
                execution_precision_end=require_precision(device,contract),frozen_student_unchanged=True,
                load_scope='Independent checkpoint-loaded model shared across unfinished clips in this invocation; clip count is not checkpoint-load count'))
            print(json.dumps(dict(stage='prediction',split=split,variant=label,sample_id=ident)),flush=True)
            del draft,reference,flow,valid,pred
        node=model.adapter.node;node.MODEL_CACHE.clear();del model,node;empty_cache(device)


def target_mask_metadata(reference, ids):
    ref=reference[ids].float().permute(3,0,1,2)[None]/255.
    left=[i for i in range(len(ids)-1) if ids[i+1]-ids[i]==1];right=[i+1 for i in left]
    delta=ref[:,:,right]-ref[:,:,left]
    mask=delta.abs().mean(1)>=THRESHOLD
    return dict(adjacent_pairs=len(left),motion_pair_pixels=int(mask.sum()),total_pair_pixels=mask.numel(),
                motion_mask_sha256=hashlib.sha256(mask.cpu().contiguous().numpy().tobytes()).hexdigest(),
                rgb_frame_indices=ids,threshold=THRESHOLD)


def rgb_metrics(output, reference, perceptual, device, regions):
    lpips_values=lpips_frames(perceptual,output,reference,device,4)
    result={}
    for region,ids in regions.items():
        pred=output[ids].float().permute(3,0,1,2)[None]/255.
        target=reference[ids].float().permute(3,0,1,2)[None]/255.
        l1=float((pred-target).abs().mean());mse=float((pred-target).square().mean())
        lpips=statistics.mean(lpips_values[i] for i in ids)
        motion=rgb_motion_losses(pred,target,ids,change_threshold=THRESHOLD)
        left=[i for i in range(len(ids)-1) if ids[i+1]-ids[i]==1];right=[i+1 for i in left]
        delta=(pred[:,:,right]-pred[:,:,left])-(target[:,:,right]-target[:,:,left])
        temporal_mse=float(delta.square().mean()) if left else 0.
        rec={k:float(v) for k,v in motion.items()}
        rec['temporal_difference_mse']=temporal_mse
        rec.update(pixel_l1=l1,mse=mse,psnr_db=-10*math.log10(max(mse,1e-12)),lpips=lpips,
                   global_score=l1+.2*lpips,frame_count=len(ids))
        require(all(math.isfinite(v) for v in rec.values()), 'Nonfinite RGB metric')
        result[region]=rec
    return dict(regions=result,lpips_per_frame=lpips_values)



def rgb_transition_metrics(output, reference, pair_groups):
    """Fixed target ROI, pair-weighted endpoints; explicit edges never omitted."""
    pred=output.float().permute(3,0,1,2)[None]/255.
    target=reference.float().permute(3,0,1,2)[None].detach()/255.
    result={}
    for name,pairs in pair_groups.items():
        require(bool(pairs) and all(b-a==1 and 0<=a<b<120 for a,b in pairs), 'Nonadjacent RGB transition')
        left=[a for a,b in pairs];right=[b for a,b in pairs]
        target_delta=target[:,:,right]-target[:,:,left]
        mask=target_delta.abs().mean(1,keepdim=True)>=THRESHOLD
        endpoint_error=(pred[:,:,left]-target[:,:,left]).abs()+(pred[:,:,right]-target[:,:,right]).abs()
        residual=(pred[:,:,right]-pred[:,:,left])-target_delta
        motion_l1=(endpoint_error*mask).sum()/(mask.sum()*2*3).clamp_min(1)
        values=dict(motion_l1=float(motion_l1),delta_l1=float(residual.abs().mean()),
                    temporal_difference_mse=float(residual.square().mean()),motion_roi_fraction=float(mask.float().mean()),
                    adjacent_pair_count=len(pairs),motion_pair_pixels=int(mask.sum()),total_pair_pixels=mask.numel())
        require(all(math.isfinite(v) for v in values.values()), 'Nonfinite transition metric')
        result[name]=dict(metrics=values,motion_mask_sha256=hashlib.sha256(mask.cpu().contiguous().numpy().tobytes()).hexdigest())
    return result


def audit_media(directory):
    media={}
    for label in (*LABELS,'comparison'):
        width,height=(672*len(LABELS),412) if label=='comparison' else (672,384)
        media[label+'.mp4']=audit_video(directory/(label+'.mp4'),width,height)
    path=directory/'comparison.png'
    with Image.open(path) as img:
        require(img.format=='PNG' and img.size==(2688,1236), 'Wrong four-column contact-sheet shape');img.verify()
    with Image.open(path) as img:img.load()
    media['comparison.png']=dict(passed=True,width=2688,height=1236,**file_record(path))
    return media


@torch.no_grad()
def decode_split(args, contract, pairs, split, device, digest):
    todo=[ident for ident in contract['split_ids'][split] if valid_receipt(args.output/'decoded'/split/ident,digest) is None]
    if not todo:return
    vae=load_video_vae(args.base_model,device).eval().requires_grad_(False);perceptual,lpips_identity=load_lpips(device)
    decoder_identity=dict(weights=vae_fingerprint(args.base_model),implementation_sha256=sha256(Path(inspect.getfile(type(vae)))))
    for key in ('weights','implementation_sha256'):require(decoder_identity[key]==contract['decoder_fingerprint'][key], 'Loaded decoder differs')
    for actual,expected in ((lpips_identity['backbone']['sha256'],'backbone_sha256'),(lpips_identity['linear_weights']['sha256'],'linear_sha256'),(lpips_identity['implementation_sha256'],'implementation_sha256')):
        require(actual==contract['lpips_fingerprint'][expected], 'Loaded LPIPS differs')
    supervisor=PixelSupervisor(vae,perceptual,device,checkpoint_decode=False)
    native_ids=sorted({i for spec in supervisor.full_specs for i in spec['rgb_frame_indices']})
    require(native_ids==contract['frame_regions']['nonblend'], 'Actual native decoder nonblend coverage differs')
    before=frozen_versions(vae,perceptual);precision_start=require_precision(device,contract)
    for ident in todo:
        directory=args.output/'decoded'/split/ident
        archive_incomplete(directory,args.output,'Decode/metric/media attempt incomplete; retained before retry');directory.mkdir(parents=True)
        pair=pairs[ident];prediction_receipts={};latent={}
        for label in LABELS[:-1]:
            folder=args.output/'predictions'/split/label/ident;receipt=valid_receipt(folder,digest)
            require(receipt is not None and receipt['checkpoint_sha256']==contract['selected_checkpoints'][label]['sha256'], 'Missing bound prediction')
            value=tensor_load(folder/'latent.pt')
            require(value['sample_id']==ident and value['split']==split and value['pair_sha256']==pair['pair_sha256'], 'Prediction metadata differs')
            latent[label]=value['latent'];prediction_receipts[label]=file_record(folder/'done.json')
        cache=args.output/'baseline_cache'/split/ident;cached=valid_receipt(cache,digest)
        if cached is None:
            archive_incomplete(cache,args.output,'Baseline RGB cache incomplete; retained before retry');cache.mkdir(parents=True)
            rgb_v5=decode_latent(vae,latent['v5'],device,keep_frames=120)
            rgb_ref=decode_latent(vae,pair['reference'],device,keep_frames=120)
            atomic_torch_save(cache/'rgb.pt',dict(v5=rgb_v5,reference=rgb_ref,sample_id=ident,split=split,pair_sha256=pair['pair_sha256']))
            immutable_json(cache/'done.json',dict(status='completed',contract_sha256=digest,sample_id=ident,split=split,
                pair_sha256=pair['pair_sha256'],v5_checkpoint_sha256=V5_SHA,prediction_receipt=prediction_receipts['v5'],
                decoder_fingerprint=decoder_identity,execution_precision=require_precision(device,contract),files={'rgb.pt':file_record(cache/'rgb.pt')}))
        else:
            require(cached['pair_sha256']==pair['pair_sha256'] and cached['v5_checkpoint_sha256']==V5_SHA and cached['decoder_fingerprint']==decoder_identity and cached['prediction_receipt']==prediction_receipts['v5'] and cached['execution_precision']==contract['execution_precision'], 'Baseline RGB cache binding differs')
            value=tensor_load(cache/'rgb.pt');rgb_v5,rgb_ref=value['v5'],value['reference']
            require(value['sample_id']==ident and value['split']==split and value['pair_sha256']==pair['pair_sha256'], 'Baseline cache identity differs')
        pixels={'v5':rgb_v5}
        for arm in ARMS:pixels[arm]=decode_latent(vae,latent[arm],device,keep_frames=120)
        pixels['reference']=rgb_ref
        require(all(tuple(p.shape)==(120,384,672,3) and p.dtype==torch.uint8 for p in pixels.values()), 'RGB coverage/dtype differs')
        masks={region:target_mask_metadata(rgb_ref,ids) for region,ids in contract['frame_regions'].items()}
        metrics={label:rgb_metrics(rgb,rgb_ref,perceptual,device,contract['frame_regions']) for label,rgb in pixels.items() if label!='reference'}
        for region,mask in masks.items():
            fractions=[metrics[label]['regions'][region]['motion_roi_fraction'] for label in LABELS[:-1]]
            require(len(set(fractions))==1 and all(metrics[label]['regions'][region]['adjacent_pair_count']==mask['adjacent_pairs'] for label in LABELS[:-1]), 'Model-dependent target mask or pairing')
        transition_metrics={label:rgb_transition_metrics(rgb,rgb_ref,contract['temporal_transition_pair_groups']) for label,rgb in pixels.items() if label!='reference'}
        transitions={}
        for group,pair_indices in contract['temporal_transition_pair_groups'].items():
            mask_hashes={value[group]['motion_mask_sha256'] for value in transition_metrics.values()}
            require(len(mask_hashes)==1, 'Model-dependent transition target mask')
            transitions[group]=dict(pair_indices=pair_indices,motion_mask_sha256=next(iter(mask_hashes)),
                variants={label:value[group]['metrics'] for label,value in transition_metrics.items()})
        entry=dict(sample_id=ident,split=split,frames=120,metric_domain=contract['metric_domain'],target_motion_masks=masks,
                   variants=metrics,temporal_transition_groups=transitions)
        codecs={label:write_mp4(rgb,directory/(label+'.mp4'),fps=24.) for label,rgb in pixels.items()}
        comparisons(pixels,directory,fps=24.,crf=16,comparison_video=True)
        media=audit_media(directory);immutable_json(directory/'metrics.json',entry)
        files={name:{k:v[k] for k in ('sha256','bytes')} for name,v in media.items()};files['metrics.json']=file_record(directory/'metrics.json')
        require(frozen_versions(vae,perceptual)==before and all(p.grad is None for m in (vae,perceptual) for p in m.parameters()), 'Frozen decoder/LPIPS changed')
        immutable_json(directory/'done.json',dict(status='completed',contract_sha256=digest,sample_id=ident,split=split,pair_sha256=pair['pair_sha256'],
            files=files,media_audit=media,video_codecs=codecs,prediction_receipts=prediction_receipts,baseline_cache_receipt=file_record(cache/'done.json'),
            decoder_fingerprint=decoder_identity,lpips_fingerprint=lpips_identity,execution_precision_start=precision_start,
            execution_precision_end=require_precision(device,contract),frozen_modules_unchanged=True))
        print(json.dumps(dict(stage='full_pixel_evaluated',split=split,sample_id=ident,frames=120)),flush=True)
        del pixels,latent,rgb_ref,rgb_v5
    del supervisor,vae,perceptual;empty_cache(device)


def aggregate(entries, regions):
    result={}
    for region in regions:
        means={label:{key:statistics.mean(entry['variants'][label]['regions'][region][key] for entry in entries.values())
            for key in next(iter(entries.values()))['variants'][label]['regions'][region]} for label in LABELS[:-1]}
        result[region]=dict(mean=means,minus_v5={arm:{k:means[arm][k]-means['v5'][k] for k in means[arm]} for arm in ARMS},
            motion_minus_control={k:means['motion'][k]-means['control'][k] for k in means['motion']})
    return dict(sample_ids=list(entries),count=len(entries),weighting='Equal clip means; pair-weighted ROI inside each clip; mean per-clip PSNR',regions=result)



def aggregate_transition_groups(entries, pair_groups):
    groups={}
    for group,pairs in pair_groups.items():
        for entry in entries.values():
            require(entry['temporal_transition_groups'][group]['pair_indices']==pairs, 'Transition pair definition changed')
        means={label:{key:statistics.mean(entry['temporal_transition_groups'][group]['variants'][label][key] for entry in entries.values())
            for key in next(iter(entries.values()))['temporal_transition_groups'][group]['variants'][label]} for label in LABELS[:-1]}
        groups[group]=dict(pair_indices=pairs,mean=means,
            minus_v5={arm:{k:means[arm][k]-means['v5'][k] for k in means[arm]} for arm in ARMS},
            motion_minus_control={k:means['motion'][k]-means['control'][k] for k in means['motion']})
    return dict(count=len(entries),weighting='Equal clip means; target-ROI endpoints are pair-weighted within a clip',groups=groups,
        scope='Diagnostic only; includes12 blend entry/exit transitions; no change to full120 dev gate')


def split_summary(args, contract, digest, split):
    entries={};receipts={};media_count=0;causal={arm:{} for arm in ARMS}
    latent_samples={label:{} for label in LABELS[:-1]}
    for ident in contract['split_ids'][split]:
        directory=args.output/'decoded'/split/ident;receipt=valid_receipt(directory,digest)
        require(receipt is not None, 'Incomplete decoded split: '+ident)
        require(receipt['frozen_modules_unchanged'] and receipt['execution_precision_start']==receipt['execution_precision_end']==contract['execution_precision'], 'Decode freeze/precision evidence differs')
        for label in LABELS[:-1]:
            folder=args.output/'predictions'/split/label/ident;pr=valid_receipt(folder,digest)
            require(pr is not None and file_record(folder/'done.json')==receipt['prediction_receipts'][label], 'Prediction receipt changed after decode')
            require(pr['variant']==label and pr['split']==split and pr['pair_sha256']==receipt['pair_sha256'] and pr['checkpoint_sha256']==contract['selected_checkpoints'][label]['sha256'], 'Prediction identity binding differs')
            require(pr['frozen_student_unchanged'] and pr['execution_precision_start']==pr['execution_precision_end']==contract['execution_precision'], 'Prediction freeze/precision evidence differs')
            require(all(math.isfinite(v) for v in pr['latent_metrics'].values()), 'Nonfinite latent evidence')
            latent_samples[label][ident]=pr['latent_metrics']
            if split=='dev' and label in ARMS and ident in contract['causal_dev_ids']:
                require(pr['causality']['passed'] and len(pr['causality']['checks'])==9 and pr['causality']['absolute_max_abs']<=1e-4, 'Selected-model causal check failed')
                causal[label][ident]=pr['causality']
        cache=args.output/'baseline_cache'/split/ident
        require(valid_receipt(cache,digest) is not None and file_record(cache/'done.json')==receipt['baseline_cache_receipt'], 'Baseline RGB cache receipt changed')
        media=receipt['media_audit'];require(set(media)=={name+'.mp4' for name in (*LABELS,'comparison')}|{'comparison.png'}, 'Wrong four-column media coverage')
        for name,audit in media.items():
            require(audit['passed'], 'Media audit failed')
            if name.endswith('.mp4'):
                media_count+=1;require(audit['decoded_frames']==120 and audit['fps']==24 and audit['duration_seconds']==5, 'Media timing differs')
        entry=read_json(directory/'metrics.json');require(entry['sample_id']==ident and entry['split']==split and entry['frames']==120, 'Pixel entry differs')
        entries[ident]=entry;receipts[ident]=file_record(directory/'done.json')
    if split=='dev':require(all(len(v)==4 for v in causal.values()), 'Expected36 default-shape causal checks per new arm')
    latent_mean={label:{key:statistics.mean(v[key] for v in values.values()) for key in next(iter(values.values()))} for label,values in latent_samples.items()}
    latent_report=dict(samples=latent_samples,mean=latent_mean,
        minus_v5={arm:{key:latent_mean[arm][key]-latent_mean['v5'][key] for key in latent_mean[arm]} for arm in ARMS},
        motion_minus_control={key:latent_mean['motion'][key]-latent_mean['control'][key] for key in latent_mean['motion']},
        note='Legacy raw ewarp_l1 remains reported even when target-residual or pixel delta metrics improve; these metrics are different. No latent/test metric selects checkpoints here.')
    return dict(split=split,pixel_metrics=entries,aggregate=aggregate(entries,contract['frame_regions']),latent_metrics=latent_report,
                temporal_transition_groups=aggregate_transition_groups(entries,contract['temporal_transition_pair_groups']),
                clip_receipt_fingerprints=receipts,full_decoded_mp4_files=media_count,causality=causal,
                causality_scope='4dev x9checks for each new selected arm in recorded default HF32; no new test causality checks')


def gate(group):
    means=group['aggregate']['regions']['full120']['mean'];base=means['v5'];arms={}
    for arm in ARMS:
        value=means[arm]
        checks={'global_score':value['global_score']<=base['global_score']*1.01,
                'pixel_l1':value['pixel_l1']<=base['pixel_l1']*1.02,'lpips':value['lpips']<=base['lpips']*1.02,
                'motion_l1':value['motion_l1']<base['motion_l1'],'delta_l1':value['delta_l1']<base['delta_l1']}
        arms[arm]=dict(passed=all(checks.values()),checks=checks,observed=value,baseline=base,
            ratios_to_v5={key:value[key]/base[key] if base[key]>0 else None for key in checks},
            scope='Complete16dev equal clip means, full120 only; no gate on test')
    return dict(arms=arms,motion_passed=arms['motion']['passed'],test_unlocked=arms['motion']['passed'],
                no_checkpoint_reselection=True,thresholds=dict(global_score_factor=1.01,pixel_l1_factor=1.02,lpips_factor=1.02,motion_l1='strict improvement',delta_l1='strict improvement'))


def run_dev(args):
    contract,rows,_,digest=verify_contract(args)
    existing=args.output/'dev_summary.json'
    if existing.exists():
        result=read_json(existing);require(result['contract_sha256']==digest, 'Dev summary contract changed')
        actual=split_summary(args,contract,digest,'dev');require(result['evaluation']==actual and result['gate']==gate(actual), 'Completed dev evidence changed')
        return result
    device=select_device(args.device);require_precision(device,contract)
    pairs=load_split(args,contract,rows,'dev')
    infer_split(args,contract,pairs,'dev',device,digest);decode_split(args,contract,pairs,'dev',device,digest)
    evaluation=split_summary(args,contract,digest,'dev');decision=gate(evaluation)
    verify_contract(args)
    result=dict(status='dev_passed' if decision['test_unlocked'] else 'dev_gate_failed',contract_sha256=digest,
        selected_checkpoints=contract['selected_checkpoints'],evaluation=evaluation,gate=decision,test_tensors_opened=False,
        source_contract_unchanged=True,checkpoint_selection='Frozen training-dev selections; full120 gate only accepts/rejects, never reselects',
        causal_scope=contract['causal_scope'])
    immutable_json(existing,result)
    print(json.dumps(dict(stage='dev_completed',status=result['status'],motion_passed=decision['motion_passed'],control_passed=decision['arms']['control']['passed'])),flush=True)
    return result


def run_test(args, dev):
    require(dev['gate']['test_unlocked'] and dev['status']=='dev_passed', 'Motion dev gate failed: test remains locked')
    contract,rows,_,digest=verify_contract(args)
    require(dev['contract_sha256']==digest, 'Dev gate contract changed')
    if (args.output/'test_summary.json').exists():
        saved=read_json(args.output/'test_summary.json');require(saved['contract_sha256']==digest and saved['dev_summary_sha256']==sha256(args.output/'dev_summary.json'), 'Test summary identity changed')
        require(saved['groups']=={split:split_summary(args,contract,digest,split) for split in ('test','legacy')}, 'Completed test evidence changed')
        return saved
    # This is the only path that calls load_pairs on either held-out manifest.
    immutable_json(args.output/'test_opened.json',dict(contract_sha256=digest,dev_summary_sha256=sha256(args.output/'dev_summary.json'),
        motion_dev_gate_passed=True,selected_checkpoints=contract['selected_checkpoints'],checkpoint_reselection=False))
    device=select_device(args.device);require_precision(device,contract)
    for split in ('test','legacy'):
        pairs=load_split(args,contract,rows,split)
        infer_split(args,contract,pairs,split,device,digest);decode_split(args,contract,pairs,split,device,digest);del pairs
    groups={split:split_summary(args,contract,digest,split) for split in ('test','legacy')}
    verify_contract(args)
    result=dict(status='completed',contract_sha256=digest,dev_summary_sha256=sha256(args.output/'dev_summary.json'),
        groups=groups,selected_checkpoints=contract['selected_checkpoints'],test_tensors_opened=True,
        no_checkpoint_reselection=True,source_contract_unchanged=True,
        test_reuse='Previously inspected16test+3legacy; notblind. Test and legacy aggregates are separate.',causal_scope=contract['causal_scope'])
    immutable_json(args.output/'test_summary.json',result)
    return result


def main():
    root=Path(os.environ.get('EXP_ROOT',Path(__file__).resolve().parents[1]));home=root/'upscaler_v5_motion_20261106'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('check','dev','test','all'),default='all')
    p.add_argument('--baseline-run',type=Path,default=root/'upscaler_conv3d_hybrid/kd_pixel_steps2000')
    p.add_argument('--control-run',type=Path,default=home/'control_steps2000')
    p.add_argument('--motion-run',type=Path,default=home/'motion_steps2000')
    p.add_argument('--output',type=Path,default=home/'evaluation')
    p.add_argument('--pairs',type=Path,default=root/'upscaler_ultraflash/pairs')
    p.add_argument('--upstream',type=Path,default=root/'upstream')
    p.add_argument('--weights',type=Path,default=root/'assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors')
    p.add_argument('--base-model',type=Path,default=os.environ['H3_BASE_MODEL'])
    p.add_argument('--device',default='npu:0');args=p.parse_args();torch.set_num_threads(4)
    contract,_,_=context(args)
    if args.stage=='check':
        print(json.dumps(dict(status='inputs_verified',v5_sha256=V5_SHA,selected_checkpoints=contract['selected_checkpoints'],test_tensors_opened=False)),flush=True);return
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        immutable_json(args.output/'contract.json',contract)
        dev=run_dev(args)
        if not dev['gate']['test_unlocked']:
            immutable_json(args.output/'summary.json',dict(status='dev_gate_failed',contract_sha256=sha256(args.output/'contract.json'),
                dev_summary_sha256=sha256(args.output/'dev_summary.json'),gate=dev['gate'],test_tensors_opened=False,
                selected_checkpoints=contract['selected_checkpoints'],no_checkpoint_reselection=True,
                note='Full dev results/videos retained; no test tensor was opened by this evaluator.'))
            return
        if args.stage=='dev':return
        result=run_test(args,dev)
        immutable_json(args.output/'summary.json',dict(status='completed',contract_sha256=sha256(args.output/'contract.json'),
            dev_summary_sha256=sha256(args.output/'dev_summary.json'),test_summary_sha256=sha256(args.output/'test_summary.json'),
            gate=dev['gate'],selected_checkpoints=contract['selected_checkpoints'],test_tensors_opened=True,
            no_checkpoint_reselection=True,groups=result['groups'],causal_scope=contract['causal_scope']))


if __name__=='__main__':main()
