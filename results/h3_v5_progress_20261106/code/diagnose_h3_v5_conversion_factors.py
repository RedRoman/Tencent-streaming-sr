"""Read-only 2x2 H3 conversion diagnostic; no optimizer or checkpoint writes.

Four deliberately inspected dev clips, not a representative or blind benchmark.
The Conv-only branch retains whole-clip GroupNorm and is NONCAUSAL. The GN-only
branch retains symmetric convolutions and is NONCAUSAL. Only BothInitial has
the complete causal latent adapter. All RGB is decoded by the original,
noncausal frozen H3 VAE. Existing v5/motion predictions are SHA-bound comparators.

Run CPU-only --stage self-test first, then --stage check and --stage latent.
Only --stage decode/all loads the decoder. --stage finalize verifies completion.
This script does not set HF32 flags or acquire a device lock: the caller must
hold the established physical-device lock for the entire invocation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

# This standalone diagnostic never copies or changes the frozen implementation.
_bootstrap = argparse.ArgumentParser(add_help=False)
_bootstrap.add_argument('--root', type=Path, default=Path(os.environ.get(
    'EXP_ROOT', '${EXPERIMENT_ROOT}')))
_early, _ = _bootstrap.parse_known_args()
sys.path.insert(0, str(_early.root.resolve() / 'experiment'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from causal_upscaler import CausalUpscalerAdapter, initialize_aligned_past_, temporal_structure
from data_utils import decode_latent, empty_cache, load_video_vae, select_device, synchronize, tensor_load, write_mp4
from decode_results import comparisons
from evaluate_h3_cmn import audit_video, file_record, module_source, read_json, require, vae_fingerprint
from evaluate_h3_conv3d_hybrid import load_lpips
from evaluate_h3_conv3d_test16 import archive_incomplete, immutable_json
from evaluate_h3_v5_motion import (SOURCE_NAMES, frame_regions, transition_pair_groups,
    rgb_metrics, rgb_transition_metrics, target_mask_metadata)
from h3_v5_motion import target_residual_ewarp
from prepare_data import load_author_upscaler
from train_h3_cmn import objective
from train_h3_v5_motion import execution_precision
from train_upscaler_trial import atomic_torch_save, sha256, wrapper_forward
from ultraflash_ewarp import FLOW_VERSION

IDS = ('VIDGEN5S_0421364', 'VIDGEN5S_0665341', 'VIDGEN5S_0589299', 'VIDGEN5S_0760711')
FACTORS = ('teacher', 'gn_only', 'convs_only', 'both_initial')
LABELS = (*FACTORS, 'v5', 'motion', 'reference')
SPECS = {
    'teacher': dict(aligned_causal_convs=False, frame_groupnorm=False, causal_latent=False),
    'gn_only': dict(aligned_causal_convs=False, frame_groupnorm=True, causal_latent=False),
    'convs_only': dict(aligned_causal_convs=True, frame_groupnorm=False, causal_latent=False),
    'both_initial': dict(aligned_causal_convs=True, frame_groupnorm=True, causal_latent=True),
}
DISPLAY = dict(teacher='teacher NONCAUSAL', gn_only='GNonly NONCAUSAL',
    convs_only='ConvsOnly NONCAUSAL', both_initial='BothInitial', v5='v5', motion='motion', reference='decoded HR')
PARAMETERS = 345280216


def named_record(path):
    path = Path(path).resolve()
    return dict(path=str(path), **file_record(path))


def freeze_versions(*models):
    return tuple((id(t), t._version, str(t.dtype), tuple(t.shape))
                 for m in models for t in (*m.parameters(), *m.buffers()))


def check_frozen(*models):
    require(all(not m.training for m in models), 'Frozen model entered training mode')
    require(all(not p.requires_grad and p.grad is None for m in models for p in m.parameters()), 'Frozen parameter/gradient changed')
    require(all(p.dtype == torch.float32 for m in models for p in m.parameters() if p.is_floating_point()), 'Non-FP32 model parameters')


def state_sha(model):
    """Hash actual in-memory parameters/buffers, one tensor at a time."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(json.dumps(list(value.shape)).encode()); h.update(value.tobytes())
    return h.hexdigest()


def apply_factor(node, model, label):
    """Reuse the exact frozen adapter, restoring only the excluded module type.

    These are instance-bound forward restorations, never changes to a class,
    imported module global, or any source file. No streaming is used here.
    """
    require(label in FACTORS, 'Unknown conversion factor')
    spec = SPECS[label]; initialization = None; restored = []
    if spec['aligned_causal_convs']:
        initialization = initialize_aligned_past_(model)
    adapter = None
    if label != 'teacher':
        adapter = CausalUpscalerAdapter(node, model)
        by_identity = {id(module): name for name, module in model.named_modules()}
        for module, original in adapter._original_forwards:
            if ((isinstance(module, nn.Conv3d) and not spec['aligned_causal_convs']) or
                    (isinstance(module, nn.GroupNorm) and not spec['frame_groupnorm'])):
                module.forward = original
                restored.append(by_identity[id(module)])
    return adapter, dict(**spec, initialization=initialization, restored_original_forwards=restored,
        scope='Whole-clip forward only; noncausal labels are diagnostic controls, never streaming candidates')


def json_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def context(args):
    """Hash only metadata, 4 selected data pairs/flows and required frozen assets."""
    history = read_json(args.history/'contract.json')
    historical_summary = read_json(args.history/'summary.json')
    history_sha = sha256(args.history/'contract.json')
    require(historical_summary['status'] == 'completed' and historical_summary['contract_sha256'] == history_sha,
            'Original completed evaluation contract differs')
    training_contract = read_json(args.root/'upscaler_v5_motion_20261106/motion_steps2000/contract.json')
    original = training_contract['original_upscaler_fingerprint']
    require(original['parameter_count'] == PARAMETERS, 'Original architecture differs')
    require(training_contract['execution_precision'] == history['execution_precision'], 'Historical precision differs')
    for key in ('decoder_fingerprint', 'lpips_fingerprint'):
        require(training_contract[key] == history[key], 'Historical '+key+' differs')
    files = {}
    def bind(name, path, expected_sha=None):
        rec = named_record(path)
        if expected_sha is not None: require(rec['sha256'] == expected_sha, 'Input SHA differs: '+name)
        files[name] = rec
        return rec
    bind('diagnostic_script', Path(__file__))
    bind('historical_contract', args.history/'contract.json', history_sha)
    bind('historical_summary', args.history/'summary.json')
    bind('motion_training_contract', args.root/'upscaler_v5_motion_20261106/motion_steps2000/contract.json')
    bind('upscaler_weights', args.weights, original['base_weights_sha256'])
    bind('upscaler_loader', args.upstream/'open_h3/upscaler.py', original['loader_sha256'])
    bind('upscaler_node', args.upstream/'open_h3/upscaler_node.py', original['loaded_node_sha256'])
    imported_sources = {Path(module.__file__).name for module in tuple(sys.modules.values())
        if getattr(module, '__file__', None) and Path(module.__file__).suffix=='.py' and
        Path(module.__file__).resolve().parent==(args.root/'experiment').resolve()}
    for name in sorted(set(SOURCE_NAMES) | imported_sources):
        rec = bind('source/'+name, module_source(name))
        if name in history['source_sha256']:
            require(rec['sha256'] == history['source_sha256'][name], 'Frozen historical source changed: '+name)
    ready = read_json(args.pairs/'READY.json')
    require(ready['status'] == 'ready' and ready['plan_sha256'] == history['plan_sha256'], 'Data plan identity differs')
    bind('data_ready', args.pairs/'READY.json', history['ready_sha256'])
    bind('data_plan', args.pairs/'plan.json', history['plan_sha256'])
    bind('dev_manifest', args.pairs/'dev.jsonl', history['manifests']['dev']['sha256'])
    rows = json_rows(args.pairs/'dev.jsonl')
    require(len(rows) == 16 and len({r['sample_id'] for r in rows}) == 16 and all(r['split']=='dev' for r in rows), 'Unexpected dev manifest')
    by_id = {r['sample_id']:r for r in rows}
    require(set(IDS) <= set(by_id), 'Requested dev diagnostic IDs absent')
    selected = {}
    for ident in IDS:
        row = by_id[ident]
        pair = Path(row['path']); pair = pair if pair.is_absolute() else args.pairs/pair
        pair_rec = bind('pair/'+ident, pair, row['sha256'])
        flow_rec = bind('flow/'+ident, args.pairs/'flow'/pair.name)
        selected[ident] = dict(row=row, pair=pair_rec, flow=flow_rec)
    decoder = vae_fingerprint(args.base_model)
    require(decoder == history['decoder_fingerprint']['weights'], 'VAE files differ')
    for name, rec in decoder.items(): bind('decoder/'+name, Path(args.base_model)/'vae'/name, rec['sha256'])
    from diffusers import AutoencoderKLMiniMaxH3
    bind('decoder/implementation', Path(inspect.getfile(AutoencoderKLMiniMaxH3)), history['decoder_fingerprint']['implementation_sha256'])
    import lpips
    lpips_files = {'backbone':Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth',
        'linear':Path(lpips.__file__).resolve().parent/'weights/v0.1/alex.pth',
        'implementation':Path(inspect.getfile(lpips.LPIPS))}
    for name, path in lpips_files.items():
        key = {'backbone':'backbone_sha256','linear':'linear_sha256','implementation':'implementation_sha256'}[name]
        bind('lpips/'+name, path, history['lpips_fingerprint'][key])
    caches = {}
    for label in ('v5','motion'):
        selected_checkpoint = history['selected_checkpoints'][label]
        bind('checkpoint/'+label, selected_checkpoint['path'], selected_checkpoint['sha256'])
        require(selected_checkpoint == historical_summary['selected_checkpoints'][label], 'Historical selected checkpoint differs')
        caches[label] = {}
        for ident in IDS:
            directory = args.history/'predictions/dev'/label/ident
            receipt = read_json(directory/'done.json')
            require(receipt['status']=='completed' and receipt['contract_sha256']==history_sha and
                    receipt['sample_id']==ident and receipt['split']=='dev' and receipt['variant']==label and
                    receipt['pair_sha256']==by_id[ident]['sha256'] and
                    receipt['checkpoint_sha256']==selected_checkpoint['sha256'], 'Historical prediction identity differs')
            require(receipt['frozen_student_unchanged'] is True and
                    receipt['execution_precision_start']==receipt['execution_precision_end']==history['execution_precision'],
                    'Historical prediction freeze/precision differs')
            require(set(receipt['files']) in ({'latent.pt'}, {'latent.pt','causality.json'}), 'Unexpected historical prediction files')
            for name, rec in receipt['files'].items():
                require(file_record(directory/name)==rec, 'Historical prediction receipt file differs')
                bind('cache/'+label+'/'+ident+'/'+name, directory/name, rec['sha256'])
            bind('cache/'+label+'/'+ident+'/done.json', directory/'done.json')
            caches[label][ident] = dict(path=str(directory/'latent.pt'), receipt_path=str(directory/'done.json'),
                checkpoint=selected_checkpoint, files=receipt['files'], prior_latent_metrics=receipt['latent_metrics'])
    teacher_ready = read_json(args.teacher_cache/'READY.json')
    require(teacher_ready['status']=='ready' and teacher_ready['contract']['teacher_sha256']==original['base_weights_sha256'] and
            teacher_ready['contract']['plan_sha256']==history['plan_sha256'], 'Teacher cache provenance differs')
    bind('teacher_ready', args.teacher_cache/'READY.json')
    teacher_caches = {}
    for ident in IDS:
        rec = teacher_ready['records'][ident]
        require(rec['split']=='dev' and rec['pair_sha256']==by_id[ident]['sha256'], 'Historical teacher pair differs')
        teacher_caches[ident] = bind('teacher_cache/'+ident, args.teacher_cache/rec['path'], rec['sha256'])
    return dict(schema_version=1, diagnostic='H3 conversion factorial 2x2, frozen original weights; no training',
        sample_ids=list(IDS), sample_scope='Four preselected inspected development clips; no train/test tensors loaded; no representative or blind claim',
        factors=SPECS, labels=list(LABELS), display_labels=DISPLAY, selected_data=selected,
        historical_prediction_caches=caches, historical_teacher_caches=teacher_caches,
        original_upscaler_fingerprint=original, decoder_fingerprint=history['decoder_fingerprint'],
        lpips_fingerprint=history['lpips_fingerprint'], execution_precision=history['execution_precision'],
        files=files, frame_regions=frame_regions(), temporal_transition_pair_groups=transition_pair_groups(),
        metric_domain=history['metric_domain'], frames=120, fps=24, pixels=[672,384],
        comparison_pixels=[672*len(LABELS),412], expected_mp4_files=len(IDS)*(len(LABELS)+1),
        upscaler_input_shape=[1,24,37,12,21], upscaler_output_shape=[1,24,37,24,42],
        resolution_scope='Existing experiment pairs: RGB LR336x192 to HR672x384; this diagnostic does not establish validity at the upstream deployment resolutions or isolate resolution-domain mismatch.',
        latent_metric_scope='MSE, delta MSE and raw eWarp valid first36 latent frames; residual eWarp uses supplied full37 pair mask, as frozen helper',
        causality_scope='Only BothInitial has complete causal latent conversion. teacher/GNonly/ConvsOnly are NONCAUSAL diagnostic controls. Decoder is noncausal for every column.',
        teacher_cache_comparison_scope='Fresh teacher is primary. Historical teacher comparison is reported, not assumed identical internal precision because its old READY did not record all math flags.',
        no_checkpoint_writes=True, no_optimizer=True, output=str(args.output.resolve()))


def verify_context(args):
    current = context(args)
    path = args.output/'contract.json'
    if path.exists(): require(read_json(path)==current, 'Diagnostic input/source contract changed')
    else:
        require(not args.output.exists(), 'Fresh output directory required')
        args.output.mkdir(parents=True)
        immutable_json(path,current)
    return current, sha256(path)


def require_precision(device, contract):
    value = execution_precision(device)
    require(value==contract['execution_precision'], 'Math flags differ; this diagnostic never changes them')
    return value


def receipt(directory, contract_sha, kind):
    path = directory/'done.json'
    if not path.exists(): return None
    value = read_json(path)
    require(value['status']=='completed' and value['kind']==kind and value['contract_sha256']==contract_sha,
            'Diagnostic receipt identity differs')
    if kind == 'prediction': expected={'latent.pt'}
    elif kind == 'decode': expected={label+'.mp4' for label in (*LABELS,'comparison')}|{'comparison.png','metrics.json'}
    else: raise ValueError('Unknown receipt kind')
    require(set(value['files'])==expected, 'Receipt file coverage differs')
    for name, rec in value['files'].items(): require(file_record(directory/name)==rec, 'Diagnostic receipt changed')
    require(value['sample_id']==directory.name, 'Receipt sample/parent mismatch')
    return value


def load_pairs_selected(contract):
    pairs = {}
    for ident in IDS:
        rec = contract['selected_data'][ident]; row=rec['row']
        value=tensor_load(rec['pair']['path']); flow=tensor_load(rec['flow']['path']);meta=value['metadata']
        require(meta['split']=='dev' and meta.get('sample_id',meta.get('id'))==ident and
                meta['source_video_sha256']==row['source_video_sha256'], 'Pair metadata differs')
        require(flow['metadata']['version']==FLOW_VERSION and flow['pair_sha256']==rec['pair']['sha256'], 'Flow provenance differs')
        require(tuple(value['draft'].shape)==(24,37,12,21) and tuple(value['reference'].shape)==(24,37,24,42), 'Pair geometry differs')
        pairs[ident]=dict(sample_id=ident, pair_sha256=rec['pair']['sha256'], draft=value['draft'].float(),
            reference=value['reference'].float(), flow=flow['flow'], valid_pairs=flow['valid_pairs'])
        require(all(bool(torch.isfinite(pairs[ident][key]).all()) for key in ('draft','reference','flow')), 'Nonfinite source tensors')
    return pairs


def latent_metrics(pred, pair, device):
    reference=pair['reference'][None].to(device);flow=pair['flow'][None].to(device);valid=pair['valid_pairs'][None].to(device)
    values={k:float(v) for k,v in objective(pred,reference,flow,valid,.1).items()}
    values.update({k:float(v) for k,v in target_residual_ewarp(pred,reference,flow,valid).items()})
    values['temporal_difference_mse']=float(((pred[:,:,1:36]-pred[:,:,:35])-(reference[:,:,1:36]-reference[:,:,:35])).square().mean())
    require(all(math.isfinite(v) for v in values.values()), 'Nonfinite latent metrics')
    return values


def cached_latent(contract, label, ident):
    rec=contract['historical_prediction_caches'][label][ident]
    require(file_record(rec['path'])==rec['files']['latent.pt'], 'Cached prediction changed')
    value=tensor_load(rec['path'])
    require(value['sample_id']==ident and value['split']=='dev' and
            value['pair_sha256']==contract['selected_data'][ident]['pair']['sha256'], 'Cached latent identity differs')
    require(tuple(value['latent'].shape)==(24,37,24,42) and bool(torch.isfinite(value['latent']).all()), 'Cached latent shape/values differ')
    return value['latent'].float()


@torch.no_grad()
def infer(args, contract, digest, device, pairs):
    for label in FACTORS:
        todo=[ident for ident in IDS if receipt(args.output/'predictions'/label/ident,digest,'prediction') is None]
        if not todo: continue
        # Fresh imported node/model per factor: never restore from another ablation.
        node=load_author_upscaler(args.upstream,args.weights,device)
        require(not node.MODEL_CACHE, 'Fresh original-node model cache expected')
        model=node.load_model('upscaler',device,'fp32').eval().requires_grad_(False)
        require(sum(p.numel() for p in model.parameters())==PARAMETERS, 'Original model size differs')
        structure=temporal_structure(model)
        require(structure==contract['original_upscaler_fingerprint']['adapter_structure'], 'Temporal structure differs')
        adapter, change=apply_factor(node,model,label)
        model.eval().requires_grad_(False); check_frozen(model)
        versions=freeze_versions(model); before_sha=state_sha(model); precision=require_precision(device,contract)
        pending=[]
        for ident in todo:
            directory=args.output/'predictions'/label/ident
            archive_incomplete(directory,args.output,'Incomplete conversion-factor prediction retained before retry')
            directory.mkdir(parents=True)
            pair=pairs[ident];draft=pair['draft'][None].to(device)
            started=time.perf_counter()
            pred=wrapper_forward(node,model,draft,(37,24,42))
            synchronize(device); seconds=time.perf_counter()-started
            require(tuple(pred.shape)==(1,24,37,24,42) and bool(torch.isfinite(pred).all()), 'Invalid diagnostic prediction')
            metrics=latent_metrics(pred,pair,device); teacher_parity=None
            if label=='teacher':
                old=tensor_load(contract['historical_teacher_caches'][ident]['path'])
                require(old['metadata']['split']=='dev' and old['metadata']['pair_sha256']==pair['pair_sha256'], 'Teacher cache identity differs')
                require(tuple(old['latent'].shape)==tuple(pred[0].shape), 'Teacher cache shape differs')
                delta=(pred[0].cpu()-old['latent'].float()).abs()
                teacher_parity=dict(max_abs=float(delta.max()),mean_abs=float(delta.mean()),bit_exact=bool((delta==0).all()),
                    note=contract['teacher_cache_comparison_scope'])
            atomic_torch_save(directory/'latent.pt',dict(latent=pred[0].cpu(),sample_id=ident,split='dev',
                pair_sha256=pair['pair_sha256'],variant=label))
            check_frozen(model);require(freeze_versions(model)==versions,'Upscaler tensors mutated')
            pending.append((directory,dict(status='completed',kind='prediction',contract_sha256=digest,
                sample_id=ident,split='dev',variant=label,pair_sha256=pair['pair_sha256'],factor=change,
                original_weights_sha256=contract['original_upscaler_fingerprint']['base_weights_sha256'],
                latent_metrics=metrics,historical_teacher_comparison=teacher_parity,forward_seconds=seconds,
                execution_precision_start=precision,execution_precision_end=require_precision(device,contract),
                files={'latent.pt':file_record(directory/'latent.pt')})))
            print(json.dumps(dict(stage='factor_inference',variant=label,sample_id=ident,latent_metrics=metrics)),flush=True)
            del draft,pred
        after_sha=state_sha(model);require(before_sha==after_sha,'Upscaler actual state changed during inference')
        for directory,value in pending:
            value.update(state_sha256_before=before_sha,state_sha256_after=after_sha,frozen_parameters_unchanged=True)
            immutable_json(directory/'done.json',value)
        if adapter is not None: adapter.remove()
        node.MODEL_CACHE.clear();del model,node,adapter;empty_cache(device)
    # Their two corresponding parameter sets must be identical; only GN forward changes.
    states={label:receipt(args.output/'predictions'/label/IDS[0],digest,'prediction')['state_sha256_before'] for label in FACTORS}
    require(states['teacher']==states['gn_only'] and states['convs_only']==states['both_initial'], 'Factorial weight identity differs')


def audit_media(directory):
    media={}
    for label in (*LABELS,'comparison'):
        size=(672*len(LABELS),412) if label=='comparison' else (672,384)
        media[label+'.mp4']=audit_video(directory/(label+'.mp4'),*size)
    path=directory/'comparison.png'
    with Image.open(path) as img:
        require(img.format=='PNG' and img.size==(672*len(LABELS),1236),'Comparison PNG shape differs');img.verify()
    with Image.open(path) as img: img.load()
    media['comparison.png']=dict(passed=True,width=672*len(LABELS),height=1236,**file_record(path))
    return media


@torch.no_grad()
def decode(args, contract, digest, device, pairs):
    todo=[ident for ident in IDS if receipt(args.output/'decoded'/ident,digest,'decode') is None]
    if not todo: return
    vae=load_video_vae(args.base_model,device).eval().requires_grad_(False)
    perceptual,lpips_identity=load_lpips(device)
    implementation=Path(inspect.getfile(type(vae)))
    require(sha256(implementation)==contract['decoder_fingerprint']['implementation_sha256'],'Loaded decoder source differs')
    decoder_source_before=named_record(implementation)
    for actual,key in ((lpips_identity['backbone']['sha256'],'backbone_sha256'),
        (lpips_identity['linear_weights']['sha256'],'linear_sha256'),(lpips_identity['implementation_sha256'],'implementation_sha256')):
        require(actual==contract['lpips_fingerprint'][key],'Loaded LPIPS identity differs')
    check_frozen(vae,perceptual);before=freeze_versions(vae,perceptual);precision=require_precision(device,contract)
    for ident in todo:
        directory=args.output/'decoded'/ident
        archive_incomplete(directory,args.output,'Incomplete conversion-factor decode retained before retry');directory.mkdir(parents=True)
        pair=pairs[ident];pixels={};provenance={};latent_values={};decode_seconds={}
        for label in LABELS:
            if label in FACTORS:
                folder=args.output/'predictions'/label/ident;pr=receipt(folder,digest,'prediction')
                require(pr is not None and pr['variant']==label and pr['pair_sha256']==pair['pair_sha256'],'Missing/mismatched factor prediction')
                value=tensor_load(folder/'latent.pt');z=value['latent']
                require(value['sample_id']==ident and value['variant']==label and value['split']=='dev' and
                    value['pair_sha256']==pair['pair_sha256'],'Prediction payload identity differs')
                provenance[label]=dict(kind='fresh_original_factor',receipt=named_record(folder/'done.json'))
                latent_values[label]=pr['latent_metrics']
            elif label=='reference':
                z=pair['reference'];provenance[label]=dict(kind='encoded_hr_reference',pair_sha256=pair['pair_sha256'])
            else:
                z=cached_latent(contract,label,ident)
                provenance[label]=dict(kind='historical_frozen_prediction',**contract['historical_prediction_caches'][label][ident])
                latent_values[label]=latent_metrics(z[None].to(device),pair,device)
                for key,prior in contract['historical_prediction_caches'][label][ident]['prior_latent_metrics'].items():
                    require(abs(latent_values[label][key]-prior)<=1e-6,'Cached baseline latent metric parity failed')
            require(tuple(z.shape)==(24,37,24,42) and bool(torch.isfinite(z).all()),'Decode latent shape/values differ')
            started=time.perf_counter();pixels[label]=decode_latent(vae,z,device,keep_frames=120)
            synchronize(device);decode_seconds[label]=time.perf_counter()-started
        require(list(pixels)==list(LABELS) and all(tuple(p.shape)==(120,384,672,3) and p.dtype==torch.uint8 for p in pixels.values()),'RGB shape/order differs')
        target=pixels['reference'];regions=contract['frame_regions'];groups=contract['temporal_transition_pair_groups']
        masks={region:target_mask_metadata(target,ids) for region,ids in regions.items()}
        metrics={label:rgb_metrics(pixels[label],target,perceptual,device,regions) for label in LABELS[:-1]}
        transitions={label:rgb_transition_metrics(pixels[label],target,groups) for label in LABELS[:-1]}
        for region in regions:
            require(len({metrics[label]['regions'][region]['motion_roi_fraction'] for label in LABELS[:-1]})==1,'Variant-dependent GT motion mask')
        for group in groups:
            require(len({transitions[label][group]['motion_mask_sha256'] for label in LABELS[:-1]})==1,'Variant-dependent transition mask')
        entry=dict(sample_id=ident,split='dev',frames=120,variants=metrics,latent_metrics=latent_values,
            target_motion_masks=masks,temporal_transition_groups=transitions,provenance=provenance,
            decode_seconds=decode_seconds,metric_domain=contract['metric_domain'])
        codecs={label:write_mp4(rgb,directory/(label+'.mp4'),fps=24.,crf=16) for label,rgb in pixels.items()}
        comparisons({DISPLAY[label]:rgb for label,rgb in pixels.items()},directory,fps=24.,crf=16,comparison_video=True)
        media=audit_media(directory);immutable_json(directory/'metrics.json',entry)
        check_frozen(vae,perceptual);require(freeze_versions(vae,perceptual)==before,'Frozen decoder/LPIPS mutated')
        require(named_record(implementation)==decoder_source_before,'Decoder implementation changed during evaluation')
        files={name:{k:v[k] for k in ('sha256','bytes')} for name,v in media.items()};files['metrics.json']=file_record(directory/'metrics.json')
        immutable_json(directory/'done.json',dict(status='completed',kind='decode',contract_sha256=digest,
            sample_id=ident,split='dev',pair_sha256=pair['pair_sha256'],files=files,media_audit=media,
            provenance=provenance,video_codecs=codecs,decoder_source=decoder_source_before,lpips_identity=lpips_identity,
            frozen_modules_unchanged=True,execution_precision_start=precision,execution_precision_end=require_precision(device,contract)))
        print(json.dumps(dict(stage='decoded',sample_id=ident,frames=120,mp4_files=len(LABELS)+1)),flush=True)
        del pixels,target
    del vae,perceptual;empty_cache(device)


def effects(means):
    keys=next(iter(means.values())).keys()
    differences={name:{k:means[a][k]-means[b][k] for k in keys} for name,a,b in (
        ('GN_change_with_symmetric_convs','gn_only','teacher'),
        ('GN_change_with_aligned_convs','both_initial','convs_only'),
        ('conv_change_with_full_GN','convs_only','teacher'),
        ('conv_change_with_frame_GN','both_initial','gn_only'),
        ('full_conversion','both_initial','teacher'),('v5_minus_initial','v5','both_initial'),
        ('motion_minus_v5','motion','v5'))}
    differences['factor_interaction']={k:means['both_initial'][k]-means['gn_only'][k]-means['convs_only'][k]+means['teacher'][k] for k in keys}
    return differences


@torch.no_grad()
def latent_summary(args, contract, digest, device, pairs):
    """Complete the isolated latent stage without loading or invoking the VAE."""
    samples={label:{} for label in LABELS[:-1]};records={};states={};teacher_comparison={}
    for label in FACTORS:
        values=set()
        for ident in IDS:
            directory=args.output/'predictions'/label/ident;pr=receipt(directory,digest,'prediction')
            require(pr is not None and pr['variant']==label and pr['pair_sha256']==pairs[ident]['pair_sha256'], 'Factor receipt mismatch')
            require(pr['frozen_parameters_unchanged'] and pr['state_sha256_before']==pr['state_sha256_after'] and
                pr['execution_precision_start']==pr['execution_precision_end']==contract['execution_precision'], 'Factor state/precision failed')
            values.add(pr['state_sha256_before']);samples[label][ident]=pr['latent_metrics'];records[label+'/'+ident]=named_record(directory/'done.json')
            if label=='teacher':teacher_comparison[ident]=pr['historical_teacher_comparison']
        require(len(values)==1, 'One factor has different states across clips/invocations')
        states[label]=next(iter(values))
    require(states['teacher']==states['gn_only'] and states['convs_only']==states['both_initial'], 'Factorial state equality failed')
    precision=require_precision(device,contract)
    for label in ('v5','motion'):
        for ident in IDS:
            z=cached_latent(contract,label,ident)
            samples[label][ident]=latent_metrics(z[None].to(device),pairs[ident],device)
            for key,prior in contract['historical_prediction_caches'][label][ident]['prior_latent_metrics'].items():
                require(abs(samples[label][ident][key]-prior)<=1e-6, 'Cached baseline latent metric parity failed')
    keys=samples['teacher'][IDS[0]]
    means={label:{key:statistics.mean(v[key] for v in values.values()) for key in keys} for label,values in samples.items()}
    require(context(args)==contract, 'Post-latent frozen source/weights/data changed')
    value=dict(status='latent_completed',contract_sha256=digest,sample_ids=list(IDS),samples=samples,mean=means,
        differences=effects(means),per_clip_differences={ident:effects({label:samples[label][ident] for label in LABELS[:-1]}) for ident in IDS},
        factor_state_sha256=states,prediction_receipts=records,historical_teacher_comparisons=teacher_comparison,
        forward_count=16,cached_trained_predictions=8,no_decoder_loaded=True,no_training=True,
        frozen_source_and_weights_unchanged=True,execution_precision_start=precision,execution_precision_end=require_precision(device,contract),
        scope=contract['sample_scope'],resolution_scope=contract['resolution_scope'],causality_scope=contract['causality_scope'])
    immutable_json(args.output/'latent_summary.json',value)
    print(json.dumps(dict(stage='latent_completed',mean=means,differences=effects(means),no_decoder_loaded=True)),flush=True)
    return value


def finalize(args, contract, digest):
    entries={};decoded_records={};factor_records={};media_count=0
    for ident in IDS:
        for label in FACTORS:
            folder=args.output/'predictions'/label/ident;pr=receipt(folder,digest,'prediction')
            require(pr is not None and pr['variant']==label and pr['pair_sha256']==contract['selected_data'][ident]['pair']['sha256'],'Missing/wrong prediction receipt')
            require(pr['frozen_parameters_unchanged'] and pr['state_sha256_before']==pr['state_sha256_after'] and
                pr['execution_precision_start']==pr['execution_precision_end']==contract['execution_precision'],'Prediction state/precision differs')
            factor_records[label+'/'+ident]=named_record(folder/'done.json')
        folder=args.output/'decoded'/ident;dr=receipt(folder,digest,'decode')
        require(dr is not None and dr['pair_sha256']==contract['selected_data'][ident]['pair']['sha256'],'Missing/wrong decode receipt')
        require(dr['frozen_modules_unchanged'] and dr['execution_precision_start']==dr['execution_precision_end']==contract['execution_precision'],'Decoder freeze/precision differs')
        for label in FACTORS:
            require(dr['provenance'][label]['receipt']==factor_records[label+'/'+ident],'Decoded prediction provenance changed')
        for name,audit in dr['media_audit'].items():
            require(audit['passed'],'Media failed')
            if name.endswith('.mp4'):
                require(audit['decoded_frames']==120 and audit['fps']==24 and audit['duration_seconds']==5,'Media time coverage differs')
                media_count+=1
        entry=read_json(folder/'metrics.json')
        require(entry['sample_id']==ident and entry['split']=='dev' and entry['frames']==120 and set(entry['variants'])==set(LABELS[:-1]),'Metric sample/variant coverage differs')
        entries[ident]=entry;decoded_records[ident]=named_record(folder/'done.json')
    require(media_count==contract['expected_mp4_files']==32,'Incomplete32 MP4 evidence')
    aggregates={}
    for region in contract['frame_regions']:
        keys=entries[IDS[0]]['variants']['teacher']['regions'][region]
        means={label:{k:statistics.mean(entry['variants'][label]['regions'][region][k] for entry in entries.values()) for k in keys} for label in LABELS[:-1]}
        aggregates[region]=dict(mean=means,differences=effects(means))
    latent_keys=entries[IDS[0]]['latent_metrics']['teacher']
    latent_mean={label:{k:statistics.mean(entry['latent_metrics'][label][k] for entry in entries.values()) for k in latent_keys} for label in LABELS[:-1]}
    # Rehash every frozen input/source/asset after all stages. No SHA rewriting.
    require(context(args)==contract,'Final source/weights/data fingerprint changed')
    result=dict(status='completed',contract_sha256=digest,sample_ids=list(IDS),pixel_metrics=entries,
        aggregate=aggregates,latent_mean=latent_mean,latent_differences=effects(latent_mean),
        aggregation='Equal clip means, mean per-clip PSNR; within-clip fixed GT ROI endpoint weighting',
        full_decoded_mp4_files=media_count,prediction_receipts=factor_records,decode_receipts=decoded_records,
        frozen_source_and_weights_unchanged=True,no_training=True,no_saved_weights=True,
        limitations=[contract['sample_scope'],contract['causality_scope'],
            'Factor differences include nonlinear interactions; they do not prove why trained models ghost.',
            'v5/motion have different training histories from zero-update factors; they are descriptive comparators, not equal-compute factor arms.',
            'Scores are against decoded HR, not independent raw RGB; visual review is still required.',
            'Full-sequence latent inference, no stream cache; structural conversion diagnosis, not an end-to-end streaming claim.'])
    immutable_json(args.output/'summary.json',result)
    print(json.dumps(dict(stage='completed',sample_ids=list(IDS),mp4_files=media_count,
        full120_means=aggregates['full120']['mean'])),flush=True)
    return result


def self_test():
    """Small CPU-only semantic test, no real weights/data/decoder/NPU access."""
    torch.manual_seed(706);torch.set_num_threads(2)
    class Node:
        @staticmethod
        def _make_norm_tensors(device,dtype):
            return torch.zeros(1,4,1,1,1,device=device,dtype=dtype),torch.ones(1,4,1,1,1,device=device,dtype=dtype)
    class Toy(nn.Module):
        def __init__(self):
            super().__init__();self.conv=nn.Conv3d(4,4,3,padding=1);self.norm=nn.GroupNorm(2,4)
        def forward(self,x,scale,target_size,enable_chunking):
            return F.interpolate(self.norm(self.conv(x))+x,size=target_size,mode='trilinear',align_corners=False)
    original=Toy().eval().requires_grad_(False);x=torch.randn(1,4,9,3,4);target=(9,6,8)
    values={};checks={}
    with torch.no_grad():
        for label in FACTORS:
            model=copy.deepcopy(original);adapter,report=apply_factor(Node,model,label)
            actual=wrapper_forward(Node,model,x,target);spec=SPECS[label]
            w=original.conv.weight.clone()
            if spec['aligned_causal_convs']:
                manual=torch.zeros_like(w);manual[:,:,1:]=w[:,:,:2];w=manual
                z=F.conv3d(F.pad(x,(0,0,0,0,2,0)),w,original.conv.bias,padding=(0,1,1))
            else:z=F.conv3d(x,w,original.conv.bias,padding=1)
            if spec['frame_groupnorm']:
                b,c,t,h,wd=z.shape
                z=F.group_norm(z.permute(0,2,1,3,4).reshape(b*t,c,h,wd),2,original.norm.weight,original.norm.bias,original.norm.eps)
                z=z.reshape(b,t,c,h,wd).permute(0,2,1,3,4)
            else:z=F.group_norm(z,2,original.norm.weight,original.norm.bias,original.norm.eps)
            expected=F.interpolate(z+x,size=target,mode='trilinear',align_corners=False)
            error=float((actual-expected).abs().max());require(error<2e-6,'Independent factor formula mismatch')
            changed=x.clone();changed[:,:,5:]+=torch.randn_like(changed[:,:,5:])*3
            future=float((actual[:,:,:5]-wrapper_forward(Node,model,changed,target)[:,:,:5]).abs().max())
            require((future==0) if label=='both_initial' else (future>1e-5),'Toy causal/noncausal distinction differs')
            checks[label]=dict(manual_formula_max_abs=error,future_response_max_abs=future)
            values[label]=state_sha(model)
            if adapter is not None:adapter.remove()
        require(values['teacher']==values['gn_only'] and values['convs_only']==values['both_initial'],'Factor parameter identity test failed')
        direct=copy.deepcopy(original);initialize_aligned_past_(direct);a=CausalUpscalerAdapter(Node,direct)
        ours=copy.deepcopy(original);_,_=apply_factor(Node,ours,'both_initial')
        error=float((a.native_forward(x,target)-wrapper_forward(Node,ours,x,target)).abs().max())
        require(error==0,'BothInitial differs from exact frozen adapter')
    print(json.dumps(dict(status='passed',scope='CPU toy only; no real-model/decoder quality inference',checks=checks,
        exact_adapter_max_abs=error,factor_weight_identity=True,expected_clips=4,expected_mp4_files=32)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,parents=[_bootstrap])
    p.add_argument('--stage',choices=('check','latent','infer','decode','finalize','all','self-test'),default='check')
    p.add_argument('--upstream',type=Path);p.add_argument('--weights',type=Path);p.add_argument('--pairs',type=Path)
    p.add_argument('--base-model',type=Path);p.add_argument('--history',type=Path);p.add_argument('--teacher-cache',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--device',default='npu:0')
    args=p.parse_args();args.root=args.root.resolve()
    require(args.root==_early.root.resolve(),'Import root differs from runtime root')
    if args.stage=='self-test':self_test();return
    home=args.root/'upscaler_v5_motion_20261106'
    args.upstream=args.upstream or args.root/'upstream'
    args.weights=args.weights or args.root/'assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors'
    args.pairs=args.pairs or args.root/'upscaler_ultraflash/pairs'
    args.base_model=args.base_model or Path('${H3_BASE_MODEL}')
    args.history=args.history or home/'evaluation'
    args.teacher_cache=args.teacher_cache or args.root/'upscaler_hybrid/teacher_cache'
    args.output=args.output or home/'diagnostics/conversion_factors_v1'
    torch.set_num_threads(4)
    contract,digest=verify_context(args)
    if args.stage=='check':
        print(json.dumps(dict(status='contract_checked',contract_sha256=digest,files=len(contract['files']),sample_ids=list(IDS),
            no_select_device_or_model_load=True,expected_mp4_files=32)),flush=True);return
    if args.stage=='finalize':finalize(args,contract,digest);return
    device=select_device(args.device);require_precision(device,contract)
    pairs=load_pairs_selected(contract)
    if args.stage in ('latent','infer','all'):
        infer(args,contract,digest,device,pairs)
        latent_summary(args,contract,digest,device,pairs)
    if args.stage in ('decode','all'):decode(args,contract,digest,device,pairs)
    require(context(args)==contract,'Post-stage frozen source/weights/data changed')
    if args.stage=='all':finalize(args,contract,digest)


if __name__=='__main__':
    main()
