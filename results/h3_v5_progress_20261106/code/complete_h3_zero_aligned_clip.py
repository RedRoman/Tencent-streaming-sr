"""Add one fixed zero-update aligned-past video, without changing any old result.

The historical initial_model.pt is already shifted. Reload it verbatim with its
causal operators; never initialize/shift again. This is inference, not training.
The caller must acquire the shared accelerator lock before infer/decode/all.
"""
from __future__ import annotations
import argparse
import importlib
import inspect
import json
from pathlib import Path
import sys
import time
import traceback

import torch
from data_utils import decode_latent, empty_cache, load_video_vae, select_device, synchronize, tensor_load, write_mp4
from evaluate_h3_cmn import audit_video, file_record, read_json, require, vae_fingerprint
from evaluate_upscaler_5s import causal_checks
from train_causal_upscaler import reload_causal_model
from train_h3_v5_motion import execution_precision, frozen_versions
from train_upscaler_trial import atomic_torch_save, sha256

SAMPLE_ID = 'VIDGEN5S_0086441'
INITIAL_SHA = 'e989d93ec87e751d295b9ec13a0af1c41401f1d75b69c4e25b2a3596c9dbc31e'


def exclusive_json(path, value):
    path = Path(path)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def imported_sources():
    directory = Path(__file__).resolve().parent
    paths = {Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.parent == directory and path.suffix == '.py':
                paths.add(path)
    return {p.name: file_record(p) for p in sorted(paths)}


def context(args):
    root = args.root
    pairs = root/'upscaler_ultraflash/pairs'
    current = root/'upscaler_v5_motion_20261106/evaluation'
    initial = root/'upscaler_causal_finetune/aligned_past_steps50/initial_model.pt'
    prior_path = root/'upscaler_causal_finetune/aligned_past_steps50/summary.json'
    old_path = root/'upscaler_5s/evaluation/summary.json'
    old = read_json(old_path); prior = read_json(prior_path)
    evaluation_path = current/'contract.json'; evaluation = read_json(evaluation_path)
    require(old['checkpoint_sha256']['causal_before'] == INITIAL_SHA, 'Historical zero-step checkpoint changed')
    require(prior['completed_updates'] == 50 and prior['passed'], 'Unexpected historical training provenance')
    require(sha256(initial) == INITIAL_SHA, 'Initial checkpoint SHA changed')
    plan_path = pairs/'plan.json'; ready_path = pairs/'READY.json'
    plan = read_json(plan_path); ready = read_json(ready_path)
    require(sha256(plan_path) == evaluation['plan_sha256'] == ready['plan_sha256'], 'Pair plan identity differs')
    manifest_path = pairs/'test.jsonl'
    require(sha256(manifest_path) == ready['manifests']['test']['sha256'] == evaluation['manifests']['test']['sha256'], 'Test manifest differs')
    rows = [json.loads(s) for s in manifest_path.read_text().splitlines() if s.strip()]
    rows = [r for r in rows if r['sample_id'] == SAMPLE_ID]
    require(len(rows) == 1 and rows[0]['split'] == 'test', 'Expected one test pair')
    row = rows[0]; pair_path = pairs/row['path']
    require(file_record(pair_path) == {'sha256': row['sha256'], 'bytes': row['bytes']}, 'Pair file changed')
    plan_row = next(r for r in plan['entries'] if r['sample_id'] == SAMPLE_ID)
    require(plan_row['source_video_sha256'] == row['source_video_sha256'], 'Source identity differs')
    source_path = Path(plan_row['source_video_path'])
    require(sha256(source_path) == row['source_video_sha256'], 'Original source video changed')
    done_path = current/'decoded/test'/SAMPLE_ID/'done.json'; done = read_json(done_path)
    cache_dir = current/'baseline_cache/test'/SAMPLE_ID
    cache_done_path = cache_dir/'done.json'; cache_done = read_json(cache_done_path)
    require(done['status'] == cache_done['status'] == 'completed', 'Current comparison is incomplete')
    require(done['sample_id'] == cache_done['sample_id'] == SAMPLE_ID and done['split'] == cache_done['split'] == 'test', 'Receipt sample mismatch')
    require(done['pair_sha256'] == cache_done['pair_sha256'] == row['sha256'], 'Current comparison uses another pair')
    require(done['contract_sha256'] == cache_done['contract_sha256'] == sha256(evaluation_path), 'Current evaluation contract differs')
    require(file_record(cache_done_path) == done['baseline_cache_receipt'], 'Baseline cache receipt changed')
    require(file_record(cache_dir/'rgb.pt') == cache_done['files']['rgb.pt'], 'Baseline RGB cache changed')
    base_model = Path(args.base_model or plan['base_model'])
    decoder_weights = vae_fingerprint(base_model)
    require(decoder_weights == evaluation['decoder_fingerprint']['weights'], 'Frozen decoder assets changed')
    module = importlib.import_module('diffusers.models.autoencoders.autoencoder_kl_minimax_h3')
    decoder_impl = Path(inspect.getfile(module))
    require(sha256(decoder_impl) == evaluation['decoder_fingerprint']['implementation_sha256'], 'Decoder implementation changed')
    original = read_json(root/'upscaler_conv3d_hybrid/kd_pixel_steps2000/contract.json')['original_upscaler_fingerprint']
    upstream = root/'upstream'; upscaler = root/'assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors'
    require(sha256(upscaler) == original['base_weights_sha256'], 'Original upscaler weights changed')
    require(sha256(upstream/'open_h3/upscaler.py') == original['loader_sha256'] and
            sha256(upstream/'open_h3/upscaler_node.py') == original['loaded_node_sha256'], 'Upstream loader/model source changed')
    identities = {'initial_checkpoint': initial, 'prior50_summary': prior_path, 'old_5s_summary': old_path,
        'current_evaluation_contract': evaluation_path, 'current_done_receipt': done_path,
        'baseline_rgb_receipt': cache_done_path, 'baseline_rgb': cache_dir/'rgb.pt',
        'pair': pair_path, 'source_video': source_path, 'plan': plan_path, 'ready': ready_path, 'test_manifest': manifest_path,
        'upscaler_base_weights': upscaler, 'upstream_loader': upstream/'open_h3/upscaler.py',
        'upstream_node': upstream/'open_h3/upscaler_node.py', 'decoder_implementation': decoder_impl}
    contract = dict(schema_version=1, sample_id=SAMPLE_ID, split='test', variant='zero_aligned', step=0,
        new_training_updates=0, initial_sha256=INITIAL_SHA, adapter_config=prior['adapter_config'],
        assets={k: dict(path=str(p), **file_record(p)) for k,p in identities.items()},
        imported_source_files=imported_sources(), decoder_weights=decoder_weights,
        expected_execution_precision=evaluation['execution_precision'],
        target_pixels=[672,384], frames=120, fps=24, duration_seconds=5,
        normalization='D(U(N(native_normalized_H3_latent)))', target_latent_shape=[24,37,24,42],
        source_video_sha256=row['source_video_sha256'], pair_sha256=row['sha256'],
        base_model=str(base_model), output=str(args.output),
        checkpoint_loading='Load saved step0 state verbatim; install causal operators; NEVER repeat aligned_past conversion',
        scope='One previously viewed test clip; fixed weights, no training or checkpoint selection; VAE decoder noncausal')
    return contract, dict(initial=initial, pair_path=pair_path, base_model=base_model,
        upstream=upstream, upscaler=upscaler, cache_dir=cache_dir)


def verify_frozen(model, versions):
    require(frozen_versions(model) == versions, 'Frozen parameter versions changed')
    require(all(not p.requires_grad and p.grad is None and p.dtype == torch.float32 for p in model.parameters()), 'Nonfrozen or non-FP32 parameter')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--base-model')
    parser.add_argument('--device',default='npu:0')
    parser.add_argument('--stage',choices=('check','infer','decode','all'),default='all')
    args=parser.parse_args();args.root=args.root.resolve()
    args.output=(args.output or args.root/'upscaler_v5_motion_20261106/diagnostics/zero_aligned_0086441').resolve()
    torch.set_num_threads(4)
    contract, paths = context(args)
    if args.stage=='check':
        print(json.dumps({'status':'inputs_verified','sample_id':SAMPLE_ID,'initial_sha256':INITIAL_SHA,
            'pair_sha256':contract['pair_sha256'],'output':str(args.output),'expected_execution_precision':contract['expected_execution_precision']}),flush=True)
        return
    if args.stage in ('infer','all'):
        require(not args.output.exists(), 'Output exists; preserve evidence and choose a new output')
        args.output.mkdir(parents=True)
        exclusive_json(args.output/'contract.json',contract)
    else:
        require(read_json(args.output/'contract.json')==contract, 'Frozen completion contract differs')
    contract_sha=sha256(args.output/'contract.json')
    started=time.perf_counter()
    try:
        device=select_device(args.device); precision=execution_precision(device)
        require(precision==contract['expected_execution_precision'], 'Effective precision differs; do not silently change HF32 flags')
        sample=tensor_load(paths['pair_path'])
        require(tuple(sample['draft'].shape)==(24,37,12,21) and tuple(sample['reference'].shape)==(24,37,24,42) and
                tuple(sample['source_pixels'].shape)==(120,384,672,3), 'Pair geometry differs')
        if args.stage in ('infer','all'):
            saved=tensor_load(paths['initial']);require(saved['step']==0 and saved['adapter_config']==contract['adapter_config'], 'Expected historical aligned step0 state');del saved
            loader_args=argparse.Namespace(upstream=paths['upstream'],upscaler=paths['upscaler'])
            node,model,adapter=reload_causal_model(loader_args,paths['initial'],device,contract['adapter_config'])
            require(sum(p.numel() for p in model.parameters())==345280216,'Wrong model architecture')
            versions=frozen_versions(model)
            draft=sample['draft'][None].to(device,torch.float32)
            with torch.no_grad():
                prediction=adapter.native_forward(draft,(37,24,42)).float().cpu()
                checks=causal_checks(adapter,draft,prediction,(37,24,42),1e-4)
            synchronize(device);verify_frozen(model,versions)
            require(torch.isfinite(prediction).all().item(), 'Nonfinite prediction')
            exclusive_json(args.output/'causality.json',checks)
            require(checks['passed'],'Fixed-input causal checks failed; evidence preserved')
            atomic_torch_save(args.output/'zero_aligned.pt',dict(latent=prediction[0],sample_id=SAMPLE_ID,
                split='test',variant='zero_aligned',step=0,pair_sha256=contract['pair_sha256'],
                checkpoint_sha256=INITIAL_SHA,contract_sha256=contract_sha))
            exclusive_json(args.output/'inference_done.json',dict(status='completed',sample_id=SAMPLE_ID,
                contract_sha256=contract_sha,latent=file_record(args.output/'zero_aligned.pt'),
                causality=file_record(args.output/'causality.json'),execution_precision=execution_precision(device),
                frozen_parameters_unchanged=True,new_training_updates=0))
            adapter.remove();model.to('cpu');node.MODEL_CACHE.clear()
            del adapter,model,node,draft,prediction;empty_cache(device)
        if args.stage in ('decode','all'):
            inference=read_json(args.output/'inference_done.json')
            require(inference['status']=='completed' and inference['sample_id']==SAMPLE_ID and inference['contract_sha256']==contract_sha,'Inference receipt mismatch')
            require(inference['latent']==file_record(args.output/'zero_aligned.pt') and inference['causality']==file_record(args.output/'causality.json'),'Inference evidence changed')
            require(inference['execution_precision']==precision,'Inference/decode precision differs')
            require(not (args.output/'zero_aligned.mp4').exists() and not (args.output/'reference_parity.json').exists(),'Decode output exists; refuse overwrite')
            vae=load_video_vae(paths['base_model'],device);versions=frozen_versions(vae)
            cache=tensor_load(paths['cache_dir']/'rgb.pt')
            require(cache['sample_id']==SAMPLE_ID and cache['pair_sha256']==contract['pair_sha256'],'RGB cache sample mismatch')
            reference=decode_latent(vae,sample['reference'],device,keep_frames=120)
            parity=dict(equal=torch.equal(reference,cache['reference']),max_abs=int((reference.to(torch.int16)-cache['reference'].to(torch.int16)).abs().max()),
                shape=list(reference.shape),dtype=str(reference.dtype),baseline_cache=contract['assets']['baseline_rgb'])
            exclusive_json(args.output/'reference_parity.json',parity)
            require(parity['equal'],'Reference decoder differs from current comparison; evidence preserved')
            latent=tensor_load(args.output/'zero_aligned.pt')
            require(latent['sample_id']==SAMPLE_ID and latent['contract_sha256']==contract_sha,'Latent identity mismatch')
            rgb=decode_latent(vae,latent['latent'],device,keep_frames=120)
            require(tuple(rgb.shape)==(120,384,672,3) and rgb.dtype==torch.uint8,'Wrong decoded RGB geometry')
            synchronize(device);verify_frozen(vae,versions)
            codec=write_mp4(rgb,args.output/'zero_aligned.mp4',fps=24.,crf=16)
            del vae;empty_cache(device)
            media=audit_video(args.output/'zero_aligned.mp4',672,384)
            end=execution_precision(device);require(end==precision,'Precision changed during execution')
            require(context(args)[0]==contract,'Sources/assets changed during execution')
            exclusive_json(args.output/'summary.json',dict(status='completed',sample_id=SAMPLE_ID,
                contract_sha256=contract_sha,checkpoint_sha256=INITIAL_SHA,step=0,new_training_updates=0,
                latent=file_record(args.output/'zero_aligned.pt'),reference_parity=parity,
                media_audit={'zero_aligned.mp4':media},video_codec=codec,frames=120,fps=24,seconds_per_video=5,
                execution_precision_start=precision,execution_precision_end=end,
                frozen_model_and_decoder_unchanged=True,source_contract_unchanged=True,
                elapsed_seconds=time.perf_counter()-started))
            print(json.dumps({'status':'completed','video':str(args.output/'zero_aligned.mp4'),'media_audit':media}),flush=True)
        else:
            require(execution_precision(device)==precision and context(args)[0]==contract,'Inference precision/source changed')
            print(json.dumps({'status':'inference_completed','output':str(args.output)}),flush=True)
    except BaseException as error:
        failure=args.output/f'failure_{time.time_ns()}.json'
        exclusive_json(failure,dict(status='failed',stage=args.stage,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
