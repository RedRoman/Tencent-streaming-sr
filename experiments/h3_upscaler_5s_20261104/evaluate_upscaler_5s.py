"""Evaluate existing upscaler weights on genuine five-second source clips.

No additional training. Reuse the original, initial causal, and 50-update causal
weights; encode 120 real source frames and decode/crop to 120 frames at 24 fps.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from data_utils import (decode_latent, empty_cache, load_video_vae, select_device,
                        synchronize, tensor_load, write_mp4)
from decode_results import comparisons
from evaluate_causal_upscaler_pixels import metric
from prepare_data import load_author_upscaler
from train_causal_upscaler import compare, reload_causal_model
from train_upscaler_trial import (atomic_json, metrics, read_pairs, save_latent,
                                 sha256, wrapper_forward)


def causal_checks(adapter, draft, full, target, tolerance):
    checks = {}
    with torch.no_grad():
        for cut in (1, 13, 25):
            changed = draft.clone()
            changed[:, :, cut:] = 3 * torch.randn(changed[:, :, cut:].shape,
                generator=torch.Generator().manual_seed(20261104+cut)).to(draft.device)
            actual = adapter.native_forward(changed, target).cpu()
            checks[f'future_cut{cut}'] = compare(full[:, :, :cut], actual[:, :, :cut], tolerance)
            prefix = adapter.native_forward(draft[:, :, :cut].contiguous(), (cut, *target[-2:])).cpu()
            checks[f'prefix_cut{cut}'] = compare(full[:, :, :cut], prefix, tolerance)
        for chunks in ([5]*7+[2], [1]*37, [7,13,17]):
            adapter.reset_stream()
            outputs, offset = [], 0
            for count in chunks:
                outputs.append(adapter.stream_step(draft[:, :, offset:offset+count].contiguous(), target[-2:]).cpu())
                offset += count
            assert offset == 37
            record = compare(full, torch.cat(outputs, dim=2), tolerance)
            record.update(partition=chunks, cache=adapter.stream_state())
            assert record['cache']['processed_latent_frames'] == 37
            checks['chunks_'+'_'.join(map(str,chunks))] = record
        adapter.reset_stream()
    return {'checks':checks,'passed':all(v['passed'] for v in checks.values())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--prior-trial',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--upscaler',type=Path,required=True)
    p.add_argument('--base-model',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='npu:0')
    p.add_argument('--tolerance',type=float,default=1e-4)
    args=p.parse_args()
    torch.set_num_threads(4)
    if args.output.exists():raise FileExistsError(args.output)
    pairs=read_pairs(args.pairs/'train.jsonl','train')+read_pairs(args.pairs/'holdout.jsonl','holdout')
    ids=[q['sample_id'] for q in pairs]
    if len(ids)!=len(set(ids)):raise ValueError('Overlapping sample IDs')
    for q in pairs:
        if q['draft'].shape!=(24,37,12,21) or q['reference'].shape!=(24,37,24,42):
            raise ValueError('Expected 37 latent frames, 2x spatial pair')
        if q['metadata']['decoded_frames']!=120 or q['metadata']['fps']!=24.:
            raise ValueError('Expected exactly 120 frames at 24 fps')
    prior=json.loads((args.prior_trial/'summary.json').read_text())
    if prior['completed_updates']!=50 or not prior['passed']:
        raise ValueError('Expected the previously validated 50-update checkpoint')
    saved_paths={'causal_before':args.prior_trial/'initial_model.pt',
                 'causal_after':args.prior_trial/'checkpoint/model.pt'}
    hashes={k:sha256(v) for k,v in saved_paths.items()}
    if hashes['causal_after']!=prior['checkpoint']['model_sha256']:
        raise ValueError('Final checkpoint hash mismatch')
    if sha256(args.upscaler)!=prior['data_signature']['released_upscaler_sha256']:
        raise ValueError('Published upscaler hash mismatch')
    args.output.mkdir(parents=True)
    device=select_device(args.device)
    summary={'status':'running','new_training_updates':0,'previous_training_updates':50,
        'previous_training_latent_frames':13,'evaluation_latent_frames':37,
        'source_frames':120,'fps':24.,'output_frames':120,'output_seconds':5.,
        'raw_vae_decoded_frames':124,'vae_internal_pad_frames':16,
        'vae_padded_input_frames':136,'decode_tail_crop_frames':4,'checkpoint_sha256':hashes,
        'variants':{},'causality':{},'pixel_metrics':{},
        'scope':'Five-second evaluation of existing weights; no refiner, no further training. VAE decoder is noncausal; end-to-end pixel causality is not established.',
        'limitations':['These are longer versions of the same three development clips, not unseen videos.',
            'The causal weights were trained on13-frame prefixes, not on these37-frame sequences.',
            'All variants use the same120-frame VAE encoding context; prior43-frame metrics are not directly comparable.']}
    started=time.perf_counter()
    for pair in pairs:save_latent(args.output,pair,'reference',pair['reference'])
    for label in ('pretrained','causal_before','causal_after'):
        if label=='pretrained':
            node=load_author_upscaler(args.upstream,args.upscaler,device)
            model=node.load_model('upscaler',device,'fp32').eval().requires_grad_(False)
            adapter=None
        else:
            node,model,adapter=reload_causal_model(args,saved_paths[label],device,prior['adapter_config'])
        summary['variants'][label]={}
        if adapter:summary['causality'][label]={}
        with torch.no_grad():
            for pair in pairs:
                native=pair['draft'][None].to(device)
                target=tuple(pair['reference'].shape[-3:])
                then=time.perf_counter()
                full=(adapter.native_forward(native,target) if adapter else wrapper_forward(node,model,native,target)).cpu()
                synchronize(device)
                record=metrics(full[0],pair['reference'],.1)
                record.update(seconds=time.perf_counter()-then,split=pair['split'],
                    latent_path=save_latent(args.output,pair,label,full[0]))
                summary['variants'][label][pair['sample_id']]=record
                if adapter:
                    summary['causality'][label][pair['sample_id']]=causal_checks(adapter,native,full,target,args.tolerance)
                print(json.dumps({'evaluation':label,'sample_id':pair['sample_id'],**record}),flush=True)
                del native,full
        if adapter:adapter.remove()
        model.to('cpu');node.MODEL_CACHE.clear()
        del node,model,adapter
        empty_cache(device)
        atomic_json(args.output/'summary.json',summary)
    summary['causality_passed']=all(v['passed'] for samples in summary['causality'].values() for v in samples.values())
    summary['status']='decoding'
    atomic_json(args.output/'summary.json',summary)
    vae=load_video_vae(args.base_model,device)
    for pair in pairs:
        ident=pair['sample_id'];directory=args.output/'decoded'/ident;directory.mkdir(parents=True)
        frames={}
        for label in ('pretrained','causal_before','causal_after','reference'):
            value=tensor_load(args.output/'latents'/ident/f'{label}.pt')['latent']
            frames[label]=decode_latent(vae,value,device,keep_frames=120)
            if frames[label].shape!=(120,384,672,3):raise ValueError('Incorrect decoded video geometry')
            write_mp4(frames[label],directory/f'{label}.mp4',fps=24.)
            print(json.dumps({'decoded':ident,'variant':label,'frames':len(frames[label])}),flush=True)
        source=tensor_load(pair['path'])['source_pixels']
        row={'split':pair['split'],'variants':{}}
        for label in ('pretrained','causal_before','causal_after'):
            row['variants'][label]={'vs_decoded_reference':metric(frames[label],frames['reference']),
                'vs_source_rgb_secondary':metric(frames[label],source)}
        row['reference_vs_source_rgb_secondary']=metric(frames['reference'],source)
        summary['pixel_metrics'][ident]=row
        comparisons(frames,directory,fps=24.,crf=16,comparison_video=True)
        atomic_json(args.output/'summary.json',summary)
        print(json.dumps({'pixel_metrics':ident,**row}),flush=True)
    summary['seconds']=time.perf_counter()-started
    summary['status']='completed'
    atomic_json(args.output/'summary.json',summary)
    print(json.dumps({'completed':True,'causality_passed':summary['causality_passed'],'seconds':summary['seconds']}),flush=True)
    if not summary['causality_passed']:
        raise AssertionError('Causal consistency failed; results preserved')


if __name__=='__main__':main()
