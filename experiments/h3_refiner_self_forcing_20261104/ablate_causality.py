"""Frozen-refiner ablation: separate prefix length, static mask and noise strength.

No optimization. The 2x2 matrix keeps all video rows in one block, so its
causal/dense contrast only removes static-query access to video keys.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch_npu
import torch.distributed as dist

from model_runtime import Runtime


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    args = p.parse_args()
    root = Path(args.root)
    out = root/'causality_ablation'
    out.mkdir(exist_ok=True)
    rank, world = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    torch.npu.set_device(rank)
    dist.init_process_group('hccl')
    device = torch.device(f'npu:{rank}')
    torch.manual_seed(20261104)
    torch.npu.manual_seed_all(20261104)
    started = time.time()
    runtime = Runtime(root/'merged_transformer', root/'assets', rank, world)
    runtime.model.eval()
    runtime.set_role('real')
    sample_row = json.loads((root/'data/holdout.jsonl').read_text().splitlines()[0])
    sample = torch.load(sample_row['path'], map_location='cpu', weights_only=False)
    source = sample['upscaled'].to(device).float()
    noise = torch.randn(source.shape, generator=torch.Generator().manual_seed(20261104+991)).to(device)
    records = []
    results = {}
    cases = [('dense13',13,False,.909375), ('static_mask13',13,True,.909375),
             ('dense5',5,False,.909375), ('static_mask5',5,True,.909375),
             ('static_mask5_sigma03',5,True,.3)]
    with torch.no_grad():
        for label, frames, causal, sigma in cases:
            runtime.block_frames = frames
            state = (1-sigma)*source[:,:frames] + sigma*noise[:,:frames]
            value = state + sigma*runtime.forward(state,sample,sigma,causal=causal,block_index=0)
            assert torch.isfinite(value).all()
            results[label] = value.cpu()
            rec = dict(label=label,frames=frames,static_reads_video=not causal,
                       all_video_queries_see_all_video_in_prefix=True,sigma=sigma,
                       rms=float(value.square().mean().sqrt()),seconds=time.time()-started)
            records.append(rec)
            if rank==0:
                print(json.dumps(rec),flush=True)
                # Common five-latent-frame decode removes unequal decoder context.
                torch.save({'latent':value[:,:5].cpu(),'metadata':{'decoded_frames':17,'fps':24},
                            'experiment':rec},out/f'{label}.pt')
    if rank==0:
        ref_dense=torch.load(root/'outputs/dense_one_untrained.pt',weights_only=False,map_location='cpu')['latent']
        ref_causal=torch.load(root/'outputs/causal_one_untrained.pt',weights_only=False,map_location='cpu')['latent']
        dense_error=float((results['dense13']-ref_dense).abs().max())
        causal_error=float((results['static_mask5']-ref_causal[:,:5]).abs().max())
        assert dense_error<1e-4 and causal_error<1e-4,(dense_error,causal_error)
        def delta(a,b):
            x,y=results[a][:,:5],results[b][:,:5]
            return {'rms':float((x-y).square().mean().sqrt()),'max_abs':float((x-y).abs().max())}
        report={'completed':True,'world_size':world,'physical_devices':os.getenv('ASCEND_RT_VISIBLE_DEVICES'),
                'cases':records,'dense13_reproduction_max_abs':dense_error,
                'causal_first_block_reproduction_max_abs':causal_error,
                'static_mask_effect_at13':delta('static_mask13','dense13'),
                'static_mask_effect_at5':delta('static_mask5','dense5'),
                'prefix_effect_dense':delta('dense5','dense13'),
                'prefix_effect_static_mask':delta('static_mask5','static_mask13'),
                'noise_effect_at5_static_mask':delta('static_mask5_sigma03','static_mask5'),
                'limitations':['Single held-out clip; no generalization claim.',
                               'Static mask includes both text and audio query groups; not separately isolated.',
                               'First-block ablation has no self-generated history or mixed video times.',
                               'Latent distances are not perceptual quality metrics.']}
        (out/'summary.json').write_text(json.dumps(report,indent=2))
        torch.save({'latent':sample['upscaled'][:,:5], 'metadata':{'decoded_frames':17,'fps':24}},out/'upscaled.pt')
    dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':
    main()
