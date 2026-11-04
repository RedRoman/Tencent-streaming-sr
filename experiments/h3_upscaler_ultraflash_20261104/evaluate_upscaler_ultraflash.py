"""Final fixed-split evaluation for longer H3 upscaler adaptation.

Report all retained-test latent metrics; decode a priori selected categories and
one legacy example. Pixel metrics do not claim full-test coverage. No model
selection/optimization occurs here.
"""
from __future__ import annotations
import argparse,json,time,os
from pathlib import Path
import torch
from data_utils import decode_latent,empty_cache,load_video_vae,select_device,synchronize,tensor_load,write_mp4
from decode_results import comparisons
from evaluate_causal_upscaler_pixels import metric
from evaluate_upscaler_5s import causal_checks
from train_causal_upscaler import compare
from prepare_data import load_author_upscaler
from train_upscaler_trial import atomic_json,atomic_torch_save,sha256,wrapper_forward
from train_upscaler_ultraflash import create_model,load_pairs,objective,tensors


def main():
    root=Path(os.environ.get('EXP_ROOT',Path(__file__).resolve().parents[1]))
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',type=Path,default=root/'upscaler_ultraflash/pairs')
    p.add_argument('--ewarp-run',type=Path,required=True);p.add_argument('--control-run',type=Path,required=True)
    p.add_argument('--initial',type=Path,default=root/'upscaler_causal_finetune/aligned_past_steps50/checkpoint/model.pt')
    p.add_argument('--upstream',type=Path,default=root/'upstream');p.add_argument('--upscaler',type=Path,default=root/'assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors')
    p.add_argument('--base-model',default=os.environ.get('H3_BASE_MODEL','/path/to/h3-base'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='npu:0')
    p.add_argument('--finalize',action='store_true')
    p.add_argument('--decode-only',action='store_true');p.add_argument('--shard-index',type=int,default=0);p.add_argument('--num-shards',type=int,default=1)
    args=p.parse_args();torch.set_num_threads(4)
    if not 0<=args.shard_index<args.num_shards:raise ValueError('Invalid shard identity')
    if args.finalize:
        summary=json.loads((args.output/'summary.json').read_text())
        if summary['status']!='latents_completed':raise ValueError('Finalize requires completed latent evaluation')
        digest=sha256(args.output/'summary.json');combined={};seen=set()
        for index in range(args.num_shards):
            done=json.loads((args.output/f'decode_complete_shard{index}.json').read_text())
            path=args.output/f'pixel_metrics_shard{index}.json';part=json.loads(path.read_text())
            expected={v for i,v in enumerate(summary['selected_videos']) if i%args.num_shards==index}
            if done['status']!='completed' or done['shard_index']!=index or done['num_shards']!=args.num_shards:raise ValueError('Shard identity mismatch')
            if done['latent_summary_sha256']!=digest or done['metrics_sha256']!=sha256(path):raise ValueError('Shard input/output fingerprint differs')
            if set(part)!=expected or set(done['samples'])!=expected or seen & expected:raise ValueError('Missing/overlapping decode samples')
            for ident in expected:
                directory=args.output/'decoded'/ident
                for name in ('pretrained','prior50','long_temporal','long_ewarp','reference','comparison'):
                    if not (directory/f'{name}.mp4').is_file():raise FileNotFoundError(directory/name)
                if not (directory/'comparison.png').is_file():raise FileNotFoundError(directory/'comparison.png')
            seen.update(expected);combined.update(part)
        if seen!=set(summary['selected_videos']):raise ValueError('Incomplete selected videos')
        summary.update(status='completed',pixel_metrics=combined,decode_shards=args.num_shards,
            causality_test_count=sum(len(v['checks']) for model in summary['causality'].values() for v in model.values()))
        atomic_json(args.output/'summary.json',summary);print(json.dumps({'completed':True,'decoded_samples':len(seen),'causality_checks':summary['causality_test_count']}));return
    if not args.decode_only:
        if args.output.exists():raise FileExistsError(args.output)
        args.output.mkdir(parents=True)
        runs={k:json.loads((v/'summary.json').read_text()) for k,v in [('ewarp',args.ewarp_run),('control',args.control_run)]}
        if any(v['status']!='completed' for v in runs.values()):raise ValueError('Train runs not finished')
        for label,expected,runpath in [('ewarp','ewarp',args.ewarp_run),('control','temporal',args.control_run)]:
            run=runs[label]
            if run['config']['family']!=expected:raise ValueError('Incorrectly labelled loss family')
            if run['plan_sha256']!=sha256(args.pairs/'plan.json'):raise ValueError('Evaluation plan differs from training')
            if run['initial_checkpoint_sha256']!=sha256(args.initial):raise ValueError('Initialization checkpoint differs')
            if run['best_checkpoint_sha256']!=sha256(runpath/'best.pt'):raise ValueError('Selected checkpoint changed')
        if runs['ewarp']['plan_sha256']!=runs['control']['plan_sha256']:raise ValueError('Different datasets')
        for key in ('steps','lr','min_lr','warmup','weight','seed'):
            if runs['ewarp']['config'][key]!=runs['control']['config'][key]:raise ValueError('Unmatched budget')
        plan=json.loads((args.pairs/'plan.json').read_text())
        # Fixed rule, never use metrics or images to select presentation clips.
        selected=[sorted(r['sample_id'] for r in plan['entries'] if r['split']=='test' and r['category']==c)[0] for c in ('animal','daily','sport')]
        selected+=['VIDGEN5S_0380588']
        pairs=load_pairs(args.pairs/'test.jsonl')+load_pairs(args.pairs/'legacy.jsonl')
        summary={'status':'evaluating','plan_sha256':sha256(args.pairs/'plan.json'),'selected_videos':selected,
            'video_selection':'Lexicographically first retained-test animal/daily/sport + original legacy holdout; fixed rule before scores',
            'train_runs':{k:{q:v[q] for q in ('completed_updates','best_step','best_validation_mse','best_checkpoint_sha256')} for k,v in runs.items()},
            'variants':{},'causality':{},'reload_parity':{},'checkpoint_sha256':{},'frames':120,'fps':24,'seconds_per_video':5,
            'latent_metric_scope':'All16 retained test plus3 legacy; reconstruction first36 real-anchor latents, warp35 valid pairs',
            'pixel_metric_scope':'Three fixed test examples and one legacy example, not full-test pixel evaluation',
            'causal_scope':'upscaler on fixed latents only; decoder remains noncausal'}
        device=select_device(args.device)
        checkpoints={'prior50':args.initial,'long_temporal':args.control_run/'best.pt','long_ewarp':args.ewarp_run/'best.pt'}
        for label in ('pretrained','prior50','long_temporal','long_ewarp'):
            if label=='pretrained':
                node=load_author_upscaler(args.upstream,args.upscaler,device);model=node.load_model('upscaler',device,'fp32').eval().requires_grad_(False);adapter=None
            else:
                node,model,adapter,_=create_model(args,device,checkpoints[label]);model.eval().requires_grad_(False)
                summary['checkpoint_sha256'][label]=sha256(checkpoints[label]);summary['causality'][label]={}
            summary['variants'][label]={}
            if label.startswith('long_'):summary['reload_parity'][label]={}
            with torch.no_grad():
                for pair in pairs:
                    draft,ref,flow,valid=tensors(pair,device)
                    start=time.perf_counter()
                    pred=adapter.native_forward(draft,ref.shape[-3:]) if adapter else wrapper_forward(node,model,draft,ref.shape[-3:])
                    synchronize(device);seconds=time.perf_counter()-start
                    if label.startswith('long_'):
                        run=args.ewarp_run if label=='long_ewarp' else args.control_run
                        expected=tensor_load(run/'predictions'/pair['split']/f"{pair['sample_id']}.pt")['latent']
                        parity=compare(pred[0].cpu(),expected,1e-4)
                        summary['reload_parity'][label][pair['sample_id']]=parity
                        if not parity['passed']:raise AssertionError('Independent checkpoint reload differs')
                    res=objective(pred,ref,flow,valid,'ewarp',.1)
                    summary['variants'][label][pair['sample_id']]={'split':pair['split'],'seconds':seconds,**{k:float(v) for k,v in res.items()}}
                    if pair['sample_id'] in selected:
                        atomic_torch_save(args.output/'latents'/pair['sample_id']/f'{label}.pt',{'latent':pred[0].cpu()})
                        if label=='pretrained':atomic_torch_save(args.output/'latents'/pair['sample_id']/'reference.pt',{'latent':ref[0].cpu()})
                        if adapter:summary['causality'][label][pair['sample_id']]=causal_checks(adapter,draft,pred.cpu(),ref.shape[-3:],1e-4)
            if adapter:adapter.remove()
            model.to('cpu');node.MODEL_CACHE.clear();del node,model,adapter;empty_cache(device)
            atomic_json(args.output/'summary.json',summary);print(json.dumps({'variant_done':label}),flush=True)
        checks=[v for a in summary['causality'].values() for v in a.values()]
        summary['causality_passed']=all(v['passed'] for v in checks)
        summary['status']='latents_completed';atomic_json(args.output/'summary.json',summary)
        if not summary['causality_passed']:raise AssertionError('Causality failed; retain evidence')
        return
    summary=json.loads((args.output/'summary.json').read_text())
    if summary['status'] not in ('latents_completed','completed'):raise ValueError('Latents not finalized')
    device=select_device(args.device);vae=load_video_vae(args.base_model,device)
    latent_summary_hash=sha256(args.output/'summary.json')
    output={}
    for index,ident in enumerate(summary['selected_videos']):
        if index%args.num_shards!=args.shard_index:continue
        dst=args.output/'decoded'/ident;dst.mkdir(parents=True)
        pairsample=tensor_load(args.pairs/'samples'/f'{ident}.pt');source=pairsample['source_pixels']
        pixels={}
        for name in ('pretrained','prior50','long_temporal','long_ewarp','reference'):
            latent=tensor_load(args.output/'latents'/ident/f'{name}.pt')['latent']
            rgb=decode_latent(vae,latent,device,keep_frames=120)
            if rgb.shape[0]!=120:raise ValueError('Wrong decode length')
            write_mp4(rgb,dst/f'{name}.mp4',fps=24.);pixels[name]=rgb
            print(json.dumps({'decoded':ident,'variant':name}),flush=True)
        # Existing helper supports an arbitrary ordered list of columns.
        comparisons(pixels,dst,fps=24.,crf=16,comparison_video=True)
        output[ident]={name:{'vs_decoded_reference':metric(value,pixels['reference']),
                           'vs_source_rgb_secondary':metric(value,source)} for name,value in pixels.items() if name!='reference'}
        atomic_json(args.output/f'pixel_metrics_shard{args.shard_index}.json',output)
        print(json.dumps({'pixel_done':ident,'metrics':output[ident]}),flush=True)
    atomic_json(args.output/f'decode_complete_shard{args.shard_index}.json',{'status':'completed','samples':list(output),'shard_index':args.shard_index,'num_shards':args.num_shards,
        'latent_summary_sha256':latent_summary_hash,'metrics_sha256':sha256(args.output/f'pixel_metrics_shard{args.shard_index}.json')})

if __name__=='__main__':main()
