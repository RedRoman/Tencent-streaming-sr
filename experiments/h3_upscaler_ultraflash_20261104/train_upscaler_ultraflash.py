"""Larger H3 causal-upscaler adaptation using Ultra Flash Eq.3 loss family.

This retains H3's published causalized Conv3d network. It is not an UltraFlash
architecture/recipe reproduction. Test clips are not opened until selection by
validation latent MSE is complete. All unreported paper choices are recorded.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import time
from pathlib import Path
import torch
from data_utils import select_device, synchronize, tensor_load, empty_cache
from prepare_data import load_author_upscaler
from causal_upscaler import CausalUpscalerAdapter
from train_upscaler_trial import (atomic_json, atomic_torch_save, enable_activation_checkpointing,
                                 parameter_sample, memory_stats, sha256, to_cpu)
from ultraflash_ewarp import losses, FLOW_VERSION

def load_pairs(manifest):
    records=[]
    for row in [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]:
        path=Path(row['path']);path=path if path.is_absolute() else manifest.parent/path
        value=tensor_load(path)
        fp=manifest.parent/'flow'/path.name;flow=tensor_load(fp)
        if flow['metadata']['version']!=FLOW_VERSION or flow['pair_sha256']!=sha256(path):raise ValueError('Stale flow cache')
        if tuple(value['draft'].shape)!=(24,37,12,21) or tuple(value['reference'].shape)!=(24,37,24,42):raise ValueError('Expected full5s pair')
        if value['metadata']['split']!=row['split']:raise ValueError('Split metadata mismatch')
        if value['metadata'].get('sample_id',value['metadata'].get('id'))!=row['sample_id']:raise ValueError('Sample identity mismatch')
        if row.get('sha256',flow['pair_sha256'])!=flow['pair_sha256']:raise ValueError('Pair differs from finalized manifest')
        if row.get('source_video_sha256',value['metadata']['source_video_sha256'])!=value['metadata']['source_video_sha256']:raise ValueError('Source identity mismatch')
        records.append({'sample_id':row['sample_id'],'split':row['split'],'draft':value['draft'].float(),
            'reference':value['reference'].float(),'flow':flow['flow'],'valid_pairs':flow['valid_pairs'],
            'pair_sha256':flow['pair_sha256'],'source_sha256':value['metadata']['source_video_sha256']})
        del value,flow
    if not records:raise ValueError('Empty manifest')
    return records

def objective(pred,target,flow,valid,family,weight):
    result=losses(pred,target,flow,valid,weight,valid_frames=36)
    if family=='temporal':
        d=pred[:,:,1:36]-pred[:,:,:35]
        dt=target[:,:,1:36]-target[:,:,:35]
        temporal=(d.float()-dt.float()).square().mean()
        result['loss']=result['latent_mse']+weight*temporal
        result['temporal_difference_mse']=temporal
    return result

def tensors(pair,device):
    return [pair[k][None].to(device) for k in ('draft','reference','flow','valid_pairs')]

def evaluate(adapter,pairs,device,family,weight,save_dir=None):
    records={}
    with torch.no_grad():
        for pair in pairs:
            draft,reference,flow,valid=tensors(pair,device)
            pred=adapter.native_forward(draft,reference.shape[-3:])
            result=objective(pred,reference,flow,valid,family,weight)
            records[pair['sample_id']]={k:float(v) for k,v in result.items()}
            if save_dir:
                atomic_torch_save(save_dir/f"{pair['sample_id']}.pt",{'latent':pred[0].cpu(),'sample_id':pair['sample_id']})
    return {'samples':records,'mean':{k:sum(v[k] for v in records.values())/len(records) for k in next(iter(records.values()))}}

def create_model(args,device,checkpoint):
    saved=tensor_load(checkpoint)
    cfg=saved['adapter_config']
    if cfg['initialization']!='aligned_past' or cfg['groupnorm']!='per_frame':raise ValueError('Wrong initial adapter')
    node=load_author_upscaler(args.upstream,args.upscaler,device)
    model=node.load_model('upscaler',device,'fp32')
    model.load_state_dict(saved['model_state_dict'],strict=True)
    model._aligned_past_initialized=True
    adapter=CausalUpscalerAdapter(node,model)
    return node,model,adapter,cfg

def main():
    root=Path(os.environ.get('EXP_ROOT',Path(__file__).resolve().parents[1]))
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',type=Path,default=root/'upscaler_ultraflash/pairs')
    p.add_argument('--initial',type=Path,default=root/'upscaler_causal_finetune/aligned_past_steps50/checkpoint/model.pt')
    p.add_argument('--upstream',type=Path,default=root/'upstream')
    p.add_argument('--upscaler',type=Path,default=root/'assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='npu:0')
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--min-lr',type=float,default=1e-6);p.add_argument('--warmup',type=int,default=100)
    p.add_argument('--weight',type=float,default=.1);p.add_argument('--family',choices=['ewarp','temporal'],default='ewarp')
    p.add_argument('--validate-every',type=int,default=200);p.add_argument('--save-every',type=int,default=200)
    p.add_argument('--seed',type=int,default=20261105);p.add_argument('--resume',type=Path)
    p.add_argument('--preflight',action='store_true')
    args=p.parse_args();torch.set_num_threads(4);torch.manual_seed(args.seed)
    if args.output.exists() and not args.resume:raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(dtype='fp32',optimizer='AdamW',weight_decay=.01,grad_clip=1.,batch_size=1,
        train_latent_frames=37,loss_valid_latent_frames=36,source_frames=120,output_pixels=[672,384],
        recipe='Ultra Flash inspired H3 causal-upscaler adaptation; original network retained',
        paper='https://arxiv.org/html/2606.09150v4#S2.SS2',flow_version=FLOW_VERSION,
        initialization='Prior 50-update checkpoint; fresh optimizer and schedule',
        paper_missing_choices='Flow estimator/lambda/masks/reduction/batch/steps unpublished; these are experiment defaults',
        selection='Lowest mean validation latent MSE, includes step0; test never selects checkpoint',
        causal_scope='Fixed H3 latent inputs only; H3 decoder remains noncausal',
        schedule=f'{args.warmup}-step linear warmup then cosine to {args.min_lr}; adapted low LR for pretrained345M H3',
        prior_updates=50,activation_checkpointing=True)
    if not args.preflight:
        ready=json.loads((args.pairs/'READY.json').read_text())
        if ready['status']!='ready' or ready['plan_sha256']!=sha256(args.pairs/'plan.json'):raise ValueError('Dataset not finalized')
        for split in ('train','dev','test','legacy'):
            if sha256(args.pairs/f'{split}.jsonl')!=ready['manifests'][split]['sha256']:raise ValueError('Manifest changed after finalize')
    train=load_pairs(args.pairs/'train.jsonl');dev=load_pairs(args.pairs/'dev.jsonl')
    if set(x['sample_id'] for x in train)&set(x['sample_id'] for x in dev):raise ValueError('Train/dev overlap')
    if set(x['source_sha256'] for x in train)&set(x['source_sha256'] for x in dev):raise ValueError('Source train/dev overlap')
    plan_sha=sha256(args.pairs/'plan.json')
    contract={k:config[k] for k in ('steps','lr','min_lr','warmup','weight','family','seed','validate_every','save_every','preflight')}
    contract.update(plan_sha256=plan_sha,initial_sha256=sha256(args.initial),
        train_manifest_sha256=sha256(args.pairs/'train.jsonl'),dev_manifest_sha256=sha256(args.pairs/'dev.jsonl'),
        upscaler_sha256=sha256(args.upscaler),flow_version=FLOW_VERSION,
        training_source_sha256=sha256(Path(__file__)),loss_source_sha256=sha256(Path(__file__).with_name('ultraflash_ewarp.py')))
    device=select_device(args.device)
    node,model,adapter,adapter_config=create_model(args,device,args.resume or args.initial)
    enable_activation_checkpointing(model);model.eval().requires_grad_(True)
    params=list(model.parameters());optimizer=torch.optim.AdamW(params,lr=args.lr,weight_decay=.01,foreach=False)
    schedule=[];rng=random.Random(args.seed)
    while len(schedule)<args.steps:
        ids=list(range(len(train)));rng.shuffle(ids);schedule.extend(ids)
    summary={'status':'running','config':config,'parameter_count':sum(v.numel() for v in params),
        'plan_sha256':plan_sha,'initial_checkpoint_sha256':sha256(args.initial),'adapter_config':adapter_config,
        'train_count':len(train),'dev_count':len(dev),'validation':[],'test_opened':False}
    start_step=0;best=float('inf');best_step=0;records=[]
    if args.resume:
        if args.resume.resolve().parent!=args.output.resolve() or not (args.output/'steps.jsonl').exists():
            raise ValueError('Resume requires the original output directory and logs')
        ck=tensor_load(args.resume)
        if ck.get('contract')!=contract:raise ValueError('Exact training/config/data contract differs')
        if not (args.output/ck['best_file']).exists() or sha256(args.output/ck['best_file'])!=ck['best_sha256']:
            raise ValueError('Missing or changed immutable best checkpoint')
        if ck['plan_sha256']!=plan_sha or ck['family']!=args.family or ck['schedule_seed']!=args.seed:raise ValueError('Resume contract mismatch')
        optimizer.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['torch_rng'])
        start_step=ck['step'];best=ck['best_metric'];best_step=ck['best_step']
        summary=ck['summary'];records=ck['records']
        (args.output/'steps.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in records))
        if device.type=='npu' and ck.get('device_rng') is not None:torch.npu.set_rng_state(ck['device_rng'],device)
    atomic_json(args.output/'config.json',config)
    def save(path,step,with_optimizer=False):
        value={'model_state_dict':to_cpu(model.state_dict()),'step':step,'adapter_config':adapter_config,
            'prior_updates':50,'plan_sha256':plan_sha,'family':args.family,'schedule_seed':args.seed,
            'best_metric':best,'best_step':best_step,'contract':contract}
        if with_optimizer:
            best_file=f'best_step_{best_step:06d}.pt'
            value.update(optimizer=to_cpu(optimizer.state_dict()),torch_rng=torch.get_rng_state(),
                device_rng=torch.npu.get_rng_state(device) if device.type=='npu' else None,
                summary=summary,records=records,best_file=best_file,best_sha256=sha256(args.output/best_file))
        atomic_torch_save(path,value)
    if not args.resume:
        initial=evaluate(adapter,dev,device,args.family,args.weight)
        best=initial['mean']['latent_mse'];summary['validation'].append({'step':0,**initial})
        save(args.output/'best_step_000000.pt',0)
    synchronize(device);started=time.perf_counter();last_sample=parameter_sample(model)
    atomic_json(args.output/'summary.json',summary)
    with (args.output/'steps.jsonl').open('a' if args.resume else 'w') as handle:
        for step in range(start_step,args.steps):
            tick=time.perf_counter()
            ratio=min(1.,(step+1)/max(1,args.warmup))
            cosine=(1+math.cos(math.pi*max(0,step-args.warmup)/max(1,args.steps-args.warmup)))/2
            lr=(args.min_lr+(args.lr-args.min_lr)*cosine)*ratio
            optimizer.param_groups[0]['lr']=lr
            pair=train[schedule[step]];draft,reference,flow,valid=tensors(pair,device)
            optimizer.zero_grad(set_to_none=True)
            pred=adapter.native_forward(draft,reference.shape[-3:])
            result=objective(pred,reference,flow,valid,args.family,args.weight)
            if not torch.isfinite(result['loss']):raise FloatingPointError('Nonfinite loss')
            result['loss'].backward()
            if any(v.grad is None for v in params):raise RuntimeError('Disconnected parameter')
            norm=torch.nn.utils.clip_grad_norm_(params,1.,foreach=False,error_if_nonfinite=True)
            if not float(norm)>0:raise RuntimeError('Zero gradients')
            optimizer.step();synchronize(device)
            if step==start_step or (step+1)%args.save_every==0:
                sample=parameter_sample(model)
                if not torch.isfinite(sample).all() or torch.equal(sample,last_sample):raise RuntimeError('Invalid update')
                last_sample=sample
            record={'step':step+1,'sample_id':pair['sample_id'],'lr':lr,'grad_norm_before_clip':float(norm),
                    'seconds':time.perf_counter()-tick,**{k:float(v.detach()) for k,v in result.items()}}
            records.append(record);handle.write(json.dumps(record)+'\n');handle.flush()
            if step<5 or (step+1)%20==0:print(json.dumps(record),flush=True)
            del draft,reference,flow,valid,pred,result,norm
            summary['completed_updates']=step+1;summary['elapsed_seconds']=time.perf_counter()-started
            if (step+1)%args.validate_every==0 or step+1==args.steps:
                val=evaluate(adapter,dev,device,args.family,args.weight)
                entry={'step':step+1,**val};summary['validation'].append(entry)
                if val['mean']['latent_mse']<best:
                    best=val['mean']['latent_mse'];best_step=step+1;save(args.output/f'best_step_{best_step:06d}.pt',best_step)
                summary.update(best_validation_mse=best,best_step=best_step)
                print(json.dumps({'validation':step+1,'mean':val['mean'],'best_step':best_step}),flush=True)
            if (step+1)%args.save_every==0 or step+1==args.steps:
                save(args.output/'latest.pt',step+1,with_optimizer=True)
                atomic_json(args.output/'summary.json',summary)
    save(args.output/'final.pt',args.steps)
    summary.update(status='training_completed',training_seconds=time.perf_counter()-started,best_step=best_step,best_validation_mse=best,memory=memory_stats(device))
    atomic_json(args.output/'summary.json',summary)
    if args.preflight:
        summary['status']='preflight_completed';atomic_json(args.output/'summary.json',summary);print(json.dumps(summary),flush=True);return
    selected=args.output/f'best_step_{best_step:06d}.pt'
    link=args.output/'best.pt'
    if link.is_symlink():link.unlink()
    elif link.exists():raise FileExistsError(link)
    link.symlink_to(selected.name)
    # Freeze selection before opening retained test/legacy tensors.
    model.load_state_dict(tensor_load(args.output/'best.pt')['model_state_dict'],strict=True)
    model.requires_grad_(False);adapter.reset_stream()
    test=load_pairs(args.pairs/'test.jsonl');legacy=load_pairs(args.pairs/'legacy.jsonl')
    all_ids=[x['sample_id'] for group in (train,dev,test,legacy) for x in group]
    all_sha=[x['source_sha256'] for group in (train,dev,test,legacy) for x in group]
    if len(all_ids)!=len(set(all_ids)) or len(all_sha)!=len(set(all_sha)):raise ValueError('Split collision')
    summary['test_opened']=True;summary['test']=evaluate(adapter,test,device,args.family,args.weight,args.output/'predictions/test')
    summary['legacy']=evaluate(adapter,legacy,device,args.family,args.weight,args.output/'predictions/legacy')
    summary['best_checkpoint_sha256']=sha256(args.output/'best.pt')
    summary['status']='completed';summary['seconds']=time.perf_counter()-started
    atomic_json(args.output/'summary.json',summary)
    print(json.dumps({'completed':True,'best_step':best_step,'test':summary['test']['mean'],'seconds':summary['seconds']}),flush=True)

if __name__=='__main__':main()
