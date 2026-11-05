"""Train an UltraFlash-inspired H3 CMN upsampler, independently of the decoder.

This is a new architecture experiment, not the unpublished UltraFlash recipe.
The existing frozen H3 latent pairs, split identities and flow caches are reused.
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
import torch.nn.functional as F
from data_utils import select_device, synchronize, tensor_load
from train_upscaler_trial import atomic_json, atomic_torch_save, sha256, to_cpu, memory_stats
from train_upscaler_ultraflash import load_pairs, tensors
from ultraflash_ewarp import losses, FLOW_VERSION
from h3_cmn import H3LatentUpsampler


def bilinear(x):
    b,c,t,h,w=x.shape
    y=F.interpolate(x.permute(0,2,1,3,4).reshape(b*t,c,h,w),scale_factor=2,
                    mode='bilinear',align_corners=False)
    return y.reshape(b,t,c,h*2,w*2).permute(0,2,1,3,4).contiguous()


def objective(pred,target,flow,valid,weight):
    return losses(pred,target,flow,valid,warp_weight=weight,valid_frames=36)


@torch.no_grad()
def evaluate(model,pairs,device,weight,save_dir=None):
    model.eval() if hasattr(model,'eval') else None
    records={}
    for pair in pairs:
        draft,reference,flow,valid=tensors(pair,device)
        pred=model(draft)
        value=objective(pred,reference,flow,valid,weight)
        records[pair['sample_id']]={k:float(v) for k,v in value.items()}
        if save_dir:
            atomic_torch_save(save_dir/f"{pair['sample_id']}.pt",{'latent':pred[0].cpu(),
                'sample_id':pair['sample_id'],'split':pair['split'],'pair_sha256':pair['pair_sha256']})
    return {'samples':records,'mean':{k:sum(v[k] for v in records.values())/len(records)
                                    for k in next(iter(records.values()))}}


@torch.no_grad()
def causal_checks(model,x,tolerance=3e-5):
    model.eval();full=model(x);checks={}
    def record(name,a,b):
        diff=(a-b).abs();passed=torch.allclose(a,b,atol=tolerance,rtol=tolerance)
        checks[name]={'passed':bool(passed),'max_abs':float(diff.max()),'mean_abs':float(diff.mean())}
    for cut in (1,13,25):
        changed=x.clone();changed[:,:,cut:]=changed[:,:,cut:]*-2.3+1.7
        record(f'future_{cut}',full[:,:,:cut],model(changed)[:,:,:cut])
        record(f'prefix_{cut}',full[:,:,:cut],model(x[:,:,:cut]))
    for chunks in ([1]*37,[5]*7+[2],[7,13,17]):
        state=None;out=[];start=0
        for count in chunks:
            y,state=model.forward_chunk(x[:,:,start:start+count],state,latent_index=start)
            out.append(y);start+=count
        record('chunks_'+'_'.join(map(str,chunks)),full,torch.cat(out,dim=2))
    return {'passed':all(v['passed'] for v in checks.values()),'checks':checks}


def main():
    root=Path(os.environ.get('EXP_ROOT',Path(__file__).resolve().parents[1]))
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',type=Path,default=root/'upscaler_ultraflash/pairs')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='npu:0')
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--width',type=int,default=64)
    p.add_argument('--blocks',type=int,default=2);p.add_argument('--weight',type=float,default=.1)
    p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--min-lr',type=float,default=2e-5)
    p.add_argument('--warmup',type=int,default=200);p.add_argument('--validate-every',type=int,default=500)
    p.add_argument('--seed',type=int,default=20261105);p.add_argument('--resume',type=Path)
    p.add_argument('--preflight',action='store_true')
    args=p.parse_args();torch.set_num_threads(4);torch.manual_seed(args.seed)
    if args.steps<1 or args.validate_every<1:raise ValueError('Positive step counts required')
    if args.output.exists() and not args.resume:raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    ready=json.loads((args.pairs/'READY.json').read_text())
    if ready['status']!='ready' or ready['plan_sha256']!=sha256(args.pairs/'plan.json'):
        raise ValueError('Data plan not finalized')
    for split in ('train','dev','test','legacy'):
        if sha256(args.pairs/f'{split}.jsonl')!=ready['manifests'][split]['sha256']:
            raise ValueError('Manifest changed')
    train=load_pairs(args.pairs/'train.jsonl');dev=load_pairs(args.pairs/'dev.jsonl')
    ids=[x['sample_id'] for x in train+dev];sources=[x['source_sha256'] for x in train+dev]
    if len(ids)!=len(set(ids)) or len(sources)!=len(set(sources)):raise ValueError('Train/dev collision')
    if args.preflight:train=train[:2];dev=dev[:1]
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='resume'}
    config.update(recipe='UltraFlash-inspired new CMN for H3; not an exact recipe reproduction',
        initialization='New random CMN; zero residual head over bilinear interpolation',
        spatial_factors=[1,2,1],temporal_factors=[1,1,1],latent_channels=24,
        dtype='fp32',batch_size=1,optimizer='AdamW',weight_decay=.01,grad_clip=1.,
        source_frames=120,fps=24,train_latent_frames=37,loss_valid_latent_frames=36,
        selection='Lowest dev latent MSE, including step0; loss and test do not select checkpoints',
        scope='Only upscaler trained. Frozen existing H3 decoder; no refiner/DMD updates',
        flow_version=FLOW_VERSION,flow_lambda=args.weight,
        test_reuse='The retained16 test clips have appeared in the earlier Conv3d comparison; not a new blind test',
        padding='Latent36 is padding-only and excluded from loss, retained as context for the original noncausal decoder',
        paper='https://arxiv.org/html/2606.09150v4#S2.SS2')
    contract={**config,'plan_sha256':sha256(args.pairs/'plan.json'),
        'train_manifest_sha256':sha256(args.pairs/'train.jsonl'),'dev_manifest_sha256':sha256(args.pairs/'dev.jsonl'),
        'source_sha256':{name:sha256(Path(__file__).with_name(name)) for name in
            ('train_h3_cmn.py','h3_cmn.py','ultraflash_ewarp.py','train_upscaler_ultraflash.py')}}
    atomic_json(args.output/'config.json',config)
    device=select_device(args.device);model=H3LatentUpsampler(width=args.width,blocks=args.blocks).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01,foreach=False)
    schedule=[];rng=random.Random(args.seed)
    while len(schedule)<args.steps:
        order=list(range(len(train)));rng.shuffle(order);schedule.extend(order)
    summary={'status':'running','config':config,'contract':contract,'train_count':len(train),'dev_count':len(dev),
        'parameter_count':sum(v.numel() for v in model.parameters()),'validation':[],
        'completed_updates':0,'test_opened':False,'prior_training_updates':0}
    records=[];best=float('inf');best_step=0;start_step=0
    def save(path,step,with_optimizer=False):
        state={'model_state_dict':to_cpu(model.state_dict()),'step':step,'contract':contract,
            'model_config':{'width':args.width,'blocks':args.blocks},'best_step':best_step,'best_mse':best}
        if with_optimizer:
            state.update(optimizer=to_cpu(optimizer.state_dict()),summary=summary,records=records,
                rng=torch.get_rng_state(),best_sha256=sha256(args.output/f'best_step_{best_step:06d}.pt'))
        atomic_torch_save(path,state)
    if args.resume:
        saved=tensor_load(args.resume)
        if saved['contract']!=contract:raise ValueError('Resume training/data/source contract changed')
        model.load_state_dict(saved['model_state_dict'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        summary=saved['summary'];records=saved['records'];start_step=saved['step']
        best=saved['best_mse'];best_step=saved['best_step'];torch.set_rng_state(saved['rng'])
        if saved['best_sha256']!=sha256(args.output/f'best_step_{best_step:06d}.pt'):
            raise ValueError('Selected best checkpoint changed')
        (args.output/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    else:
        initial=evaluate(model,dev,device,args.weight)
        baseline=evaluate(bilinear,dev,device,args.weight)
        summary['bilinear_dev']=baseline;summary['validation'].append({'step':0,**initial})
        if abs(initial['mean']['latent_mse']-baseline['mean']['latent_mse'])>1e-7:
            raise AssertionError('Initial model must equal bilinear')
        best=initial['mean']['latent_mse'];save(args.output/'best_step_000000.pt',0)
        print(json.dumps({'initial_validation':initial['mean'],'params':summary['parameter_count']}),flush=True)
    synchronize(device);started=time.perf_counter()
    atomic_json(args.output/'summary.json',summary)
    with (args.output/'steps.jsonl').open('a' if args.resume else 'w') as handle:
        for step in range(start_step,args.steps):
            tick=time.perf_counter();model.train()
            warm=min(1.,(step+1)/max(1,args.warmup))
            ratio=max(0,step-args.warmup)/max(1,args.steps-args.warmup-1)
            lr=warm*(args.min_lr+(args.lr-args.min_lr)*(1+math.cos(math.pi*ratio))/2)
            optimizer.param_groups[0]['lr']=lr
            pair=train[schedule[step]];draft,reference,flow,valid=tensors(pair,device)
            optimizer.zero_grad(set_to_none=True);pred=model(draft)
            result=objective(pred,reference,flow,valid,args.weight)
            if not bool(torch.isfinite(result['loss'])):raise FloatingPointError('Nonfinite loss')
            result['loss'].backward()
            if any(v.grad is None for v in model.parameters()):raise RuntimeError('Disconnected parameter')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False,error_if_nonfinite=True)
            if not float(norm)>0:raise RuntimeError('Zero gradient')
            optimizer.step();synchronize(device)
            record={'step':step+1,'sample_id':pair['sample_id'],'lr':lr,'grad_norm_before_clip':float(norm),
                'seconds':time.perf_counter()-tick,**{k:float(v.detach()) for k,v in result.items()}}
            records.append(record);handle.write(json.dumps(record)+'\n');handle.flush()
            if step<5 or (step+1)%50==0:print(json.dumps(record),flush=True)
            del draft,reference,flow,valid,pred,result,norm
            summary.update(completed_updates=step+1,elapsed_seconds=time.perf_counter()-started)
            if (step+1)%args.validate_every==0 or step+1==args.steps:
                val=evaluate(model,dev,device,args.weight);summary['validation'].append({'step':step+1,**val})
                if val['mean']['latent_mse']<best:
                    best=val['mean']['latent_mse'];best_step=step+1
                    save(args.output/f'best_step_{best_step:06d}.pt',best_step)
                summary.update(best_step=best_step,best_validation_mse=best)
                save(args.output/'latest.pt',step+1,True);atomic_json(args.output/'summary.json',summary)
                print(json.dumps({'validation':step+1,'mean':val['mean'],'best_step':best_step}),flush=True)
    save(args.output/'final.pt',args.steps)
    summary.update(status='training_completed',training_seconds=time.perf_counter()-started,
                   best_step=best_step,best_validation_mse=best,memory=memory_stats(device))
    selected=args.output/f'best_step_{best_step:06d}.pt'
    model.load_state_dict(tensor_load(selected)['model_state_dict'],strict=True);model.eval()
    summary['causality']=causal_checks(model,dev[0]['draft'][None].to(device))
    if not summary['causality']['passed']:raise AssertionError('Causality check failed')
    atomic_json(args.output/'summary.json',summary)
    if args.preflight:
        summary['status']='preflight_completed';atomic_json(args.output/'summary.json',summary)
        print(json.dumps({'preflight_completed':True,'steps':args.steps,'seconds':summary['training_seconds']}),flush=True);return
    link=args.output/'best.pt'
    if link.is_symlink():link.unlink()
    elif link.exists():raise FileExistsError(link)
    link.symlink_to(selected.name)
    test=load_pairs(args.pairs/'test.jsonl');legacy=load_pairs(args.pairs/'legacy.jsonl')
    all_pairs=train+dev+test+legacy
    if len({r['sample_id'] for r in all_pairs})!=len(all_pairs) or len({r['source_sha256'] for r in all_pairs})!=len(all_pairs):
        raise ValueError('Split collision')
    summary['test_opened']=True
    for split,pairs in [('test',test),('legacy',legacy)]:
        summary[split]=evaluate(model,pairs,device,args.weight,args.output/'predictions'/split)
        summary[f'bilinear_{split}']=evaluate(bilinear,pairs,device,args.weight)
    summary.update(status='completed',best_checkpoint_sha256=sha256(selected),seconds=time.perf_counter()-started)
    atomic_json(args.output/'summary.json',summary)
    print(json.dumps({'completed':True,'best_step':best_step,'test':summary['test']['mean'],
                      'bilinear_test':summary['bilinear_test']['mean'],'seconds':summary['seconds']}),flush=True)


if __name__=='__main__':main()
