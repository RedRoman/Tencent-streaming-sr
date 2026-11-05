"""Warm-start H3 CMN with HR supervision, frozen-teacher KD and decoder-aware pixels.

The decoder and LPIPS network are frozen. Only the causal upscaler is optimized.
This is one combined adaptation run, not an ablation or a DMD experiment.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import inspect
import time
from pathlib import Path
import torch
from data_utils import select_device, synchronize, tensor_load, load_video_vae
from train_upscaler_trial import atomic_json, atomic_torch_save, sha256, to_cpu, memory_stats
from train_upscaler_ultraflash import load_pairs, tensors
from train_h3_cmn import objective, evaluate, causal_checks, bilinear
from h3_cmn import H3LatentUpsampler
from h3_hybrid_pixel import PixelSupervisor


def attach_teacher(pairs, cache, ready):
    for pair in pairs:
        rec = ready['records'][pair['sample_id']]
        path = cache/rec['path']
        if sha256(path) != rec['sha256'] or rec['pair_sha256'] != pair['pair_sha256']:
            raise ValueError('Teacher/pair fingerprint mismatch')
        value = tensor_load(path)
        if value['metadata']['split'] != pair['split'] or value['metadata']['pair_sha256'] != pair['pair_sha256']:
            raise ValueError('Wrong teacher identity')
        if value['latent'].shape != pair['reference'].shape or not torch.isfinite(value['latent']).all():
            raise ValueError('Invalid teacher tensor')
        pair['teacher'] = value['latent'].float()


def frozen_versions(*models):
    return tuple((id(p), p._version) for model in models for p in model.parameters())


@torch.no_grad()
def validate(model, pairs, supervisor, device, warp_weight, lpips_weight):
    model.eval()
    rows = {}
    for i, pair in enumerate(pairs):
        draft, reference, flow, valid = tensors(pair, device)
        pred = model(draft)
        values = objective(pred, reference, flow, valid, warp_weight)
        # One predeclared full-spatial temporal block per dev clip, unchanged across updates.
        spec = supervisor.full_specs[i % len(supervisor.full_specs)]
        pixels = supervisor.losses(pred, reference, spec)
        kd = (pred[:, :, :36] - pair['teacher'][None, :, :36].to(device)).square().mean()
        rec = {k:float(v) for k,v in values.items()}
        rec.update({k:float(v) for k,v in pixels.items() if torch.is_tensor(v) and v.numel()==1})
        rec.update(teacher_mse=float(kd), selection_score=float(pixels['pixel_l1']+lpips_weight*pixels['lpips']), spec=spec)
        rows[pair['sample_id']] = rec
    numeric = [k for k,v in next(iter(rows.values())).items() if isinstance(v,(int,float))]
    return dict(samples=rows, mean={k:sum(v[k] for v in rows.values())/len(rows) for k in numeric})


def main():
    root = Path(os.environ.get('EXP_ROOT', Path(__file__).resolve().parents[1]))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, default=root/'upscaler_ultraflash/pairs')
    p.add_argument('--teacher-cache', type=Path, default=root/'upscaler_hybrid/teacher_cache')
    p.add_argument('--initial', type=Path, default=root/'upscaler_cmn/ewarp_steps10000/best_step_009500.pt')
    p.add_argument('--base-model', type=Path, default=os.environ.get('H3_BASE_MODEL'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='npu:0')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--min-lr', type=float, default=3e-6)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--validate-every', type=int, default=200)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--kd-weight', type=float, default=.25)
    p.add_argument('--warp-weight', type=float, default=.1)
    p.add_argument('--l1-weight', type=float, default=1.)
    p.add_argument('--lpips-weight', type=float, default=.2)
    p.add_argument('--full-every', type=int, default=4)
    p.add_argument('--seed', type=int, default=20261105)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    if args.base_model is None:
        p.error('--base-model or H3_BASE_MODEL is required')
    if min(args.steps,args.validate_every,args.save_every,args.full_every)<1:
        raise ValueError('Positive counts required')
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    ready = json.loads((args.pairs/'READY.json').read_text())
    if ready['status']!='ready' or ready['plan_sha256']!=sha256(args.pairs/'plan.json'):
        raise ValueError('Unfinalized data plan')
    for split in ('train','dev','test','legacy'):
        if sha256(args.pairs/f'{split}.jsonl')!=ready['manifests'][split]['sha256']:
            raise ValueError('Manifest changed')
    train,dev = load_pairs(args.pairs/'train.jsonl'),load_pairs(args.pairs/'dev.jsonl')
    if args.preflight:
        train,dev = train[:2],dev[:1]
    if len({x['source_sha256'] for x in train+dev}) != len(train+dev):
        raise ValueError('Source train/dev collision')
    teacher_ready = json.loads((args.teacher_cache/'READY.json').read_text())
    if teacher_ready['status']!='ready' or teacher_ready['contract']['plan_sha256']!=ready['plan_sha256']:
        raise ValueError('Teacher cache not finalized for current plan')
    attach_teacher(train+dev,args.teacher_cache,teacher_ready)
    initial = tensor_load(args.initial)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='resume'}
    config.update(recipe='CMN + frozen original H3 teacher + HR decoder-aware pixel supervision',
        initialization='Warm start from CMN best9500; fresh AdamW and cosine schedule',prior_updates=int(initial['step']),
        model_config=initial['model_config'], dtype='fp32',batch_size=1,weight_decay=.01,grad_clip=1.,
        selection='Minimum mean dev RGB L1 + 0.2 LPIPS on fixed full-spatial native temporal regions, including step0',
        pixel_target='Frozen H3 decoder of HR reference latent; not teacher pixels',
        pixel_sampling='One full-spatial native block every4 updates, otherwise one native spatial tile interior; 4 LPIPS frames per region',
        temporal_pixel_scope='Non-crossfaded real RGB positions only; no synthetic tail RGB',
        padding='MSE/KD first36, eWarp35; token36 retained as decoder context and can receive pixel gradients',
        causal_scope='Causal latent upscaler only; frozen original decoder remains noncausal',
        test_reuse='Retained16 test +3 legacy, previously inspected; not a new blind test',
        loss_definition='1 HR latent MSE + .25 teacher latent MSE + .1 eWarp + 1 RGB L1 + .2 LPIPS alex v0.1')
    config['selection'] = f"Minimum mean dev RGB L1 + {args.lpips_weight} LPIPS on fixed full-spatial native temporal regions, including step0"
    contract = dict(config=config,plan_sha256=ready['plan_sha256'],initial_sha256=sha256(args.initial),
        teacher_cache_ready_sha256=sha256(args.teacher_cache/'READY.json'),
        train_manifest_sha256=sha256(args.pairs/'train.jsonl'),dev_manifest_sha256=sha256(args.pairs/'dev.jsonl'),
        source_sha256={name:sha256(Path(__file__).with_name(name)) for name in
            ('train_h3_hybrid.py','h3_hybrid_pixel.py','h3_cmn.py','ultraflash_ewarp.py','data_utils.py')})
    atomic_json(args.output/'config.json',config)
    device = select_device(args.device)
    model = H3LatentUpsampler(**initial['model_config']).to(device)
    model.load_state_dict(initial['model_state_dict'],strict=True)
    vae = load_video_vae(args.base_model,device)
    import lpips
    lpips_fn = lpips.LPIPS(net='alex',version='0.1').to(device).eval().requires_grad_(False)
    supervisor = PixelSupervisor(vae,lpips_fn,device,checkpoint_decode=True)
    from evaluate_h3_cmn import vae_fingerprint
    contract['decoder_fingerprint'] = dict(weights=vae_fingerprint(args.base_model),
        implementation_sha256=sha256(Path(inspect.getfile(type(vae)))),
        recipe='Frozen FP32 native H3; spatial tile256/overlap64; native temporal7/stride5')
    contract['lpips_fingerprint'] = dict(network='alex',version='0.1',
        backbone_sha256=sha256(Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth'),
        linear_sha256=sha256(Path(lpips.__file__).resolve().parent/'weights/v0.1/alex.pth'),
        implementation_sha256=sha256(Path(inspect.getfile(type(lpips_fn)))))
    atomic_json(args.output/'contract.json',contract)
    frozen_before = frozen_versions(vae,lpips_fn)
    assert all(not q.requires_grad for frozen in (vae,lpips_fn) for q in frozen.parameters())
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01,foreach=False)
    order_rng,window_rng = random.Random(args.seed),random.Random(args.seed+1)
    schedule = []
    while len(schedule)<args.steps:
        order=list(range(len(train)));order_rng.shuffle(order);schedule.extend(order)
    summary = dict(status='running',config=config,contract=contract,train_count=len(train),dev_count=len(dev),
        parameter_count=sum(v.numel() for v in model.parameters()),validation=[],completed_updates=0,test_opened=False)
    start_step,best_step,best_score = 0,0,float('inf')
    def save(path,step,optimizer_state=False):
        value = dict(model_state_dict=to_cpu(model.state_dict()),model_config=initial['model_config'],
            step=step,contract=contract,best_step=best_step,best_score=best_score)
        if optimizer_state:
            value.update(optimizer=to_cpu(optimizer.state_dict()),summary=summary,window_rng=window_rng.getstate(),
                torch_rng=torch.get_rng_state(),best_sha256=sha256(args.output/f'best_step_{best_step:06d}.pt'))
        atomic_torch_save(path,value)
    if args.resume:
        if args.resume.resolve().parent != args.output.resolve():
            raise ValueError('Resume requires original output directory')
        saved=tensor_load(args.resume)
        if saved['contract']!=contract:
            raise ValueError('Resume contract changed')
        model.load_state_dict(saved['model_state_dict'],strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        summary=saved['summary'];start_step=saved['step'];best_step=saved['best_step'];best_score=saved['best_score']
        window_rng.setstate(saved['window_rng']);torch.set_rng_state(saved['torch_rng'])
        if sha256(args.output/f'best_step_{best_step:06d}.pt')!=saved['best_sha256']:
            raise ValueError('Selected checkpoint changed')
        log_path=args.output/'steps.jsonl'
        lines=log_path.read_text().splitlines()
        kept=[json.loads(line) for line in lines if line.strip() and json.loads(line)['step']<=start_step]
        if [rec['step'] for rec in kept] != list(range(1,start_step+1)):
            raise ValueError('Checkpoint training log has missing or duplicate steps')
        if len(kept)!=len(lines):
            log_path.rename(args.output/f'steps.before_resume_{start_step}_{time.time_ns()}.jsonl')
            log_path.write_text(''.join(json.dumps(rec)+'\n' for rec in kept))
    else:
        # Independent full-decode equivalence + pixel-only gradient check on a real train sample.
        d,r,_,_=tensors(train[0],device)
        pred=model(d).detach()
        audit=supervisor.audit(pred,r)
        atomic_json(args.output/'pixel_preflight.json',audit)
        if not audit.get('passed',False):
            raise AssertionError('Pixel contract audit failed')
        del d,r,pred
        val=validate(model,dev,supervisor,device,args.warp_weight,args.lpips_weight)
        best_score=val['mean']['selection_score']
        summary['validation'].append(dict(step=0,**val));summary['initial_dev']=val
        save(args.output/'best_step_000000.pt',0)
        print(json.dumps(dict(initial_validation=val['mean'],pixel_audit=audit)),flush=True)
    previous_seconds=float(summary.get('elapsed_seconds',0.)) if args.resume else 0.
    synchronize(device);started=time.perf_counter()
    with (args.output/'steps.jsonl').open('a' if args.resume else 'w') as log:
        for step in range(start_step,args.steps):
            tick=time.perf_counter();model.train()
            warm=min(1.,(step+1)/max(1,args.warmup))
            ratio=max(0,step-args.warmup)/max(1,args.steps-args.warmup-1)
            lr=warm*(args.min_lr+(args.lr-args.min_lr)*(1+math.cos(math.pi*ratio))/2)
            optimizer.param_groups[0]['lr']=lr
            pair=train[schedule[step]]
            draft,reference,flow,valid=tensors(pair,device)
            teacher=pair['teacher'][None].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred=model(draft)
            values=objective(pred,reference,flow,valid,args.warp_weight)
            kd=(pred[:,:,:36]-teacher[:,:,:36]).square().mean()
            spec=supervisor.sample_window(window_rng,full=(step%args.full_every==0))
            pixels=supervisor.losses(pred,reference,spec)
            pixel_loss=args.l1_weight*pixels['pixel_l1']+args.lpips_weight*pixels['lpips']
            loss=values['loss']+args.kd_weight*kd+pixel_loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite hybrid loss')
            if step==start_step:
                pixel_grad=torch.autograd.grad(pixel_loss,model.output.weight,retain_graph=True)[0]
                summary['pixel_head_grad_l2']=float(pixel_grad.norm())
                if not torch.isfinite(pixel_grad).all() or not summary['pixel_head_grad_l2']>0:
                    raise AssertionError('Pixel loss disconnected from student head')
                del pixel_grad
            loss.backward()
            if any(q.grad is None for q in model.parameters()):
                raise RuntimeError('Disconnected student parameter')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False,error_if_nonfinite=True)
            if not float(norm)>0:
                raise RuntimeError('Zero student gradient')
            optimizer.step();synchronize(device)
            rec=dict(step=step+1,sample_id=pair['sample_id'],lr=lr,loss=float(loss.detach()),teacher_mse=float(kd.detach()),
                **{k:float(v.detach()) for k,v in values.items() if k!='loss'},
                pixel_l1=float(pixels['pixel_l1'].detach()),lpips=float(pixels['lpips'].detach()),
                grad_norm_before_clip=float(norm),spec=spec,seconds=time.perf_counter()-tick)
            log.write(json.dumps(rec)+'\n');log.flush()
            if step<5 or (step+1)%25==0:
                print(json.dumps(rec),flush=True)
            del draft,reference,flow,valid,teacher,pred,values,kd,pixels,pixel_loss,loss,norm
            summary.update(completed_updates=step+1,elapsed_seconds=previous_seconds+time.perf_counter()-started)
            if (step+1)%args.validate_every==0 or step+1==args.steps:
                val=validate(model,dev,supervisor,device,args.warp_weight,args.lpips_weight)
                summary['validation'].append(dict(step=step+1,**val))
                if val['mean']['selection_score']<best_score:
                    best_score=val['mean']['selection_score'];best_step=step+1
                    save(args.output/f'best_step_{best_step:06d}.pt',best_step)
                summary.update(best_step=best_step,best_validation_score=best_score)
                print(json.dumps(dict(validation=step+1,mean=val['mean'],best_step=best_step)),flush=True)
            if (step+1)%args.save_every==0 or step+1==args.steps:
                save(args.output/'latest.pt',step+1,True)
                atomic_json(args.output/'summary.json',summary)
    save(args.output/'final.pt',args.steps)
    if frozen_before!=frozen_versions(vae,lpips_fn) or any(q.grad is not None for frozen in (vae,lpips_fn) for q in frozen.parameters()):
        raise AssertionError('Frozen decoder/LPIPS changed or accumulated gradients')
    selected=args.output/f'best_step_{best_step:06d}.pt'
    model.load_state_dict(tensor_load(selected)['model_state_dict'],strict=True);model.eval()
    summary.update(training_seconds=previous_seconds+time.perf_counter()-started,best_step=best_step,best_validation_score=best_score,
        best_checkpoint_sha256=sha256(selected),frozen_modules_unchanged=True,memory=memory_stats(device))
    summary['causality']=causal_checks(model,dev[0]['draft'][None].to(device))
    if not summary['causality']['passed']:
        raise AssertionError('Causal checks failed')
    link=args.output/'best.pt'
    if link.is_symlink():link.unlink()
    elif link.exists():raise FileExistsError(link)
    link.symlink_to(selected.name)
    if not args.preflight:
        test,legacy=load_pairs(args.pairs/'test.jsonl'),load_pairs(args.pairs/'legacy.jsonl')
        all_pairs=train+dev+test+legacy
        if len({x['source_sha256'] for x in all_pairs})!=len(all_pairs):
            raise ValueError('Split collision')
        summary['test_opened']=True
        for split,group in [('test',test),('legacy',legacy)]:
            summary[split]=evaluate(model,group,device,args.warp_weight,args.output/'predictions'/split)
            summary[f'bilinear_{split}']=evaluate(bilinear,group,device,args.warp_weight)
    summary['status']='preflight_completed' if args.preflight else 'completed'
    atomic_json(args.output/'summary.json',summary)
    print(json.dumps(dict(status=summary['status'],best_step=best_step,best_score=best_score,
        train_seconds=summary['training_seconds'],test=summary.get('test',{}).get('mean'))),flush=True)


if __name__=='__main__':
    main()
