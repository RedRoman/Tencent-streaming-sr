"""Matched v5 continuation experiment, with fixed architecture and frozen decoder.

Control retains the v5 objective; motion aligns true reference changes instead
of penalizing all temporal changes. Test never selects checkpoints. Original
runtime files are imported, SHA-bound, and left unchanged.
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
from h3_conv3d_hybrid import load_student_checkpoint, STATE_DICT_FORMAT
from h3_v5_motion import MotionPixelSupervisor, target_residual_ewarp
from h3_hybrid_pixel import PixelSupervisor
from evaluate_h3_cmn import module_source


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


def execution_precision(device):
    """Read effective math flags without changing any accelerator setting.

    FP32 parameter/input storage does not imply full-mantissa FP32 internal
    convolution arithmetic when the NPU HF32 path is allowed.
    """
    result = dict(device_type=device.type, tensor_dtype='float32',
                  tensor_dtype_scope='Parameter/input storage; not a claim about internal kernel arithmetic',
                  torch_version=str(torch.__version__), torch_npu_version=None,
                  conv_allow_hf32=None, matmul_allow_hf32=None,
                  float32_matmul_precision=torch.get_float32_matmul_precision())
    if device.type == 'npu':
        import torch_npu
        result.update(torch_npu_version=str(torch_npu.__version__),
                      conv_allow_hf32=bool(torch.npu.conv.allow_hf32),
                      matmul_allow_hf32=bool(torch.npu.matmul.allow_hf32))
    return result


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
        rec.update(teacher_mse=float(kd), global_score=float(pixels['pixel_l1']+lpips_weight*pixels['lpips']),
                   selection_score=float(pixels['pixel_l1']+lpips_weight*pixels['lpips']+.5*pixels['motion_l1']+.2*pixels['delta_l1']), spec=spec)
        rows[pair['sample_id']] = rec
    numeric = [k for k,v in next(iter(rows.values())).items() if isinstance(v,(int,float))]
    return dict(samples=rows, mean={k:sum(v[k] for v in rows.values())/len(rows) for k in numeric})


def main():
    root = Path(os.environ.get('EXP_ROOT', Path(__file__).resolve().parents[1]))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, default=root/'upscaler_ultraflash/pairs')
    p.add_argument('--teacher-cache', type=Path, default=root/'upscaler_hybrid/teacher_cache')
    p.add_argument('--initial', type=Path, default=root/'upscaler_conv3d_hybrid/kd_pixel_steps2000/best_step_001800.pt')
    p.add_argument('--arm', choices=('control','motion'), required=True)
    p.add_argument('--initial-sha256', default='c7aaaa396d877a84c3fed2a5536189b7ed5e61226a82ef1a9e6be1e831f7f914')
    p.add_argument('--upstream', type=Path, default=root/'upstream')
    p.add_argument('--upscaler', type=Path, default=root/'assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors')
    p.add_argument('--base-model', type=Path, default=os.environ.get('H3_BASE_MODEL'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='npu:0')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--min-lr', type=float, default=1e-6)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--validate-every', type=int, default=200)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--kd-weight', type=float, default=.25)
    p.add_argument('--warp-weight', type=float, default=.1)
    p.add_argument('--l1-weight', type=float, default=1.)
    p.add_argument('--lpips-weight', type=float, default=.2)
    p.add_argument('--full-every', type=int, default=4)
    p.add_argument('--seed', type=int, default=20261106)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    if args.base_model is None:
        p.error("Set H3_BASE_MODEL or pass --base-model")
    if args.preflight and args.steps != 8:
        raise ValueError('The preflight contract requires exactly eight updates')
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
    if len(train) != 128 or len(dev) != 16:
        raise ValueError('Expected the fixed 128 train / 16 dev dataset')
    if len({x['sample_id'] for x in train+dev}) != len(train+dev):
        raise ValueError('Sample train/dev collision')
    if len({x['source_sha256'] for x in train+dev}) != len(train+dev):
        raise ValueError('Source train/dev collision')
    if args.preflight:
        train,dev = train[:2],dev[:1]
    teacher_ready = json.loads((args.teacher_cache/'READY.json').read_text())
    if teacher_ready['status']!='ready' or teacher_ready['contract']['plan_sha256']!=ready['plan_sha256']:
        raise ValueError('Teacher cache not finalized for current plan')
    attach_teacher(train+dev,args.teacher_cache,teacher_ready)
    if sha256(args.initial) != args.initial_sha256:
        raise ValueError('Initial must be the fixed named v5 checkpoint')
    initial = tensor_load(args.initial)
    initial_contract = initial['contract']
    prior_step = int(initial['step'])
    prior_base_updates = int(initial['contract']['config']['prior_updates'])
    if prior_step != 1800 or initial['contract']['config']['steps'] != 2000:
        raise ValueError('Unexpected v5 lineage')
    del initial
    device = select_device(args.device)
    precision_start = execution_precision(device)
    model = load_student_checkpoint(args.initial,args.upstream,args.upscaler,device,activation_checkpointing=True)
    model_config = model.model_config
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='resume'}
    config.update(recipe='v5 continuation matched-budget control vs target-change-preserving supervision',
        initialization='Fixed named v5 step1800 weights verbatim; fresh AdamW; no parameter/architecture/causal adapter change',
        prior_updates=prior_step+prior_base_updates,prior_selected_run_step=prior_step,prior_base_updates=prior_base_updates,
        model_config=model_config,activation_checkpointing=True,dropout_mode='disabled: original model.eval() with parameter gradients enabled', dtype='fp32',batch_size=1,weight_decay=.01,grad_clip=1.,
        dtype_scope='FP32 parameter/input tensors; NPU internal convolution/matmul precision follows the recorded HF32 flags',
        selection='Minimum mean dev RGB L1 + 0.2 LPIPS on fixed full-spatial native temporal regions, including step0',
        pixel_target='Frozen H3 decoder of HR reference latent; not teacher pixels',
        pixel_sampling='One full-spatial native block every4 updates, otherwise one native spatial tile interior; 4 LPIPS frames per region',
        temporal_pixel_scope='Non-crossfaded real RGB positions only; no synthetic tail RGB',
        padding='MSE/KD first36, eWarp35; token36 retained as decoder context and can receive pixel gradients',
        causal_scope='Causal latent upscaler only; frozen original decoder remains noncausal',
        test_reuse='Retained16 test +3 legacy, previously inspected; not a new blind test',
        loss_definition=('1 HR latent MSE + .25 teacher MSE + .1 raw eWarp + 1 RGB L1 + .2 LPIPS' if args.arm=='control' else '1 HR latent MSE + .25 teacher MSE + .1 target-residual eWarp + 1 RGB L1 + .2 LPIPS + .5 high-change RGB L1 + .2 adjacent RGB difference-residual L1'),
        motion_threshold=.05, motion_weight=.5, delta_weight=.2,
        motion_scope='Training/selection masks from detached reference RGB; model inputs and causal forward unchanged; no occlusion-confidence claim',
        matched_control='Same v5 initial SHA, seed, sample/region/LPIPS schedule, optimizer and 2000 updates; objective differs',
        prior_successful_run_updates=2000,prior_selected_lineage_updates=prior_base_updates+prior_step,
        improvement_claim='Candidate experiment only, not a promoted version or confirmed improvement')
    config['selection'] = 'Minimum 16dev mean L1 + .2 LPIPS + .5 high-change L1 + .2 delta L1 including step0, eligible only when fixed-region L1+.2LPIPS <=1.01*step0 and each L1/LPIPS <=1.02*step0. Same rule for both arms. This is a sparse-region dev guard, not a full120 guarantee; independent full120 dev gate precedes test.'
    contract = dict(config=config,plan_sha256=ready['plan_sha256'],initial_sha256=sha256(args.initial),
        execution_precision=precision_start,
        teacher_cache_ready_sha256=sha256(args.teacher_cache/'READY.json'),
        train_manifest_sha256=sha256(args.pairs/'train.jsonl'),dev_manifest_sha256=sha256(args.pairs/'dev.jsonl'),
        source_sha256={name:sha256(module_source(name)) for name in
            ('train_h3_v5_motion.py','h3_v5_motion.py','h3_conv3d_hybrid.py','causal_upscaler.py',
             'h3_hybrid_pixel.py','train_h3_cmn.py','h3_cmn.py','train_upscaler_ultraflash.py',
             'ultraflash_ewarp.py','data_utils.py','prepare_data.py','train_upscaler_trial.py',
             'evaluate_h3_cmn.py','evaluate_causal_upscaler_pixels.py','decode_results.py')})
    contract['original_upscaler_fingerprint'] = dict(
        base_weights_sha256=sha256(args.upscaler),
        loader_sha256=sha256(args.upstream/'open_h3/upscaler.py'),
        loaded_node_sha256=sha256(Path(model.adapter.node.__file__)),
        model_class=type(model.model).__name__,parameter_count=sum(v.numel() for v in model.parameters()),
        state_dict_format=STATE_DICT_FORMAT,adapter_structure=model.adapter.structure)
    vae = load_video_vae(args.base_model,device)
    import lpips
    lpips_fn = lpips.LPIPS(net='alex',version='0.1').to(device).eval().requires_grad_(False)
    supervisor = MotionPixelSupervisor(vae,lpips_fn,device,checkpoint_decode=True)
    from evaluate_h3_cmn import vae_fingerprint
    contract['decoder_fingerprint'] = dict(weights=vae_fingerprint(args.base_model),
        implementation_sha256=sha256(Path(inspect.getfile(type(vae)))),
        recipe='Frozen FP32 native H3; spatial tile256/overlap64; native temporal7/stride5')
    contract['lpips_fingerprint'] = dict(network='alex',version='0.1',
        backbone_sha256=sha256(Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth'),
        linear_sha256=sha256(Path(lpips.__file__).resolve().parent/'weights/v0.1/alex.pth'),
        implementation_sha256=sha256(Path(inspect.getfile(type(lpips_fn)))))
    contract['floating_parameter_dtypes'] = {
        name: sorted({str(value.dtype) for value in network.parameters() if value.is_floating_point()})
        for name,network in (('student',model),('decoder',vae),('lpips',lpips_fn))}
    if execution_precision(device) != precision_start:
        raise AssertionError('Model initialization changed the effective precision settings')
    # Keep the exact v5 frozen dependency/precision identity, not merely a new label.
    for key in ('decoder_fingerprint','lpips_fingerprint','original_upscaler_fingerprint',
                'execution_precision','teacher_cache_ready_sha256','plan_sha256',
                'train_manifest_sha256','dev_manifest_sha256'):
        if contract[key] != initial_contract[key]:
            raise AssertionError(f'v5 frozen dependency changed: {key}')
    shared_sources=set(initial_contract['source_sha256']) & set(contract['source_sha256'])
    for name in shared_sources:
        if contract['source_sha256'][name] != initial_contract['source_sha256'][name]:
            raise AssertionError(f'v5 inherited source changed: {name}')
    contract['v5_dependency_parity'] = dict(passed=True,shared_sources=sorted(shared_sources),
        initial_contract_sha256=__import__('hashlib').sha256(
            json.dumps(initial_contract,sort_keys=True).encode()).hexdigest())
    frozen_before = frozen_versions(vae,lpips_fn)
    assert all(not q.requires_grad for frozen in (vae,lpips_fn) for q in frozen.parameters())
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01,foreach=False)
    order_rng,window_rng = random.Random(args.seed),random.Random(args.seed+1)
    schedule = []
    while len(schedule)<args.steps:
        order=list(range(len(train)));order_rng.shuffle(order);schedule.extend(order)
    summary = dict(status='running',config=config,contract=contract,train_count=len(train),dev_count=len(dev),
        execution_precision_start=precision_start,
        parameter_count=sum(v.numel() for v in model.parameters()),validation=[],completed_updates=0,test_opened=False)
    start_step,best_step,best_score = 0,0,float('inf')
    def save(path,step,optimizer_state=False):
        value = dict(model_state_dict=to_cpu(model.original_state_dict()),model_config=model_config,
            adapter_config=model.adapter_config,state_dict_format=STATE_DICT_FORMAT,
            step=step,contract=contract,best_step=best_step,best_score=best_score)
        if optimizer_state:
            value.update(optimizer=to_cpu(optimizer.state_dict()),summary=summary,window_rng=window_rng.getstate(),
                torch_rng=torch.get_rng_state(),
                device_rng=torch.npu.get_rng_state(device) if device.type=='npu' else None,best_sha256=sha256(args.output/f'best_step_{best_step:06d}.pt'))
        atomic_torch_save(path,value)
    if args.resume:
        if args.resume.resolve().parent != args.output.resolve():
            raise ValueError('Resume requires original output directory')
        saved=tensor_load(args.resume)
        if saved['contract']!=contract:
            raise ValueError('Resume contract changed')
        model.load_original_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        summary=saved['summary'];start_step=saved['step'];best_step=saved['best_step'];best_score=saved['best_score']
        window_rng.setstate(saved['window_rng']);torch.set_rng_state(saved['torch_rng'])
        if device.type=='npu' and saved.get('device_rng') is not None:
            torch.npu.set_rng_state(saved['device_rng'],device)
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
        del saved
    else:
        # Save the exact attempted contract before auditing; failures leave
        # their original evidence. A rejected resume never overwrites it.
        atomic_json(args.output/'config.json',config)
        atomic_json(args.output/'contract.json',contract)
        # Independent full-decode equivalence + pixel-only gradient check on a real train sample.
        d,r,_,_=tensors(train[0],device)
        with torch.no_grad():
            pred=model(d)
        summary['initial_causality']=causal_checks(model,d)
        atomic_json(args.output/'initial_causality.json',summary['initial_causality'])
        if not summary['initial_causality']['passed']:
            raise AssertionError('Initial causal checks failed')
        model.reset_stream()
        audit=PixelSupervisor(vae,lpips_fn,device,checkpoint_decode=True).audit(pred,r)
        atomic_json(args.output/'pixel_preflight.json',audit)
        if not audit.get('passed',False):
            raise AssertionError('Pixel contract audit failed')
        del d,r,pred
        val=validate(model,dev,supervisor,device,args.warp_weight,args.lpips_weight)
        best_score=val['mean']['selection_score']
        summary['validation'].append(dict(step=0,**val));summary['initial_dev']=val
        summary.update(best_step=0,best_validation_score=best_score,selection_guard_baseline=val['mean'])
        save(args.output/'best_step_000000.pt',0)
        print(json.dumps(dict(initial_validation=val['mean'],pixel_audit=audit)),flush=True)
    atomic_json(args.output/'summary.json',summary)
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
            residual=target_residual_ewarp(pred,reference,flow,valid)
            pixel_loss=args.l1_weight*pixels['pixel_l1']+args.lpips_weight*pixels['lpips']
            motion_loss=.5*pixels['motion_l1']+.2*pixels['delta_l1']
            if args.arm=='motion':
                loss=values['latent_mse']+args.warp_weight*residual['residual_ewarp_l1']+args.kd_weight*kd+pixel_loss+motion_loss
            else:
                loss=values['loss']+args.kd_weight*kd+pixel_loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite hybrid loss')
            if step==start_step:
                head_name,head_parameter=model.pixel_probe_parameter()
                pixel_grad=torch.autograd.grad(pixel_loss,head_parameter,retain_graph=True)[0]
                summary['pixel_head_parameter']=head_name
                summary['pixel_head_grad_l2']=float(pixel_grad.norm())
                if not torch.isfinite(pixel_grad).all() or not summary['pixel_head_grad_l2']>0:
                    raise AssertionError('Pixel loss disconnected from student head')
                del pixel_grad
                motion_grad=torch.autograd.grad(motion_loss,head_parameter,retain_graph=True)[0]
                summary['motion_head_grad_l2']=float(motion_grad.norm())
                if not bool(torch.isfinite(motion_grad).all()) or not summary['motion_head_grad_l2']>0:
                    raise AssertionError('Nonfinite/zero motion objective gradient')
                del motion_grad
                warp_grad=torch.autograd.grad(residual['residual_ewarp_l1'],head_parameter,retain_graph=True)[0]
                summary['residual_warp_head_grad_l2']=float(warp_grad.norm())
                if not bool(torch.isfinite(warp_grad).all()) or not summary['residual_warp_head_grad_l2']>0:
                    raise AssertionError('Nonfinite/zero residual warp objective gradient')
                del warp_grad
            if float(pixels['motion_roi_fraction'])>0 and 'motion_roi_head_grad_l2' not in summary:
                _,probe_parameter=model.pixel_probe_parameter()
                roi_grad=torch.autograd.grad(pixels['motion_l1'],probe_parameter,retain_graph=True)[0]
                summary['motion_roi_head_grad_l2']=float(roi_grad.norm())
                summary['motion_roi_gradient_probe_step']=step+1
                if not bool(torch.isfinite(roi_grad).all()) or not summary['motion_roi_head_grad_l2']>0:
                    raise AssertionError('Nonfinite/zero nonempty ROI gradient')
                del roi_grad
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
                motion_l1=float(pixels['motion_l1'].detach()),delta_l1=float(pixels['delta_l1'].detach()),
                residual_ewarp_l1=float(residual['residual_ewarp_l1'].detach()),
                motion_roi_fraction=float(pixels['motion_roi_fraction']),
                adjacent_pair_count=float(pixels['adjacent_pair_count']),
                adjacent_pair_fraction=float(pixels['adjacent_pair_fraction']),
                target_ewarp_l1=float(residual['target_ewarp_l1']),
                grad_norm_before_clip=float(norm),spec=spec,seconds=time.perf_counter()-tick)
            log.write(json.dumps(rec)+'\n');log.flush()
            if step<5 or (step+1)%25==0:
                print(json.dumps(rec),flush=True)
            del draft,reference,flow,valid,teacher,pred,values,kd,pixels,pixel_loss,motion_loss,residual,loss,norm
            summary.update(completed_updates=step+1,elapsed_seconds=previous_seconds+time.perf_counter()-started)
            if (step+1)%args.validate_every==0 or step+1==args.steps:
                val=validate(model,dev,supervisor,device,args.warp_weight,args.lpips_weight)
                summary['validation'].append(dict(step=step+1,**val))
                baseline=summary['selection_guard_baseline']
                eligible=(val['mean']['global_score'] <= baseline['global_score']*1.01 and
                          val['mean']['pixel_l1'] <= baseline['pixel_l1']*1.02 and
                          val['mean']['lpips'] <= baseline['lpips']*1.02)
                summary['validation'][-1]['eligible_global_guard']=eligible
                if eligible and val['mean']['selection_score']<best_score:
                    best_score=val['mean']['selection_score'];best_step=step+1
                    save(args.output/f'best_step_{best_step:06d}.pt',best_step)
                summary.update(best_step=best_step,best_validation_score=best_score)
                print(json.dumps(dict(validation=step+1,mean=val['mean'],best_step=best_step)),flush=True)
            if (step+1)%args.save_every==0 or step+1==args.steps:
                save(args.output/'latest.pt',step+1,True)
                atomic_json(args.output/'summary.json',summary)
    if not summary.get('motion_roi_head_grad_l2',0)>0:
        raise AssertionError('No successful nonempty target-change ROI gradient probe')
    save(args.output/'final.pt',args.steps)
    if frozen_before!=frozen_versions(vae,lpips_fn) or any(q.grad is not None for frozen in (vae,lpips_fn) for q in frozen.parameters()):
        raise AssertionError('Frozen decoder/LPIPS changed or accumulated gradients')
    if any(sha256(module_source(name)) != digest
           for name,digest in contract['source_sha256'].items()):
        raise AssertionError('Imported training source changed during the run')
    if (sha256(Path(model.adapter.node.__file__)) != contract['original_upscaler_fingerprint']['loaded_node_sha256']
            or sha256(args.upstream/'open_h3/upscaler.py') != contract['original_upscaler_fingerprint']['loader_sha256']):
        raise AssertionError('Original upscaler implementation changed during the run')
    selected=args.output/f'best_step_{best_step:06d}.pt'
    model.load_original_state_dict(tensor_load(selected)['model_state_dict']);model.eval()
    summary.update(training_seconds=previous_seconds+time.perf_counter()-started,best_step=best_step,best_validation_score=best_score,
        best_checkpoint_sha256=sha256(selected),frozen_modules_unchanged=True,source_contract_unchanged=True,
        memory=memory_stats(device))
    summary['causality']=causal_checks(model,dev[0]['draft'][None].to(device))
    if not summary['causality']['passed']:
        raise AssertionError('Causal checks failed')
    model.reset_stream()
    link=args.output/'best.pt'
    if link.is_symlink():link.unlink()
    elif link.exists():raise FileExistsError(link)
    link.symlink_to(selected.name)
    # Test/legacy tensors remain unopened until the independent full120 dev gate.
    summary['test_opened']=False
    summary['execution_precision_end'] = execution_precision(device)
    summary['execution_precision_unchanged'] = summary['execution_precision_end'] == summary['execution_precision_start']
    if not summary['execution_precision_unchanged']:
        atomic_json(args.output/'summary.json',summary)
        raise AssertionError('Effective precision settings changed during the run')
    summary['status']='preflight_completed' if args.preflight else 'completed'
    atomic_json(args.output/'summary.json',summary)
    print(json.dumps(dict(status=summary['status'],best_step=best_step,best_score=best_score,
        train_seconds=summary['training_seconds'],test=summary.get('test',{}).get('mean'))),flush=True)


if __name__=='__main__':
    main()
