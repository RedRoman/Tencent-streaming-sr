"""Supervised C5/W6 task LoRA initialized from released causal control + four-step weights.

This is task adaptation with the upstream teacher-forced FM objective, not new DMD.
Native-scale patches plus periodic full frames reduce per-update work; quality remains evaluation work.
"""
import argparse
import json
import os
import signal
import time
import traceback
from pathlib import Path
import torch
import torch_npu
from safetensors.torch import load_file
from solarwm.backends.minimax_h3.model import SolarMiniMaxH3Transformer3DModel
from solarwm.backends.minimax_h3.optional import H3_FP32_PARAMETER_NAMES
from solarwm.backends.minimax_h3.flow import make_shifted_schedule
from ascend_port import install_attention_port
from common import ROOT,VENDOR,schedule,sha,cpu_tree,write_json
from patch_core import PatchCore,batch_for
from runtime import TrainingRuntime
from gates import check_plan,check_backward


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--config-sha',required=True)
    ap.add_argument('--run',required=True)
    ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--resume',help='Explicit checkpoint_XXXX.pt from a prior owned run')
    args=ap.parse_args()
    assert sha(args.config)==args.config_sha
    config=json.loads(Path(args.config).read_text())
    out=ROOT/'runs'/args.run
    out.mkdir(parents=True,exist_ok=False)
    started=time.time();updates=0;stop_requested=False
    def log(event,**kw):
        row={'event':event,'time':time.time(),'elapsed_seconds':time.time()-started,
             'optimizer_updates':updates,**kw}
        print(json.dumps(row),flush=True)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    def status(state,**kw):
        write_json(out/'status.json',{'status':state,'optimizer_updates':updates,
            'target':config['target_updates'],'updated_unix':time.time(),**kw})
    def stop(sig,frame):
        nonlocal stop_requested
        stop_requested=True
        log('graceful_stop_requested',signal=sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for name,digest in config['sources'].items():assert sha(ROOT/name)==digest,name
        assert config['target_updates']==2500 and config['save_every']==100
        assert config['rank']==16 and config['lr']==1e-5
        assert config['patch_grid']==[32,64] and config['patch_resident_blocks']==24
        assert config['full_frame_every']==100
        torch.set_num_threads(8);torch.manual_seed(config['seed'])
        for name in ('public_assets.json','base_conversion.json','causal_model.json','prompt.json','npu_attention_check.json'):
            assert sha(ROOT/'evidence'/name)==config['evidence'][name],name
        identity=json.loads((ROOT/'evidence/causal_model.json').read_text())
        model_path=Path(identity['model'])
        assert sha(model_path/'diffusion_pytorch_model.safetensors.index.json')==identity['index_sha256']
        cond_path=ROOT/'data/restoration_prompt.pt'
        assert sha(cond_path)==json.loads((ROOT/'evidence/prompt.json').read_text())['sha256']
        condition=torch.load(cond_path,map_location='cpu',weights_only=True)
        samples=[]
        if args.preflight_only:
            p=ROOT/'data/Charge_train_3856_4081_first7.pt'
            assert sha(p)=='3fcca24bf47a63cb1340b9ce4ed02262b77f8fe45eedfe37938018134aadd401'
            samples=[torch.load(p,map_location='cpu',weights_only=True)]
        else:
            for row in config['samples']:
                p=ROOT/'data'/row['filename']
                assert sha(p)==row['sha256'],p
                sample=torch.load(p,map_location='cpu',weights_only=True)
                assert sample['sample_id']==row['sample_id']
                for key in ('reference','upscaled'):
                    assert sample[key].shape==(24,67,68,120)
                    assert sample[key].dtype==torch.float32 and torch.isfinite(sample[key]).all()
                samples.append(sample)
        silence=load_file(str(VENDOR/'assets/h3_silence_153_158_170.safetensors'))['silence_158f']
        status('loading_pretrained_model')
        log('strict_load_start',initialization='Ref2VA + SolarRenderer causal control + causal 4step')
        model=SolarMiniMaxH3Transformer3DModel.strict_from_pretrained(str(model_path),
            torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True).eval().requires_grad_(False)
        for name,p in model.named_parameters():
            assert p.dtype==(torch.float32 if name in H3_FP32_PARAMETER_NAMES else torch.bfloat16)
        torch.npu.set_device('npu:0')
        torch.npu.conv.allow_hf32=False;torch.npu.matmul.allow_hf32=False
        install_attention_port()
        rt=TrainingRuntime(model,'npu:0',rank=config['rank'],resident=config['patch_resident_blocks'],log=log)
        core=PatchCore(model,silence,torch.device('npu:0'),config['causal'])
        gates=check_backward(rt)
        core.crop_origin=(0,0)
        gates['patch_plan']=check_plan(core,batch_for(samples[0],condition,0,7,[0,0,*config['patch_grid']]))
        gates['full_plan']=check_plan(core,batch_for(samples[0],condition,0,7 if args.preflight_only else 12))
        write_json(out/'gates.json',gates);log('preflight_passed',**gates)
        if args.preflight_only:
            status('preflight_complete');return
        optimizer=torch.optim.AdamW(rt.params,lr=config['lr'],betas=(.9,.99),eps=1e-8,
                                    weight_decay=0,foreach=False)
        names=[n for n,p in rt.ordered]
        if args.resume:
            resume=torch.load(args.resume,map_location='cpu',weights_only=True)
            receipt=json.loads(Path(args.resume).with_suffix('.json').read_text())
            assert sha(args.resume)==receipt['sha256']
            assert resume['parameter_names']==names
            if resume['config_sha256']!=args.config_sha:
                assert resume['config_sha256']==config['resume']['source_config_sha256']
                assert resume['step']==config['resume']['step']
                assert sha(args.resume)==config['resume']['sha256']
                assert resume['next_schedule']==config['resume']['previous_next_schedule']
            rt.load_adapter(resume['adapter']);optimizer.load_state_dict(resume['optimizer'])
            updates=resume['step']
            if resume['config_sha256']==args.config_sha:
                assert resume['next_schedule']==schedule(updates,config)
            assert all(float(optimizer.state[p]['step'])==updates for p in rt.params)
            assert all(torch.equal(p.detach().cpu(),resume['adapter'][n]) for n,p in rt.ordered)
            for i,(n,p) in enumerate(rt.ordered):
                for key in ('step','exp_avg','exp_avg_sq'):
                    assert torch.equal(optimizer.state[p][key].detach().cpu(),resume['optimizer']['state'][i][key])
            torch.set_rng_state(resume['torch_rng']);torch.npu.set_rng_state(resume['npu_rng'])
            del resume
            log('restored_complete_checkpoint',step=updates,source=args.resume,recipe_changed=True)
        else:
            raise ValueError('v2 must continue a verified checkpoint')
        resumed_at=updates
        def checkpoint():
            p=out/f'checkpoint_{updates:04d}.pt'
            if p.exists():return
            payload={'format':'solar-task-lora-v2','step':updates,'adapter':rt.adapter_state(),
                'parameter_names':names,'optimizer':cpu_tree(optimizer.state_dict()),
                'torch_rng':torch.get_rng_state().clone(),'npu_rng':torch.npu.get_rng_state().cpu().clone(),
                'next_schedule':schedule(updates,config),'config_sha256':args.config_sha,
                'pretrained_model_index_sha256':identity['index_sha256']}
            tmp=p.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(p)
            check=torch.load(p,map_location='cpu',weights_only=True)
            assert check['step']==updates and check['parameter_names']==names
            assert all(torch.equal(check['adapter'][n],payload['adapter'][n]) for n in names)
            assert check['next_schedule']==schedule(updates,config)
            if updates:
                assert len(check['optimizer']['state'])==len(rt.params)
                assert all(float(s['step'])==updates for s in check['optimizer']['state'].values())
            for k in ('torch_rng','npu_rng'):assert torch.equal(check[k],payload[k])
            write_json(p.with_suffix('.json'),{'step':updates,'sha256':sha(p),'bytes':p.stat().st_size,
                'config_sha256':args.config_sha,'resume_state_readback_verified':True,
                'inference_quality_accepted':False})
            log('checkpoint_saved',step=updates,path=str(p))
            if updates==resumed_at+1:
                # Exercise the actual reload path once, preserving the just-completed update.
                rt.load_adapter(check['adapter']);optimizer.load_state_dict(check['optimizer'])
                for i,(n,param) in enumerate(rt.ordered):
                    assert torch.equal(param.detach().cpu(),check['adapter'][n])
                    for k in ('step','exp_avg','exp_avg_sq'):
                        assert torch.equal(optimizer.state[param][k].detach().cpu(),check['optimizer']['state'][i][k])
                torch.set_rng_state(check['torch_rng']);torch.npu.set_rng_state(check['npu_rng'])
                log('live_adapter_optimizer_reload_verified',step=updates)
        checkpoint()
        while updates<config['target_updates']:
            if stop_requested:
                checkpoint();status('stopped_with_checkpoint');return
            spec=schedule(updates,config);sample=samples[spec['sample_index']]
            step_started=time.time()
            desired=0 if spec['full_frame'] else config['patch_resident_blocks']
            if rt.resident!=desired:rt.set_resident(desired)
            core.crop_origin=(0,0) if spec['crop'] is None else tuple(spec['crop'][:2])
            batch=batch_for(sample,condition,spec['latent_start'],spec['latent_count'],spec['crop'])
            lr=config['lr']*min((updates+1)/50,1.0)
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            torch.npu.reset_peak_memory_stats();t=time.time()
            status('training',next_schedule=spec)
            log('step_start',next_step=updates+1,**spec)
            loss=core.forward_loss(batch,noise_seed=spec['noise_seed'])
            assert torch.isfinite(loss),float(loss.detach())
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in rt.params)
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            norm=float(torch.nn.utils.clip_grad_norm_(rt.params,1.0))
            assert norm>0 and torch.isfinite(torch.tensor(norm)),norm
            optimizer.step();updates+=1
            assert all(float(optimizer.state[p]['step'])==updates for p in rt.params)
            torch.npu.synchronize()
            metrics={'loss':float(loss.detach()),'grad_norm':norm,'lr':lr,'seconds':time.time()-t,
                'peak_allocated_bytes':torch.npu.max_memory_allocated(),'sample':sample['sample_id'],
                'wall_seconds_including_placement':time.time()-step_started,**spec}
            log('optimizer_update',step=updates,**metrics);status('training',last_update=metrics)
            del loss,batch
            optimizer.zero_grad(set_to_none=True)
            if updates==resumed_at+1 or updates%config['save_every']==0 or stop_requested:checkpoint()
            if updates in config['evaluate_steps']:
                status('checkpoint_latent_evaluation',checkpoint=updates)
                core.crop_origin=(0,0)
                with torch.no_grad():
                    evaluated,_,_,_=core.stream(batch_for(samples[0],condition,0,7,inference=True),
                        generator=torch.Generator(device='npu:0').manual_seed(20261109),
                        schedule=make_shifted_schedule(5,shift=12.0),exit_index=3,window=6,
                        record=None,cache_structure=False)
                    evaluated=evaluated[0].float().cpu()
                    assert torch.isfinite(evaluated).all()
                    gt=samples[0]['reference'][:,:7]
                    result={'step':updates,'sample':samples[0]['sample_id'],'latent_slice':[0,7],
                        'refined_latent_l1':float((evaluated-gt).abs().mean()),
                        'draft_latent_l1':float((samples[0]['upscaled'][:,:7]-gt).abs().mean()),
                        'scope':'train sample, first 7 latents, four-step rollout; not dev or RGB quality'}
                    torch.save({'refined':evaluated,**result},out/f'eval_{updates:04d}.pt')
                    write_json(out/f'eval_{updates:04d}.json',result);log('latent_evaluation',**result)
                del evaluated
        checkpoint();status('training_complete_evaluation_pending');log('training_complete',step=updates)
    except BaseException as error:
        write_json(out/'failure.json',{'type':type(error).__name__,'error':str(error),
            'optimizer_updates':updates,'traceback':traceback.format_exc()})
        status('failed');raise

if __name__=='__main__':main()
