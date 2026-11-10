"""New user-authorized residual stage from preserved TF2500 + original AdamW.

This is a changed objective and sampling schedule, not an unchanged TF resume.
Old401 optimizer states are retained; only the newly introduced head gets new
optimizer slots. Real-model gates precede all2000 planned parameter updates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from decoder_validation_admission import predecessor_state, verify_parent_locks
from tf_resume_utils import load_cpu, validate_bundle, cpu_tree, sha


def admit(a):
    stage = Path(a.stage)
    assert sha(stage/'manifest.json') == a.manifest_sha
    manifest = json.loads((stage/'manifest.json').read_text())
    for n, h in manifest['files'].items():
        assert sha(stage/n) == h, n
    plan = json.loads((stage/'config/anchored_refiner_v1.json').read_text())
    cpu_path=stage/'evidence/anchored_cpu_v1.json'
    assert sha(cpu_path)==plan['CPU_report_sha256']
    cpu=json.loads(cpu_path.read_text())
    assert cpu['test_result']['status']=='passed' and not cpu['test_result']['accelerator_initialized']
    assert cpu['test_result']['all_model_tensors_CPU']
    for name,digest in cpu['tested_sources'].items():
        assert sha(stage/name)==digest
    assert plan['resume_step'] == 2500 and plan['new_optimizer_updates'] == 2000
    assert plan['upsampler_sha256'] == '7b9f43c17462fb2d0554645bc7889dc7abdebc052ee9f18f6780624de912e0b5'
    assert sha(a.train_manifest) == plan['train_manifest_sha256']
    assert sha(a.dev_manifest) == plan['dev_manifest_sha256']
    train = json.loads(Path(a.train_manifest).read_text())
    dev = json.loads(Path(a.dev_manifest).read_text())
    assert train['upsampler_sha256'] == dev['accepted_upsampler_sha256'] == plan['upsampler_sha256']
    assert not dev['test_unlocked']
    for record in train['samples']+dev['samples']:
        assert sha(record['path']) == record['sha256']
    assert sha(dev['condition']['path']) == train['condition_sha256'] == plan['condition_sha256']
    assert sha(a.assets_manifest) == plan['assets_manifest_sha256']
    assets = json.loads(Path(a.assets_manifest).read_text())
    for n, row in assets['files'].items():
        st = (Path(a.root)/n).stat()
        assert (st.st_size,st.st_mtime_ns,st.st_ino)==(row['bytes'],row['mtime_ns'],row['inode'])
    stop = json.loads(Path(a.training_stop_receipt).read_text())
    assert sha(a.training_stop_receipt) == plan['training_stop_receipt_sha256']
    assert stop['actual_logged_optimizer_updates'] == 2500 and stop['parent_and_child_absent']
    launch = json.loads(Path(a.diagnostic_launch).read_text())
    command = launch['command']
    assert sha(command[1]) == plan['diagnostic_worker_sha256']
    assert command[command.index('--manifest-sha')+1] == plan['diagnostic_manifest_sha256']
    assert Path(command[command.index('--output')+1]).resolve() == Path(a.diagnostic_output).resolve()
    state = predecessor_state(a.diagnostic_launch)
    completed = Path(a.diagnostic_output)/'result.json'
    ready = state['ready'] and launch.get('status') == 'complete' and completed.exists()
    diagnosed = []
    if ready:
        result = json.loads(completed.read_text())
        assert result['status'] == 'matched_history_diagnostic_complete_interpretation_pending'
        assert result['checkpoint_sha256'] == plan['adapter_sha256'] and result['new_optimizer_updates'] == 0
        expected = [r['sample_id'] for r in dev['samples']+train['samples']]
        assert [r['sample_id'] for r in result['samples']] == expected
        for rec in result['samples']:
            path = Path(a.diagnostic_output)/(rec['sample_id']+'.json')
            assert sha(path) == rec['receipt_sha256']
            row = json.loads(path.read_text())
            assert row['four_latent_full_prefix_oracle_parity'][0]['max_abs'] == 0
            assert all(v['saved_self_history_replay']['max_abs'] == 0 for v in row['rows'] if 'saved_self_history_replay' in v)
            diagnosed.append({'sample_id':rec['sample_id'],'receipt_sha256':rec['receipt_sha256']})
    report = {'status':'stdlib_integrity_admission','ready':ready,'predecessor':state,
              'completed_diagnostics':diagnosed,'manifest_sha256':a.manifest_sha,
              'torch_imported':'torch' in sys.modules,'new_optimizer_updates':0}
    return report, plan, train, dev


def main():
    parser = argparse.ArgumentParser()
    for key in ['stage','manifest-sha','root','resume-dir','train-manifest','dev-manifest',
                'assets-manifest','training-stop-receipt','diagnostic-launch','diagnostic-output','output']:
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--admission-only',action='store_true')
    parser.add_argument('--stage-resume-receipt')
    a = parser.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    started, updates = time.monotonic(), 0
    def write(name, value):
        p = out/name
        tmp = p.with_suffix(p.suffix+'.tmp')
        tmp.write_text(json.dumps(value,indent=2)+'\n')
        tmp.replace(p)
    def log(event, **values):
        row = {'event':event,'seconds':time.monotonic()-started,'stage_optimizer_updates':updates,
               'total_optimizer_updates':2500+updates,**values}
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        admission, plan, train, dev = admit(a)
        write('admission.json',admission)
        if a.admission_only:
            print(json.dumps(admission),flush=True)
            return
        assert admission['ready'], 'Existing diagnostic must complete and be reaped before accelerator import'
        write('device_lock_owner.json',verify_parent_locks(7))
        smi = subprocess.check_output(['npu-smi','info'],text=True)
        (out/'npu_smi_before_import.txt').write_text(smi)
        assert 'No running processes found in NPU 7' in smi
        memory={v.split(':')[0]:int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines()}
        assert memory['MemAvailable'] >= 100*(1<<30), 'Insufficient host memory'
        import torch
        import torch_npu
        from paged_runtime import Runtime
        from causal_attention import position_noise
        from anchored_refiner_v1 import ResidualHead, AnchoredStreamingRefiner, align_input_projection, schedule, training_input
        torch.set_num_threads(4)
        torch.manual_seed(20261110)
        torch.npu.set_device('npu:0')
        torch.npu.conv.allow_hf32 = False
        torch.npu.matmul.allow_hf32 = False
        torch.set_float32_matmul_precision('highest')
        resume = Path(a.resume_dir)
        adapter = load_cpu(resume/'adapter_step2500.pt',plan['adapter_sha256'])
        private = load_cpu(resume/'private_resume_step2500.pt',plan['optimizer_sha256'])
        bundle = validate_bundle(adapter,private,2500)
        assert private['parameter_names'] == list(adapter['adapter'])
        rt = Runtime(a.root,resident=24,canonical_rows=True,log=log)
        align_input_projection(rt)
        rt.load_adapter(adapter['adapter'])
        head = ResidualHead(device=rt.device,bound=.25)
        ordered = [(n,p) for n,p in rt.model.named_parameters() if p.requires_grad]+[('condition.weight',rt.condition.weight)]
        assert [n for n,p in ordered] == private['parameter_names']
        opt = torch.optim.AdamW(rt.params,lr=1e-4,betas=(.9,.99),eps=1e-8,weight_decay=0,foreach=False)
        opt.load_state_dict(private['optimizer'])
        for i,(n,p) in enumerate(ordered):
            assert torch.equal(p.cpu(),adapter['adapter'][n])
            for k in ('step','exp_avg','exp_avg_sq'):
                assert torch.equal(opt.state[p][k].cpu(),private['optimizer']['state'][i][k])
        opt.add_param_group({'params':list(head.parameters()),'lr':1e-3})
        ordered += [('residual_head.'+n,p) for n,p in head.named_parameters()]
        params = [p for n,p in ordered]
        samples = [torch.load(r['path'],map_location='cpu',weights_only=True) for r in train['samples']]
        dev_samples = [torch.load(r['path'],map_location='cpu',weights_only=True) for r in dev['samples']]
        assert all(tuple(s[k].shape)==(24,67,68,120) for s in samples+dev_samples for k in ['upscaled','reference'])
        condition = torch.load(dev['condition']['path'],map_location='cpu',weights_only=True)
        initial_head = cpu_tree(head.state_dict())
        s = samples[0]
        draft = s['upscaled'][:,:2]
        runner = AnchoredStreamingRefiner(rt,head,condition,sample_id=s['sample_id'])
        identity = runner.push(draft[None])
        assert torch.equal(identity,draft[None])
        log('real_initial_identity_passed',latents=2,max_abs=0,optimizer_restored=bundle)
        # Nonzero diagnostic head tests causal computation and upstream backward;
        # restore the exact zero output head before any optimizer update.
        with torch.no_grad():
            head.out_proj.weight.fill_(.001)
        runner=AnchoredStreamingRefiner(rt,head,condition,sample_id=s['sample_id'])
        nonzero_full=runner.push(draft[None])
        runner=AnchoredStreamingRefiner(rt,head,condition,sample_id=s['sample_id'])
        nonzero_chunks=torch.cat([runner.push(draft[None,:,:1]),runner.push(draft[None,:,1:])],2)
        assert torch.equal(nonzero_full,nonzero_chunks)
        assert float((nonzero_full-draft[None]).abs().max()) <= .250001
        sigma = .909375
        noise = position_noise((24,1,68,120),s['sample_id'],1,20261110)
        x = draft.clone()
        x[:,-1:] = (1-sigma)*draft[:,-1:]+sigma*noise
        with torch.no_grad():
            offline = rt.offline(x,draft,condition,[0,1],sigma)
            rt.start_stream(condition)
            rt.stream_velocity(draft[:,:1],draft[:,:1],0,commit=True)
            stream = rt.stream_velocity(x[:,1:],draft[:,1:],sigma)
            parity = float((stream-offline).abs().max())
            relative = float((stream-offline).square().mean().sqrt()/offline.square().mean().sqrt())
            assert parity <= .03 and relative <= .005
            base = rt.offline(x,draft,condition,[0,1],sigma,return_full=True)[:,:1]
            changed = x.clone();changed[:,1:] *= -3
            future = rt.offline(changed,draft,condition,[0,1],sigma,return_full=True)[:,:1]
            assert torch.equal(base,future)
            changed_draft = draft.clone();changed_draft[:,1:] *= -3
            cond_future = rt.offline(x,changed_draft,condition,[0,1],sigma,return_full=True)[:,:1]
            assert torch.equal(base,cond_future)
        opt.zero_grad(set_to_none=True)
        raw = rt.offline(x,draft,condition,[0,1],sigma,train=True)
        pred = head(x[:,-1:].to(rt.device),raw,draft[:,-1:].to(rt.device),sigma)
        loss = (pred-s['reference'][:,1:2].to(rt.device)).square().mean()
        loss.backward()
        upstream_grad = sum(float(p.grad.float().square().sum()) for p in rt.params if p.grad is not None)**.5
        head_grad = sum(float(p.grad.float().square().sum()) for p in head.parameters() if p.grad is not None)**.5
        assert upstream_grad > 0 and head_grad > 0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
        opt.zero_grad(set_to_none=True)
        head.load_state_dict(initial_head)
        rt.cache = None
        # Restore original checkpoint RNG after gate-only work; noise is also
        # position-hashed, and no optimizer update has occurred during gates.
        torch.set_rng_state(private['torch_rng_state'])
        torch.npu.set_rng_state(private['npu_rng_state'])
        assert torch.equal(torch.get_rng_state(),private['torch_rng_state'])
        assert torch.equal(torch.npu.get_rng_state().cpu(),private['npu_rng_state'])
        assert all(float(opt.state[p]['step'])==2500 for p in rt.params)
        assert all(p not in opt.state or not opt.state[p] for p in head.parameters())
        gate = {'status':'passed','initial_two_latent_identity_max_abs':0,'raw_cache_parity_max_abs':parity,
                'nonzero_head_two_latent_chunk_max_abs':0,
                'raw_cache_parity_relative_rms':relative,'future_max_abs':0,'condition_future_max_abs':0,
                'nonzero_upstream_gradient_norm':upstream_grad,'nonzero_head_gradient_norm':head_grad,
                'original401AdamW_states_preserved':True,'CPU_and_NPU_RNG_restore_exact':True,
                'new_optimizer_updates':0,'scope':'real50block full68x120 two-latent gates; not full226 quality'}
        write('gates.json',gate)
        log('real_gates_passed',**gate)
        del raw,pred,loss,adapter,private,initial_head

        def save():
            ap=out/f'adapter_stage{updates:04d}.pt'
            op=out/f'private_resume_stage{updates:04d}.pt'
            weights={'format':'anchored-streaming-refiner-v1','base_step':2500,'stage_step':updates,
                     'step':2500+updates,'adapter':rt.adapter_state(),'residual_head':cpu_tree(head.state_dict()),
                     'residual_bound':.25,'upsampler_sha256':plan['upsampler_sha256'],
                     'manifest_sha256':a.manifest_sha}
            private_state={'optimizer':cpu_tree(opt.state_dict()),'base_step':2500,'stage_step':updates,
                           'step':2500+updates,'parameter_names':[n for n,p in ordered],
                           'torch_rng_state':torch.get_rng_state(),'npu_rng_state':torch.npu.get_rng_state().cpu().clone(),
                           'next_schedule':schedule(updates),'manifest_sha256':a.manifest_sha}
            for path,value in [(ap,weights),(op,private_state)]:
                assert not path.exists()
                temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)
            row={'stage_step':updates,'total_optimizer_updates':2500+updates,'adapter_sha256':sha(ap),
                 'private_optimizer_sha256':sha(op),'adapter_bytes':ap.stat().st_size,
                 'private_state_bytes':op.stat().st_size,'selected':False}
            write(f'checkpoint_stage{updates:04d}.json',row)
            return row

        initial = save()
        # Roundtrip the actual newly saved model/head/private optimizer without
        # parameter updates, and prove all old slots and both RNGs survive.
        w=torch.load(out/'adapter_stage0000.pt',map_location='cpu',weights_only=True)
        v=torch.load(out/'private_resume_stage0000.pt',map_location='cpu',weights_only=True)
        rt.load_adapter(w['adapter']);head.load_state_dict(w['residual_head']);opt.load_state_dict(v['optimizer'])
        for i,(n,p) in enumerate(ordered):
            if i < 401:
                for k in ('step','exp_avg','exp_avg_sq'):
                    assert torch.equal(opt.state[p][k].cpu(),v['optimizer']['state'][i][k])
        torch.set_rng_state(v['torch_rng_state']);torch.npu.set_rng_state(v['npu_rng_state'])
        assert torch.equal(torch.get_rng_state(),v['torch_rng_state']) and torch.equal(torch.npu.get_rng_state().cpu(),v['npu_rng_state'])
        check=AnchoredStreamingRefiner(rt,head,condition,sample_id=s['sample_id']).push(draft[None,:,:1])
        assert torch.equal(check,draft[None,:,:1])
        # Restore saved RNG after the reload inference gate, before update1.
        torch.set_rng_state(v['torch_rng_state']);torch.npu.set_rng_state(v['npu_rng_state'])
        write('initial_checkpoint_roundtrip.json',{'status':'passed','head_and_adapter_identity_max_abs':0,
              'old_optimizer_slots_preserved':True,'CPU_NPU_RNG_exact':True,'new_optimizer_updates':0,**initial})
        del w,v,check
        rt.cache=None
        if a.stage_resume_receipt:
            receipt_path=Path(a.stage_resume_receipt)
            receipt=json.loads(receipt_path.read_text())
            resumed=receipt['stage_step']
            assert 0 < resumed < 2000 and receipt['total_optimizer_updates']==2500+resumed
            wp=receipt_path.parent/f'adapter_stage{resumed:04d}.pt'
            vp=receipt_path.parent/f'private_resume_stage{resumed:04d}.pt'
            assert sha(wp)==receipt['adapter_sha256'] and sha(vp)==receipt['private_optimizer_sha256']
            w=torch.load(wp,map_location='cpu',weights_only=True)
            v=torch.load(vp,map_location='cpu',weights_only=True)
            assert w['manifest_sha256']==v['manifest_sha256']==a.manifest_sha
            assert w['format']=='anchored-streaming-refiner-v1' and w['residual_bound']==.25
            assert w['stage_step']==v['stage_step']==resumed
            assert v['parameter_names']==[n for n,p in ordered] and v['next_schedule']==schedule(resumed)
            rt.load_adapter(w['adapter']);head.load_state_dict(w['residual_head']);opt.load_state_dict(v['optimizer'])
            for i,(name,p) in enumerate(ordered):
                for key in ('step','exp_avg','exp_avg_sq'):
                    assert torch.equal(opt.state[p][key].cpu(),v['optimizer']['state'][i][key])
            assert all(float(opt.state[p]['step'])==2500+resumed for p in rt.params)
            assert all(float(opt.state[p]['step'])==resumed for p in head.parameters())
            torch.set_rng_state(v['torch_rng_state']);torch.npu.set_rng_state(v['npu_rng_state'])
            assert torch.equal(torch.get_rng_state(),v['torch_rng_state']) and torch.equal(torch.npu.get_rng_state().cpu(),v['npu_rng_state'])
            updates=resumed
            log('new_stage_checkpoint_restored',receipt_sha256=sha(receipt_path),next_schedule=schedule(updates))
            del w,v
        log('new_stage_training_start',planned_new_updates=2000,base_updates=2500)
        for step in range(updates,2000):
            spec=schedule(step)
            sample=samples[spec['sample_index']]
            x,draft,target=training_input(sample,spec)
            opt.zero_grad(set_to_none=True)
            raw=rt.offline(x,draft,condition,spec['frames'],spec['sigma'],train=True)
            pred=head(x[:,-1:].to(rt.device),raw,draft[:,-1:].to(rt.device),spec['sigma'])
            target=target.to(rt.device);anchor=draft[:,-1:].to(rt.device)
            mse=(pred-target).square().mean();preserve=(pred-anchor).square().mean()
            loss=mse+.1*preserve
            assert torch.isfinite(loss)
            loss.backward()
            grad=float(torch.nn.utils.clip_grad_norm_(params,1.))
            assert grad>0 and torch.isfinite(torch.tensor(grad))
            opt.step();updates=step+1
            assert all(float(opt.state[p]['step'])==2500+updates for p in rt.params)
            assert all(float(opt.state[p]['step'])==updates for p in head.parameters())
            assert all(torch.isfinite(p).all() for p in params)
            log('optimizer_update',sample=sample['sample_id'],schedule=spec,nativeHR_mse=float(mse.detach()),
                preserve_mse=float(preserve.detach()),loss=float(loss.detach()),gradient_norm=grad,
                max_residual=float((pred-anchor).detach().abs().max()))
            write('status.json',{'status':'anchored_residual_training','stage_optimizer_updates':updates,
                  'total_optimizer_updates':2500+updates,'planned_stage_updates':2000,'selected_refiner':None})
            if updates % 100 == 0:
                rec=save();log('checkpoint_saved',**rec)
            if updates in [500,1000,2000]:
                results=[]
                with torch.no_grad():
                    for sample in dev_samples:
                        runner=AnchoredStreamingRefiner(rt,head,condition,sample_id=sample['sample_id'])
                        predicted=runner.push(sample['upscaled'][None,:,:4])[0]
                        hr=sample['reference'][:,:4];draft4=sample['upscaled'][:,:4]
                        mse=float((predicted-hr).square().mean());base=float((draft4-hr).square().mean())
                        results.append({'sample_id':sample['sample_id'],'latent_MSE':mse,'no_refiner_MSE':base,
                                        'MSE_ratio':mse/base,'audit':runner.audit()})
                write(f'dev_first4_stage{updates:04d}.json',{'stage_step':updates,'results':results,
                      'scope':'first4 latent only; not full226 pixels/visual acceptance','optimizer_updates_during_eval':0})
                rt.cache=None
                log('short_dev_evaluation_complete',stage_step=updates,ratios=[v['MSE_ratio'] for v in results])
        write('complete.json',{'status':'anchored2000_complete_full_evaluation_pending','stage_optimizer_updates':updates,
              'total_optimizer_updates':2500+updates,'selected_refiner':None,'full226_validated':False})
        log('training_complete')
    except BaseException as error:
        write('failure.json',{'type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),
              'stage_optimizer_updates':updates,'total_optimizer_updates':2500+updates})
        raise


if __name__ == '__main__':
    main()
