"""Continue the saved200-step TF pilot to3000 total real optimizer updates.

Model, loss, AdamW state and absolute training schedule remain unchanged.
Early quality failures are retained; they do not stop this user-directed TF run.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from tf_resume_utils import load_cpu,validate_bundle,schedule,cpu_tree,sha
from decoder_validation_admission import predecessor_state,verify_parent_locks


def main():
    parser=argparse.ArgumentParser()
    for name in ['root','config','config-sha','resume-dir','data-dir','assets-manifest','predecessor-launch','output']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--admission-only',action='store_true')
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    started=time.time();updates=200

    def write(name,value):
        p=out/name;tmp=out/(name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

    def log(event,**values):
        row={'event':event,'elapsed_seconds':time.time()-started,'optimizer_updates':updates,
             'new_optimizer_updates':updates-200,**values}
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)

    try:
        assert sha(args.config)==args.config_sha
        config=json.loads(Path(args.config).read_text());stage=Path(__file__).resolve().parent.parent
        assert config['resume_step']==200 and config['target_total_updates']==3000
        assert config['resident_layers']==24 and config['lr']==1e-4 and config['draft_loss_weight']==.05
        for name,digest in config['sources'].items():assert sha(stage/name)==digest,name
        assert sha(args.assets_manifest)==config['assets_manifest_sha256']
        assets=json.loads(Path(args.assets_manifest).read_text());assert assets['status']=='complete'
        for name,desc in assets['files'].items():
            st=(Path(args.root)/name).stat()
            assert st.st_size==desc['bytes'] and st.st_mtime_ns==desc['mtime_ns'] and st.st_ino==desc['inode'],name
        data=Path(args.data_dir);assert sha(data/'manifest.json')==config['data_manifest_sha256']
        manifest=json.loads((data/'manifest.json').read_text())
        assert manifest['upsampler_sha256']==config['upsampler_sha256']
        assert manifest['condition_sha256']==sha(data/'conditioning.pt')==config['condition_sha256']
        assert len(manifest['samples'])==2
        for row in manifest['samples']:assert row['split']=='train' and sha(row['path'])==row['sha256']
        resume=Path(args.resume_dir)
        adapter=load_cpu(resume/'adapter_step0200.pt',config['resume_adapter_sha256'])
        private=load_cpu(resume/'private_resume_step0200.pt',config['resume_optimizer_sha256'])
        bundle=validate_bundle(adapter,private,200)
        assert sha(stage/'evidence/tf_resume_cpu_audit_v1.json')==config['CPU_resume_audit_sha256']
        cpu=json.loads((stage/'evidence/tf_resume_cpu_audit_v1.json').read_text())
        assert cpu['status']=='passed_CPU_resume_audit_only' and cpu['source_sha256']==config['sources']['src/tf_resume_utils.py']
        predecessor=predecessor_state(args.predecessor_launch)
        admission={'status':'CPU_resume_integrity_passed','bundle':bundle,'predecessor':predecessor,
            'next_update':201,'next_schedule':schedule(200),'new_optimizer_updates':0,
            'accelerator_forwards':0,'config_sha256':args.config_sha}
        write('admission.json',admission)
        if args.admission_only:log('CPU_admission_complete',**admission);return
        assert predecessor['ready'],'Previous owned evaluation must exit and be reaped'
        locks=verify_parent_locks(7);write('device_lock_owner.json',locks)
        smi=subprocess.check_output(['npu-smi','info'],text=True);(out/'npu_smi_before_import.txt').write_text(smi)
        assert 'No running processes found in NPU 7' in smi

        import torch
        import torch_npu
        from paged_runtime import Runtime
        from causal_attention import position_noise
        torch.set_num_threads(4);torch.manual_seed(20261109);torch.npu.set_device('npu:0')
        torch.npu.conv.allow_hf32=False;torch.npu.matmul.allow_hf32=False
        torch.set_float32_matmul_precision('highest')
        samples=[torch.load(row['path'],map_location='cpu',weights_only=True) for row in manifest['samples']]
        condition=torch.load(data/'conditioning.pt',map_location='cpu',weights_only=True)
        rt=Runtime(args.root,resident=24,canonical_rows=True,log=log)
        rt.model.eval();rt.load_adapter(adapter['adapter'])
        ordered=[(n,p) for n,p in rt.model.named_parameters() if p.requires_grad]
        ordered.append(('condition.weight',rt.condition.weight))
        assert [n for n,p in ordered]==list(adapter['adapter'])
        assert [id(p) for n,p in ordered]==[id(p) for p in rt.params]
        opt=torch.optim.AdamW(rt.params,lr=1e-4,betas=(.9,.99),eps=1e-8,weight_decay=0,foreach=False)
        opt.load_state_dict(private['optimizer'])
        for i,(name,p) in enumerate(ordered):
            assert torch.equal(p.detach().cpu(),adapter['adapter'][name])
            for key in ('step','exp_avg','exp_avg_sq'):
                assert torch.equal(opt.state[p][key].detach().cpu(),private['optimizer']['state'][i][key])
        log('adapter_and_optimizer_restored',**bundle)

        def compare(a,b):
            d=a.float()-b.float()
            return {'max_abs':float(d.abs().max()),
                    'relative_rms':float(d.square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-8))}

        s=samples[0];sigma=.909375;hr=s['reference'];draft=s['upscaled']
        noise=position_noise((24,1,68,120),s['sample_id'],1,999)
        x=torch.cat((hr[:,:1],(1-sigma)*hr[:,1:2]+sigma*noise),1)
        with torch.no_grad():
            offline=rt.offline(x,draft[:,:2],condition,[0,1],sigma)
            rt.start_stream(condition);rt.stream_velocity(hr[:,:1],draft[:,:1],0,commit=True)
            streamed=rt.stream_velocity(x[:,1:2],draft[:,1:2],sigma)
            parity=compare(streamed,offline);assert parity['max_abs']<=.03 and parity['relative_rms']<=.005
            full=rt.offline(x,draft[:,:2],condition,[0,1],sigma,return_full=True)
            changed=x.clone();changed[:,1:]*=-3
            changed=rt.offline(changed,draft[:,:2],condition,[0,1],sigma,return_full=True)
            future=compare(changed[:,:1],full[:,:1]);assert future['max_abs']==0
            changed_draft=draft[:,:2].clone();changed_draft[:,1:]*=-3
            changed=rt.offline(x,changed_draft,condition,[0,1],sigma,return_full=True)
            condition_future=compare(changed[:,:1],full[:,:1]);assert condition_future['max_abs']==0
            rt.cache=None
        pred=rt.offline(x,draft[:,:2],condition,[0,1],sigma,train=True)
        gate_loss=(pred-(hr[:,1:2]-noise).to(rt.device)).square().mean();gate_loss.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(rt.params,1.))
        assert norm>0 and torch.isfinite(torch.tensor(norm))
        opt.zero_grad(set_to_none=True)
        torch.save({'adapter':rt.adapter_state()},out/'private_gate_reload.pt')
        restored=torch.load(out/'private_gate_reload.pt',weights_only=True,map_location='cpu')['adapter']
        rt.load_adapter(restored)
        with torch.no_grad():reloaded=rt.offline(x,draft[:,:2],condition,[0,1],sigma)
        reload_error=compare(reloaded,offline);assert reload_error['max_abs']==0
        assert all(float(opt.state[p]['step'])==200 for p in rt.params)
        gates={'status':'passed','cache_parity':parity,'future':future,'condition_future':condition_future,
               'gradient_norm':norm,'reload':reload_error,'optimizer_state_step':200,
               'new_optimizer_updates':0,'scope':'real50block full-spatial two-latent continuation gates'}
        write('gates.json',gates);log('continuation_gates_passed',**gates)
        del adapter,private,pred,gate_loss,offline,streamed,full,changed,changed_draft,reloaded,restored

        def save(step):
            ap=out/f'adapter_step{step:04d}.pt';op=out/f'private_resume_step{step:04d}.pt'
            for path,value in [(ap,{'adapter':rt.adapter_state(),'step':step,
                    'format':'streaming-refiner-adapter-v1','upsampler_sha256':config['upsampler_sha256']}),
                (op,{'optimizer':cpu_tree(opt.state_dict()),'step':step,'parameter_names':[n for n,p in ordered],
                    'torch_rng_state':torch.get_rng_state(),'npu_rng_state':torch.npu.get_rng_state().cpu().clone(),
                    'next_schedule':schedule(step),'resume_origin_step':200})]:
                assert not path.exists();tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
            record={'step':step,'adapter_sha256':sha(ap),'private_optimizer_sha256':sha(op),
                    'adapter_bytes':ap.stat().st_size,'private_optimizer_bytes':op.stat().st_size,
                    'private_state_CPU_tensors':True,'inference_weights_accepted':False}
            write(f'checkpoint_step{step:04d}.json',record);log('checkpoint_saved',**record)

        for step in range(200,3000):
            spec=schedule(step);s=samples[spec['sample_index']];frames=spec['frames'];sigma=spec['sigma']
            hr=s['reference'][:,frames];draft=s['upscaled'][:,frames]
            noise=position_noise((24,1,68,120),s['sample_id'],spec['latent_index'],spec['noise_seed'])
            x=hr.clone();x[:,-1:]=(1-sigma)*hr[:,-1:]+sigma*noise
            opt.zero_grad(set_to_none=True)
            pred=rt.offline(x,draft,condition,frames,sigma,train=True)
            target=(hr[:,-1:]-noise).to(rt.device);x0=x[:,-1:].to(rt.device)+sigma*pred
            fm=(pred-target).square().mean();draft_loss=(x0-draft[:,-1:].to(rt.device)).abs().mean()
            loss=fm+.05*draft_loss;assert bool(torch.isfinite(loss));loss.backward()
            grad=float(torch.nn.utils.clip_grad_norm_(rt.params,1.));assert grad>0 and torch.isfinite(torch.tensor(grad))
            opt.step();updates=step+1
            assert all(float(opt.state[p]['step'])==updates for p in rt.params)
            log('optimizer_update',step=updates,sample=s['sample_id'],latent_index=spec['latent_index'],
                history=frames[:-1],sigma=sigma,noise_seed=spec['noise_seed'],fm=float(fm.detach()),
                draft_l1=float(draft_loss.detach()),loss=float(loss.detach()),grad_norm=grad,
                peak_allocated_bytes=torch.npu.max_memory_allocated())
            write('status.json',{'status':'TF_training','optimizer_updates':updates,'new_optimizer_updates':updates-200,
                'target':3000,'last_loss':float(loss.detach()),'updated_unix':time.time(),'selected_refiner':None})
            if updates%100==0:save(updates)
        write('status.json',{'status':'TF3000_complete_evaluation_pending','optimizer_updates':updates,
            'new_optimizer_updates':updates-200,'selected_refiner':None,'full226_evaluation_complete':False})
        log('TF3000_complete_evaluation_pending')
    except BaseException as error:
        write('failure.json',{'type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),
            'optimizer_updates':updates,'new_optimizer_updates':updates-200});raise


if __name__=='__main__':main()
