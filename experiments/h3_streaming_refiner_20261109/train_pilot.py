"""200-step finite-data native-HR Teacher Forcing pilot, after mandatory gates."""
import argparse,hashlib,json,time,os,traceback
from pathlib import Path
import torch
import torch_npu
from causal_attention import position_noise
from prepare_pairs import prepare,sha
from paged_runtime import Runtime

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--prior-stage',required=True);p.add_argument('--config',required=True);p.add_argument('--assets-manifest',required=True);p.add_argument('--assets-manifest-sha',required=True);p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=200);p.add_argument('--resident',type=int,default=24);args=p.parse_args()
out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
started=time.time();updates=0

def log(event,**values):
    row={'event':event,'elapsed_seconds':time.time()-started,'optimizer_updates':updates,**values}
    print(json.dumps(row),flush=True)
    with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')

def write(name,value):
    tmp=out/(name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(out/name)

def compare(a,b):
    d=(a.float()-b.float());return {'max_abs':float(d.abs().max()),'relative_rms':float(d.square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-8))}

def save(runtime,opt,step):
    path=out/f'adapter_step{step:04d}.pt';torch.save({'adapter':runtime.adapter_state(),'step':step,'format':'streaming-refiner-adapter-v1','upsampler_sha256':'7b9f43c17462fb2d0554645bc7889dc7abdebc052ee9f18f6780624de912e0b5'},path)
    # Private resume state is kept separate, never a publication asset.
    if opt is not None:torch.save({'optimizer':opt.state_dict(),'step':step},out/f'private_resume_step{step:04d}.pt')
    log('checkpoint_saved',step=step,path=str(path),sha256=sha(path))

try:
    config=json.loads(Path(args.config).read_text())
    assert config['steps']==args.steps and config['resident_layers']==args.resident
    stage=Path(__file__).resolve().parent.parent
    assert all(sha(stage/name)==digest for name,digest in config['new_sources'].items())
    assert sha(args.assets_manifest)==args.assets_manifest_sha
    assets=json.loads(Path(args.assets_manifest).read_text());assert assets['status']=='complete'
    for name,desc in assets['files'].items():
        stat=(Path(args.root)/name).stat();assert stat.st_size==desc['bytes'] and stat.st_mtime_ns==desc['mtime_ns'] and stat.st_ino==desc['inode']
    torch.set_num_threads(4);torch.manual_seed(20261109);torch.npu.set_device('npu:0')
    torch.npu.conv.allow_hf32=False;torch.npu.matmul.allow_hf32=False;torch.set_float32_matmul_precision('highest')
    log('preparation_start',physical_npu=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),torch=str(torch.__version__),torch_npu=str(torch_npu.__version__))
    data=out.parent/'data_v1'
    if not (data/'manifest.json').exists():manifest=prepare(args.root,data,log,args.prior_stage)
    else:manifest=json.loads((data/'manifest.json').read_text())
    for row in manifest['samples']:assert sha(row['path'])==row['sha256']
    assert sha(data/'conditioning.pt')==manifest['condition_sha256']
    samples=[torch.load(row['path'],map_location='cpu',weights_only=True) for row in manifest['samples']]
    condition=torch.load(str(data/'conditioning.pt'),map_location='cpu',weights_only=True)
    rt=Runtime(args.root,resident=args.resident,log=log)
    sigma=.909375;s=samples[0];draft=s['upscaled'];hr=s['reference']
    noise=position_noise((24,1,68,120),s['sample_id'],1,999)
    noisy=(1-sigma)*hr[:,1:2]+sigma*noise
    with torch.no_grad():
        offline=rt.offline(torch.cat((hr[:,:1],noisy),1),draft[:,:2],condition,[0,1],sigma)
        rt.start_stream(condition)
        rt.stream_velocity(hr[:,:1],draft[:,:1],0,commit=True)
        streamed=rt.stream_velocity(noisy,draft[:,1:2],sigma)
        parity=compare(streamed,offline);log('real_model_cache_parity',**parity)
        assert parity['relative_rms']<=.005 and parity['max_abs']<=.03,'Real-model cache parity failed frozen tolerance'
        x=torch.cat((hr[:,:1],noisy),1)
        full=rt.offline(x,draft[:,:2],condition,[0,1],sigma,return_full=True)
        x[:,1:]*=-3
        changed=rt.offline(x,draft[:,:2],condition,[0,1],sigma,return_full=True)
        leak=compare(changed[:,:1],full[:,:1]);log('real_model_future_perturbation',**leak)
        assert leak['max_abs']==0,'Future leakage or nondeterminism'
        rt.cache=None
    # Real full spatial backward, no optimizer update during gate.
    pred=rt.offline(torch.cat((hr[:,:1],noisy),1),draft[:,:2],condition,[0,1],sigma,train=True)
    loss=(pred-(hr[:,1:2]-noise).to(rt.device)).square().mean();loss.backward()
    norm=float(torch.nn.utils.clip_grad_norm_(rt.params,1.));assert norm>0 and torch.isfinite(torch.tensor(norm))
    for pp in rt.params:pp.grad=None
    save(rt,None,0);initial=rt.adapter_state();rt.load_adapter(torch.load(str(out/'adapter_step0000.pt'),weights_only=True)['adapter'])
    assert all(torch.equal(initial[k],v) for k,v in rt.adapter_state().items())
    with torch.no_grad():reloaded=rt.offline(torch.cat((hr[:,:1],noisy),1),draft[:,:2],condition,[0,1],sigma)
    replay=compare(reloaded,offline);assert replay['max_abs']==0
    gates={'status':'passed','cache_parity':parity,'future_perturbation':leak,'gradient_norm':norm,'reload':replay,'geometry':[24,2,68,120],'full67_inference_tested':False,'optimizer_updates':0,'peak_allocated_bytes':torch.npu.max_memory_allocated()}
    write('gates.json',gates);log('gates_passed',**gates)
    del pred,loss,offline,streamed,full,changed,reloaded,initial
    opt=torch.optim.AdamW(rt.params,lr=1e-4,betas=(.9,.99),eps=1e-8,weight_decay=0,foreach=False)
    sigmas=[.909375,6/7,2/3]
    for step in range(args.steps):
        s=samples[step%len(samples)];pos=(step//len(samples)*17)%67;sigma=sigmas[(step//len(samples))%3]
        frames=sorted(set([0,max(0,pos-1),pos]));hr=s['reference'][:,frames];draft=s['upscaled'][:,frames]
        noise=position_noise((24,1,68,120),s['sample_id'],pos,20261109+step//134)
        x=hr.clone();x[:,-1:]=(1-sigma)*hr[:,-1:]+sigma*noise
        opt.zero_grad(set_to_none=True)
        pred=rt.offline(x,draft,condition,frames,sigma,train=True)
        target=(hr[:,-1:]-noise).to(rt.device);x0=x[:,-1:].to(rt.device)+sigma*pred
        fm=(pred-target).square().mean();draft_loss=(x0-draft[:,-1:].to(rt.device)).abs().mean()
        loss=fm+.05*draft_loss;loss.backward()
        grad=float(torch.nn.utils.clip_grad_norm_(rt.params,1.));assert grad>0 and torch.isfinite(torch.tensor(grad))
        opt.step();updates=step+1
        log('optimizer_update',step=updates,sample=s['sample_id'],latent_index=pos,history=frames[:-1],sigma=sigma,fm=float(fm.detach()),draft_l1=float(draft_loss.detach()),loss=float(loss.detach()),grad_norm=grad,peak_allocated_bytes=torch.npu.max_memory_allocated())
        write('status.json',{'status':'training','optimizer_updates':updates,'target':args.steps,'last_loss':float(loss.detach()),'updated_unix':time.time()})
        if updates%50==0 or updates==args.steps:save(rt,opt,updates)
    write('status.json',{'status':'pilot_training_complete_evaluation_pending','optimizer_updates':updates,'visual_quality_passed':False,'full226_evaluation_complete':False})
    log('pilot_training_complete_evaluation_pending')
except BaseException as e:
    write('failure.json',{'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'optimizer_updates':updates,'time':time.time()});log('failed',error=str(e));raise
