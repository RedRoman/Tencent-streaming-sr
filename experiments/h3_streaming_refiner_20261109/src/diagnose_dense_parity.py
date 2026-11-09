"""Full real two-latent dense trace; zero optimizer updates or threshold edits."""
import argparse,hashlib,json,sys,time
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    for n in ('worker-stage','root','data-manifest','assets-manifest','assets-manifest-sha','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();stage=Path(a.worker_stage);pin=json.loads((stage/'config/evaluation_workers_v4.json').read_text())
    assert all(sha(stage/n)==v for n,v in pin['new_sources'].items())
    assert sha(a.data_manifest)==pin['dev_manifest_sha256'] and sha(a.assets_manifest)==a.assets_manifest_sha
    for name,d in json.loads(Path(a.assets_manifest).read_text())['files'].items():
        s=(Path(a.root)/name).stat();assert (s.st_size,s.st_mtime_ns,s.st_ino)==(d['bytes'],d['mtime_ns'],d['inode'])
    sys.path.insert(0,str(stage/'src'))
    import torch,torch_npu
    import evaluate_dense_reference as worker
    from paged_runtime import Runtime
    from causal_attention import position_noise
    from open_h3.geometry import MiniMaxH3Geometry,unpatchify_video_rows
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.time()
    torch.set_num_threads(4);torch.manual_seed(20261109);torch.npu.set_device('npu:0')
    torch.npu.conv.allow_hf32=False;torch.npu.matmul.allow_hf32=False;torch.set_float32_matmul_precision('highest')
    data=json.loads(Path(a.data_manifest).read_text());rec=data['samples'][0]
    assert sha(rec['path'])==rec['sha256'] and sha(data['condition']['path'])==data['condition']['sha256']
    pair=torch.load(rec['path'],weights_only=True,map_location='cpu');condition=torch.load(data['condition']['path'],weights_only=True,map_location='cpu')
    rt=Runtime(a.root,resident=4,canonical_rows=False)
    for proc in rt.processors:proc.chunk_size=64
    phase='custom';captured={};report={'optimizer_updates':0,'scope':'actual full33B, full68x120, two latent dense trace','modules':{}}
    def diff(x,y):
        if x.shape!=y.shape:return {'shape_mismatch':[list(x.shape),list(y.shape)]}
        d=x.float()-y.float();return {'max_abs':float(d.abs().max()),'relative_rms':float(d.square().mean().sqrt()/y.float().square().mean().sqrt().clamp_min(1e-8))}
    names=['proj_in','audio_proj_in','time_proj','time_embedder','norm_out','proj_out']
    names += ['transformer_blocks.0.'+n for n in ('adaln_proj','norm1','attn.to_q','attn.to_k','attn.to_v','attn.norm_q','attn.norm_k','attn','norm2','ff')]
    names += [f'transformer_blocks.{i}' for i in range(50)]
    handles=[];full_last={};full_projection={}
    def tensor(z):
        if isinstance(z,tuple):z=z[0]
        return z.detach().cpu() if isinstance(z,torch.Tensor) else None
    for name in names:
        def hook(m,args,z,name=name):
            value=tensor(z)
            if value is None:return
            if name=='transformer_blocks.49':full_last[phase]=value
            if name=='proj_out':full_projection[phase]=value
            if phase=='custom':captured[name]=value
            elif name in captured:report['modules'][name]=diff(captured[name],value)
        handles.append(rt.model.get_submodule(name).register_forward_hook(hook))
    original=worker.dense_velocity;custom_output=None
    def wrapped(*args,**kwargs):
        nonlocal phase,custom_output
        phase='custom';z=original(*args,**kwargs);custom_output=z.cpu();phase='upstream';return z
    worker.dense_velocity=wrapped
    sigma=.909375;draft=pair['upscaled'][:,:2]
    noise=torch.cat([position_noise((24,1,68,120),rec['sample_id'],f,20261109) for f in range(2)],1)
    with torch.no_grad():
        try:report['original_gate']=worker.check_upstream(rt,(1-sigma)*draft+sigma*noise,draft,condition,sigma)
        except AssertionError as e:report['original_gate_failure']=str(e)
        finally:
            for h in handles:h.remove()
            worker.dense_velocity=original
        ns=full_last['custom'].shape[1]-4080
        expected=unpatchify_video_rows(full_projection['upstream'][:,ns:],MiniMaxH3Geometry(2,68,120,1))[0].float()
        report['original_output']=diff(custom_output,expected)
        times=torch.cat((torch.zeros(ns,dtype=torch.long,device=rt.device),torch.ones(4080,dtype=torch.long,device=rt.device)))
        temb,_=rt.time([1.,1.-sigma])
        fixed=rt.model.proj_out(rt.model.norm_out(full_last['custom'].to(rt.device),temb,times).float())[:,ns:]
        fixed=unpatchify_video_rows(fixed,MiniMaxH3Geometry(2,68,120,1))[0].float().cpu()
        report['full_row_output_head_candidate']=diff(fixed,expected)
    report.update(status='trace_complete_not_acceptance',seconds=time.time()-start,peak_allocated_bytes=torch.npu.max_memory_allocated())
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
