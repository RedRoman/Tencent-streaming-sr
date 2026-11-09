"""Released merged H3 bidirectional reference, same draft/noise/condition.

All video rows use the same noisy time. Static rows may read video, as in the
unmasked released model. No trained streaming adapter is loaded. This is not
Runtime.offline(dense=True), whose clean-history time schedule serves TF.
"""
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from prepare_pairs import sha
from paged_runtime import Runtime,LayerCall,AdapterLinear
from causal_attention import AttentionCall,position_noise
from open_h3.geometry import MiniMaxH3Geometry,unpatchify_video_rows

@torch.no_grad()
def check_upstream(rt,x,draft,sample,sigma):
    """Independent unmodified upstream orchestration, paged by hooks only."""
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3AttnProcessor
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
    from open_h3.geometry import pack_audio_latents
    got=dense_velocity(rt,x,draft,sample,sigma)
    static,sp,tags=rt.static_inputs(sample);ns=static.shape[1];nf=x.shape[1];nv=2040*nf
    positions=torch.cat((sp,rt.video_positions(list(range(nf)))))
    modalities=torch.cat((tags,torch.zeros(nv,dtype=torch.long,device=rt.device)))
    times=torch.cat((torch.zeros(ns,dtype=torch.long,device=rt.device),torch.ones(nv,dtype=torch.long,device=rt.device)))
    rows=patchify_video_latents(x[None].to(rt.device),rt.model.config.patch_size).reshape(1,nv,96).bfloat16()
    handles=[]
    try:
        for i,block in enumerate(rt.model.transformer_blocks):
            block.attn.set_processor(MiniMaxH3AttnProcessor())
            handles.append(block.register_forward_pre_hook(lambda m,args,i=i:rt.load_layer(i)))
            handles.append(block.register_forward_hook(lambda m,args,z,i=i:rt.release_layer(i)))
        rt.time([1.,1.-sigma])  # restore Sharp coordinates for upstream forward
        audio=sample['audio_latents'];text=sample['prompt_embeds']
        if audio.ndim==3:audio=audio[None]
        if text.ndim==2:text=text[None]
        raw=rt.model(hidden_states=rows,audio_hidden_states=pack_audio_latents(audio.to(rt.device)),
            encoder_hidden_states=text.to(rt.device),timestep=torch.tensor([1.,1.-sigma],device=rt.device),
            timestep_indices=times,token_tags=modalities,position_ids=positions,
            video_indices=torch.arange(ns,ns+nv,device=rt.device),audio_indices=torch.arange(ns,device=rt.device),
            text_indices=torch.empty(0,dtype=torch.long,device=rt.device),return_dict=False)[0]
        expected=unpatchify_video_rows(raw,MiniMaxH3Geometry(nf,68,120,1))[0].float()
        d=got-expected
        report={'max_abs':float(d.abs().max()),'relative_rms':float(d.square().mean().sqrt()/expected.square().mean().sqrt().clamp_min(1e-8)),
                'scope':'full_model_two_latent_dense_vs_upstream_forward','no_text':text.shape[1]==0}
        assert report['no_text'],'This explicit modality comparison gate currently requires no-text pilot'
        assert report['max_abs']<=.03 and report['relative_rms']<=.005,report
        return report
    finally:
        for h in handles:h.remove()
        for i,block in enumerate(rt.model.transformer_blocks):block.attn.set_processor(rt.processors[i])

@torch.no_grad()
def dense_velocity(rt,x,draft,sample,sigma):
    frames=x.shape[1]
    static,sp,tags=rt.static_inputs(sample)
    video=rt.embeds(x.to(rt.device),draft.to(rt.device));h=torch.cat((static,video),1)
    pos=torch.cat((sp,rt.video_positions(list(range(frames)))))
    rotary=rt.model.rope(pos);temb,curve=rt.time([1.,1.-sigma])
    times=torch.cat((torch.zeros(static.shape[1],dtype=torch.long,device=rt.device),
                     torch.ones(video.shape[1],dtype=torch.long,device=rt.device)))
    modalities=torch.cat((tags,torch.zeros(video.shape[1],dtype=torch.long,device=rt.device)))
    call=LayerCall(temb,times*3+modalities,rotary,curve,
                   AttentionCall((-1,)*static.shape[1]+(0,)*video.shape[1],dense=True))
    for i in range(len(rt.model.transformer_blocks)):
        h=rt.layer(i,h,call)
        if (i+1)%10==0:rt.log('dense_blocks_complete',sigma=sigma,blocks=i+1)
    out=rt.model.proj_out(rt.model.norm_out(h[:,static.shape[1]:],temb,times[static.shape[1]:]).float())
    return unpatchify_video_rows(out,MiniMaxH3Geometry(frames,68,120,1))[0].float()

def main():
    import torch_npu
    p=argparse.ArgumentParser()
    for n in ('root','data-manifest','data-manifest-sha','protocol','sample-id','output','assets-manifest','assets-manifest-sha'):
        p.add_argument('--'+n,required=True)
    p.add_argument('--resident',type=int,default=4);p.add_argument('--query-chunk',type=int,default=64)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.time()
    def log(event,**v):
        row={'event':event,'elapsed_seconds':time.time()-started,**v};print(json.dumps(row),flush=True)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    try:
        assert sha(a.data_manifest)==a.data_manifest_sha and sha(a.assets_manifest)==a.assets_manifest_sha
        assets=json.loads(Path(a.assets_manifest).read_text());assert assets['status']=='complete'
        for name,d in assets['files'].items():
            s=(Path(a.root)/name).stat();assert s.st_size==d['bytes'] and s.st_mtime_ns==d['mtime_ns'] and s.st_ino==d['inode']
        data=json.loads(Path(a.data_manifest).read_text());protocol=json.loads(Path(a.protocol).read_text())
        assert data['protocol_sha256']==sha(a.protocol) and a.sample_id in protocol['dev']
        rec=next(r for r in data['samples'] if r['sample_id']==a.sample_id)
        assert sha(rec['path'])==rec['sha256'] and sha(data['condition']['path'])==data['condition']['sha256']
        s=torch.load(rec['path'],weights_only=True,map_location='cpu');condition=torch.load(data['condition']['path'],weights_only=True,map_location='cpu')
        torch.set_num_threads(4);torch.manual_seed(20261109);torch.npu.set_device('npu:0')
        torch.npu.conv.allow_hf32=False;torch.npu.matmul.allow_hf32=False;torch.set_float32_matmul_precision('highest')
        rt=Runtime(a.root,resident=a.resident,canonical_rows=False,log=log)
        assert torch.count_nonzero(rt.condition.weight)==0
        assert all(torch.count_nonzero(m.b)==0 for m in rt.model.modules() if isinstance(m,AdapterLinear))
        for proc in rt.processors:proc.chunk_size=a.query_chunk
        draft=s['upscaled'];sigmas=protocol['shared_input']['sigmas']
        noise=torch.cat([position_noise((24,1,68,120),a.sample_id,f,20261109) for f in range(67)],1)
        parity=check_upstream(rt,(1-sigmas[0])*draft[:,:2]+sigmas[0]*noise[:,:2],draft[:,:2],condition,sigmas[0])
        (out/'upstream_gate.json').write_text(json.dumps(parity,indent=2)+'\n');log('dense_upstream_gate_passed',**parity)
        x=((1-sigmas[0])*draft+sigmas[0]*noise).to(rt.device)
        for sigma,next_sigma in zip(sigmas,sigmas[1:]):
            x=x+(sigma-next_sigma)*dense_velocity(rt,x,draft,condition,sigma)
            assert torch.isfinite(x).all();log('dense_denoise_step_complete',sigma=sigma,next_sigma=next_sigma)
        path=out/'dense_latents.npy'
        with path.open('xb') as f:np.save(f,x.cpu().numpy(),allow_pickle=False)
        (out/'result.json').write_text(json.dumps({'status':'full67_latent_inference_complete_pixels_pending','arm':'released_dense_refiner',
            'sample_id':a.sample_id,'training_step':None,'checkpoint_sha256':None,'trained_streaming_adapter_loaded':False,
            'output':{'path':str(path),'sha256':sha(path),'shape':[1,24,67,68,120]},'data_manifest_sha256':a.data_manifest_sha,
            'protocol_sha256':sha(a.protocol),'optimizer_updates':0,'model_forwards':3,'peak_allocated_bytes':torch.npu.max_memory_allocated(),
            'same_draft_noise_condition_sigmas':True,'full226_pixels_complete':False,'visual_quality_passed':False},indent=2)+'\n')
    except BaseException as e:
        (out/'failure.json').write_text(json.dumps({'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()},indent=2)+'\n');raise

if __name__=='__main__':main()
