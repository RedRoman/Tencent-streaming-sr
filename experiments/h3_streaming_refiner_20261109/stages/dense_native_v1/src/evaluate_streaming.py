"""One immutable checkpoint/clip, full67 streaming latent evaluation.

Run only inside the three-lock launcher. This records refiner output and
cache/chunk diagnostics; it does not claim decoder or visual acceptance.
"""
import argparse, json, time, traceback
from pathlib import Path
import numpy as np
import torch, torch_npu
from prepare_pairs import sha
from paged_runtime import Runtime
from causal_attention import position_noise
from streaming_refiner import StreamingRefiner

def compare(a,b):
    d=a.float()-b.float()
    return {'max_abs':float(d.abs().max()),
            'relative_rms':float(d.square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-8))}

def main():
    p=argparse.ArgumentParser()
    for n in ('root','data-manifest','data-manifest-sha','checkpoint','checkpoint-sha','protocol','output','sample-id','assets-manifest','assets-manifest-sha'):
        p.add_argument('--'+n,required=True)
    p.add_argument('--resident',type=int,default=24)
    p.add_argument('--gates-only',action='store_true')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.time()
    def log(event,**v):
        row={'event':event,'elapsed_seconds':time.time()-started,**v}
        print(json.dumps(row),flush=True)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    def write(name,v):(out/name).write_text(json.dumps(v,indent=2)+'\n')
    try:
        assert sha(a.data_manifest)==a.data_manifest_sha and sha(a.checkpoint)==a.checkpoint_sha
        assert sha(a.assets_manifest)==a.assets_manifest_sha
        assets=json.loads(Path(a.assets_manifest).read_text());assert assets['status']=='complete'
        for name,d in assets['files'].items():
            stat=(Path(a.root)/name).stat()
            assert stat.st_size==d['bytes'] and stat.st_mtime_ns==d['mtime_ns'] and stat.st_ino==d['inode']
        manifest=json.loads(Path(a.data_manifest).read_text());protocol=json.loads(Path(a.protocol).read_text())
        assert manifest['protocol_sha256']==sha(a.protocol) and a.sample_id in protocol['dev']
        rec=next(r for r in manifest['samples'] if r['sample_id']==a.sample_id)
        assert sha(rec['path'])==rec['sha256']
        assert sha(manifest['condition']['path'])==manifest['condition']['sha256']
        sample=torch.load(rec['path'],map_location='cpu',weights_only=True)
        condition=torch.load(manifest['condition']['path'],map_location='cpu',weights_only=True)
        checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
        assert checkpoint['step'] in protocol['checkpoints'] and checkpoint['format']=='streaming-refiner-adapter-v1'
        training_step=checkpoint['step']
        assert checkpoint['upsampler_sha256']==manifest['accepted_upsampler_sha256']
        assert all(torch.isfinite(v).all() for v in checkpoint['adapter'].values())
        torch.set_num_threads(4);torch.manual_seed(20261109);torch.npu.set_device('npu:0')
        torch.npu.conv.allow_hf32=False;torch.npu.matmul.allow_hf32=False;torch.set_float32_matmul_precision('highest')
        rt=Runtime(a.root,resident=a.resident,canonical_rows=True,log=log);rt.load_adapter(checkpoint['adapter'])
        draft=sample['upscaled'];hr=sample['reference'];sigma=.909375
        noise=position_noise((24,1,68,120),a.sample_id,1,999)
        noisy=(1-sigma)*hr[:,1:2]+sigma*noise
        with torch.no_grad():
            x=torch.cat((hr[:,:1],noisy),1)
            offline=rt.offline(x,draft[:,:2],condition,[0,1],sigma)
            rt.start_stream(condition);rt.stream_velocity(hr[:,:1],draft[:,:1],0,commit=True)
            streamed=rt.stream_velocity(noisy,draft[:,1:2],sigma)
            parity=compare(streamed,offline)
            assert parity['max_abs']<=.03 and parity['relative_rms']<=.005,parity
            past=rt.offline(x,draft[:,:2],condition,[0,1],sigma,return_full=True)[:,:1]
            changed=x.clone();changed[:,1:]*=-3
            future=compare(rt.offline(changed,draft[:,:2],condition,[0,1],sigma,return_full=True)[:,:1],past)
            assert future['max_abs']==0,future
            changed_draft=draft[:,:2].clone();changed_draft[:,1:]*=-5
            condition_future=compare(rt.offline(x,changed_draft,condition,[0,1],sigma,return_full=True)[:,:1],past)
            assert condition_future['max_abs']==0,condition_future
            rt.load_adapter(checkpoint['adapter'])
            reload=compare(rt.offline(x,draft[:,:2],condition,[0,1],sigma),offline)
            assert reload['max_abs']==0,reload
        write('trained_gates.json',{'cache_parity':parity,'future':future,'condition_future':condition_future,'reload':reload,'checkpoint_sha256':a.checkpoint_sha})
        log('trained_gates_passed',cache_parity=parity,future=future,condition_future=condition_future,reload=reload)
        if a.gates_only:
            write('result.json',{'status':'two_latent_gates_only_passed','sample_id':a.sample_id,
                  'checkpoint_sha256':a.checkpoint_sha,'training_step':training_step,
                  'gates':{'cache_parity':parity,'future':future,'condition_future':condition_future,'reload':reload},
                  'full67_inference_complete':False,'full226_pixels_complete':False,'optimizer_updates':0})
            return
        del offline,streamed,past,x,changed,changed_draft,checkpoint
        initialization_start=time.time()
        runner=StreamingRefiner(rt,condition,sample_id=a.sample_id)
        torch.npu.synchronize();initialization_seconds=time.time()-initialization_start
        values=[];audits=[];latencies=[]
        for i in range(67):
            start=time.time();z=runner.push(draft[None,:,i:i+1]);torch.npu.synchronize()
            elapsed=time.time()-start;values.append(z);latencies.append(elapsed);audit=runner.audit();audits.append(audit)
            assert audit['next_latent']==i+1 and audit['model_forwards']==4*(i+1)
            assert audit['max_cached_video_frames']<=2 and audit['recomputed_history_latents']==0
            log('latent_complete',latent=i,seconds=elapsed,cache_bytes=audit['cache_bytes'])
        result=torch.cat(values,2);assert result.shape==(1,24,67,68,120) and torch.isfinite(result).all()
        assert len(set(v['cache_bytes'] for v in audits[1:]))==1,'Cache grows after two latent frames'
        path=out/'streaming_latents.npy'
        with path.open('xb') as f:np.save(f,result[0].numpy(),allow_pickle=False)
        # A real rerun with a different API chunk boundary, using this trained
        # adapter and the same absolute-position noise. Four-latent scope only.
        check=StreamingRefiner(rt,condition,sample_id=a.sample_id)
        other=check.push(draft[None,:,:4]);chunk=compare(other,result[:,:,:4])
        assert chunk['max_abs']==0,chunk
        write('result.json',{'status':'full67_latent_inference_complete_pixels_pending','sample_id':a.sample_id,
              'checkpoint_sha256':a.checkpoint_sha,'training_step':training_step,
              'output':{'path':str(path),'sha256':sha(path),'shape':list(result[0].shape)},'pipeline_shape':list(result.shape),
              'trained_gates_passed':True,'chunk_invariance_first4':chunk,'cache_bytes_trace':[v['cache_bytes'] for v in audits],
              'per_latent_seconds':latencies,'model_forwards_full67':268,'static_initialization_forwards_extra':1,
              'static_initialization_seconds':initialization_seconds,'time_to_first_latent_seconds_excluding_model_load_and_gates':initialization_seconds+latencies[0],
              'peak_allocated_bytes':torch.npu.max_memory_allocated(),'actual_test_optimizer_updates':0,
              'full226_pixels_complete':False,'visual_quality_passed':False,'decoder_streaming_validated':False,
              'longer_than67_tested':False,'data_manifest_sha256':a.data_manifest_sha,'protocol_sha256':sha(a.protocol)})
        log('full67_complete_chunk4_passed',chunk=chunk)
    except BaseException as e:
        write('failure.json',{'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
