"""Exact anchored stage2000 full67 inference; zero optimizer updates."""
import argparse
import json
from pathlib import Path
import time
import traceback
from anchored_evaluation_admission2000 import sha, read, write, verify_worker_lock_ancestor, TRAINING_SHA, UPSAMPLER_SHA

def main():
    p=argparse.ArgumentParser()
    for key in ['root','data-manifest','data-manifest-sha','protocol','checkpoint-binding','checkpoint-binding-sha','sample-id','output']:
        p.add_argument('--'+key,required=True)
    p.add_argument('--physical-device',type=int,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def log(event,**value):
        row={'event':event,'seconds':time.monotonic()-started,**value}
        print(json.dumps(row),flush=True)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    try:
        write(out/'device_lock_owner.json',verify_worker_lock_ancestor(a.physical_device))
        assert sha(a.checkpoint_binding)==a.checkpoint_binding_sha
        binding=read(a.checkpoint_binding);cp=binding['checkpoint'];rec=cp['record']
        assert rec['stage_step']==2000 and rec['total_optimizer_updates']==4500
        assert sha(cp['adapter_path'])==rec['adapter_sha256'] and sha(cp['private_resume_path'])==rec['private_optimizer_sha256']
        assert sha(a.data_manifest)==a.data_manifest_sha
        data=read(a.data_manifest);sample_record=next(x for x in data['samples'] if x['sample_id']==a.sample_id)
        assert sha(sample_record['path'])==sample_record['sha256']
        assert sha(data['condition']['path'])==data['condition']['sha256']
        import torch
        import numpy as np
        from anchored_refiner_v1 import schedule, ResidualHead, AnchoredStreamingRefiner, align_input_projection
        w=torch.load(cp['adapter_path'],map_location='cpu',weights_only=True)
        v=torch.load(cp['private_resume_path'],map_location='cpu',weights_only=True)
        assert w['format']=='anchored-streaming-refiner-v1' and w['residual_bound']==.25
        assert w['base_step']==v['base_step']==2500 and w['stage_step']==v['stage_step']==2000 and w['step']==v['step']==4500
        assert w['manifest_sha256']==v['manifest_sha256']==TRAINING_SHA and w['upsampler_sha256']==UPSAMPLER_SHA
        params=list(w['adapter'].items())+[('residual_head.'+k,t) for k,t in w['residual_head'].items()]
        assert len(w['adapter'])==401 and len(w['residual_head'])==4 and v['parameter_names']==[k for k,t in params]
        assert set(v['optimizer']['state'])==set(range(405)) and v['next_schedule']==schedule(2000)
        for i,(name,t) in enumerate(params):
            state=v['optimizer']['state'][i]
            assert t.device.type=='cpu' and t.dtype==torch.float32 and torch.isfinite(t).all()
            assert float(state['step'])==(4500 if i<401 else 2000)
            for key in ['exp_avg','exp_avg_sq']:
                assert state[key].device.type=='cpu' and state[key].shape==t.shape and torch.isfinite(state[key]).all()
            assert (state['exp_avg_sq']>=0).all()
        for key in ['torch_rng_state','npu_rng_state']:
            assert v[key].device.type=='cpu' and v[key].dtype==torch.uint8 and v[key].numel()>0
        write(out/'checkpoint_CPU_audit.json',{'status':'passed','stage_step':2000,'total_step':4500,
              'all405_optimizer_steps_verified':True,'all_moments_finite':True,'RNG_and_next_schedule_present':True,
              'adapter_sha256':rec['adapter_sha256'],'private_optimizer_sha256':rec['private_optimizer_sha256'],'optimizer_updates':0})
        del v,params
        import torch_npu
        from paged_runtime import Runtime
        from causal_attention import position_noise
        torch.set_num_threads(4);torch.manual_seed(20261109);torch.npu.set_device('npu:0')
        torch.npu.conv.allow_hf32=False;torch.npu.matmul.allow_hf32=False;torch.set_float32_matmul_precision('highest')
        rt=Runtime(a.root,resident=24,canonical_rows=True,log=log);align_input_projection(rt);rt.load_adapter(w['adapter'])
        head=ResidualHead(device=rt.device,bound=.25);head.load_state_dict(w['residual_head']);head.eval()
        sample=torch.load(sample_record['path'],map_location='cpu',weights_only=True)
        condition=torch.load(data['condition']['path'],map_location='cpu',weights_only=True)
        draft=sample['upscaled'];assert draft.shape==(24,67,68,120)
        with torch.no_grad():
            sigma=.909375;x=draft[:,:4].clone()
            x[:,3:4]=(1-sigma)*x[:,3:4]+sigma*position_noise((24,1,68,120),a.sample_id,3,20261109)
            offline=rt.offline(x,draft[:,:4],condition,[0,1,2,3],sigma)
            rt.start_stream(condition)
            for i in range(3):rt.stream_velocity(draft[:,i:i+1],draft[:,i:i+1],0,commit=True)
            streamed=rt.stream_velocity(x[:,3:4],draft[:,3:4],sigma)
            diff=streamed-offline;maximum=float(diff.abs().max());relative=float(diff.square().mean().sqrt()/offline.square().mean().sqrt().clamp_min(1e-8))
            assert maximum<=.03 and relative<=.005
            base=rt.offline(x,draft[:,:4],condition,[0,1,2,3],sigma,return_full=True)[:,:1]
            future=x.clone();future[:,1:]*=-3
            changed_draft=draft[:,:4].clone();changed_draft[:,1:]*=-5
            assert torch.equal(base,rt.offline(future,draft[:,:4],condition,[0,1,2,3],sigma,return_full=True)[:,:1])
            assert torch.equal(base,rt.offline(x,changed_draft,condition,[0,1,2,3],sigma,return_full=True)[:,:1])
        gate={'status':'passed','scope':'real33B four-latent raw same-history and future controls',
              'cache_max_abs':maximum,'cache_relative_rms':relative,'future_max_abs':0,'condition_future_max_abs':0,'optimizer_updates':0}
        write(out/'trained_gates.json',gate);log('trained_gates_passed',**gate)
        del x,base,future,changed_draft,offline,streamed,sample
        runner=AnchoredStreamingRefiner(rt,head,condition,sample_id=a.sample_id,seed=20261109)
        values=[];audits=[];latencies=[]
        for i in range(67):
            start=time.monotonic();z=runner.push(draft[None,:,i:i+1]);torch.npu.synchronize()
            audit=runner.audit();elapsed=time.monotonic()-start
            assert audit['next_latent']==i+1 and audit['model_forwards']==4*(i+1)
            assert audit['max_cached_video_frames']<=2 and audit['recomputed_history_latents']==0
            assert float((z-draft[None,:,i:i+1]).abs().max())<=.25001
            values.append(z);audits.append(audit);latencies.append(elapsed)
            log('latent_complete',latent_zero_based=i,seconds=elapsed,cache_bytes=audit['cache_bytes'])
        result=torch.cat(values,2);assert result.shape==(1,24,67,68,120) and torch.isfinite(result).all()
        assert len(set(v['cache_bytes'] for v in audits[1:]))==1
        path=out/'streaming_latents.npy'
        with path.open('xb') as f:np.save(f,result[0].numpy(),allow_pickle=False)
        # Independent reload and a different push boundary; no history recomputation in either run.
        rt.load_adapter(w['adapter']);head.load_state_dict(w['residual_head'])
        again=AnchoredStreamingRefiner(rt,head,condition,sample_id=a.sample_id,seed=20261109).push(draft[None,:,:4])
        assert torch.equal(again,result[:,:,:4])
        report={'status':'full67_latent_inference_complete_pixels_pending','arm':'anchored_stage2000_candidate',
                'sample_id':a.sample_id,'stage_step':2000,'training_step':4500,'checkpoint_sha256':rec['adapter_sha256'],
                'output':{'path':str(path),'sha256':sha(path),'shape':[24,67,68,120]},'pipeline_shape':list(result.shape),
                'data_manifest_sha256':a.data_manifest_sha,'protocol_sha256':sha(a.protocol),
                'trained_gates_passed':True,'chunk_invariance_first4':{'max_abs':0},'checkpoint_reload_first4_max_abs':0,
                'cache_bytes_trace':[v['cache_bytes'] for v in audits],'model_forwards_full67':268,
                'actual_test_optimizer_updates':0,'per_latent_seconds':latencies,'longer_than67_tested':False,
                'full226_pixels_complete':False,'visual_quality_passed':False,'decoder_streaming_validated':False}
        write(out/'result.json',report);log('full67_and_reload_chunk4_complete')
    except BaseException as e:
        write(out/'failure.json',{'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'optimizer_updates':0})
        raise

if __name__=='__main__':main()
