"""Audit a completed TF pilot without loading optimizer/RNG or any accelerator."""
import argparse,collections,json,math,time
from pathlib import Path
import torch
from prepare_pairs import sha

def main():
    p=argparse.ArgumentParser()
    for n in ('training','launch','stage','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();train=Path(a.training);stage=Path(a.stage);out=Path(a.output)
    status=json.loads((train/'status.json').read_text());launch=json.loads(Path(a.launch).read_text())
    assert status['status']=='pilot_training_complete_evaluation_pending' and status['optimizer_updates']==200
    assert launch['status']=='complete' and launch['exit_code']==0 and launch['child_reaped']
    assert not (train/'failure.json').exists()
    config=json.loads((stage/'config/pilot_v3.json').read_text())
    assert all(sha(stage/n)==v for n,v in config['new_sources'].items())
    rows=[json.loads(l) for l in (train/'events.jsonl').read_text().splitlines()]
    updates=[r for r in rows if r['event']=='optimizer_update'];saves=[r for r in rows if r['event']=='checkpoint_saved']
    assert len(updates)==200 and [r['step'] for r in updates]==list(range(1,201))
    assert [r['optimizer_updates'] for r in updates]==list(range(1,201))
    assert [r['step'] for r in saves]==[0,50,100,150,200]
    assert all(math.isfinite(r[k]) for r in updates for k in ('loss','fm','draft_l1','grad_norm'))
    assert all(r['grad_norm']>0 for r in updates)
    assert rows[-1]['event']=='pilot_training_complete_evaluation_pending'
    initial=None;checkpoints=[];torch.set_num_threads(4)
    for row in saves:
        step=row['step'];path=train/f'adapter_step{step:04d}.pt';assert sha(path)==row['sha256']
        blob=torch.load(path,map_location='cpu',weights_only=True);state=blob['adapter']
        assert blob['step']==step and blob['format']=='streaming-refiner-adapter-v1'
        assert blob['upsampler_sha256']=='7b9f43c17462fb2d0554645bc7889dc7abdebc052ee9f18f6780624de912e0b5'
        assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in state.values())
        assert sum(v.numel() for v in state.values())==20586496
        if step==0:initial={k:v.clone() for k,v in state.items()}
        assert set(state)==set(initial)
        changed=sum(not torch.equal(v,initial[k]) for k,v in state.items())
        delta_l2=math.sqrt(sum(float((v.double()-initial[k].double()).square().sum()) for k,v in state.items()))
        condition_norm=float(state['condition.weight'].norm())
        assert (changed==0 and condition_norm==0) if step==0 else (changed>0 and condition_norm>0)
        checkpoints.append({'step':step,'filename':path.name,'sha256':row['sha256'],'bytes':path.stat().st_size,
                            'tensor_count':len(state),'parameters':20586496,'all_FP32_finite':True,
                            'changed_tensors_from_step0':changed,'parameter_delta_l2_from_step0':delta_l2,
                            'condition_weight_l2':condition_norm})
    by_sample={}
    for sid in sorted(set(r['sample'] for r in updates)):
        rs=[r for r in updates if r['sample']==sid]
        assert len(rs)==100 and set(r['latent_index'] for r in rs)==set(range(67))
        by_sample[sid]={'updates':len(rs),'absolute_latents_covered':67,
                       'sigma_counts':dict(collections.Counter(str(r['sigma']) for r in rs))}
    report={'status':'real200_update_TF_pilot_complete_evaluation_pending','optimizer_updates':200,
            'audit_source_sha256':sha(__file__),
            'training_event_sha256':sha(train/'events.jsonl'),'config_sha256':sha(stage/'config/pilot_v3.json'),
            'source_hashes_verified':True,'parent_exit_code':0,'child_reaped':True,
            'all200_gradients_finite_nonzero':True,'samples':by_sample,'checkpoints':checkpoints,
            'elapsed_training_seconds':rows[-1]['elapsed_seconds'],
            'max_allocated_NPU_bytes':max(r['peak_allocated_bytes'] for r in updates),
            'full67_inference_validated':False,'full226_evaluation_complete':False,'visual_quality_passed':False,
            'strict_decoder_validated':False,'optimizer_or_RNG_loaded':False,'nativeHR_guard':'not_yet_evaluated'}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'updates':200,'checkpoint_steps':[r['step'] for r in checkpoints]}),flush=True)

if __name__=='__main__':main()
