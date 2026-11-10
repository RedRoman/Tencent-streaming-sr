"""Benchmark exact native-scale cropped TF; no optimizer updates or saved trained weights."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import torch_npu
from safetensors.torch import load_file
from solarwm.backends.minimax_h3.model import SolarMiniMaxH3Transformer3DModel
from solarwm.backends.minimax_h3.omni_causal import training_plan
from ascend_port import install_attention_port
from runtime import TrainingRuntime
from patch_core import PatchCore,batch_for

ROOT=Path('/opt/refiner-cache/h3-pretrained-causal-refiner-20261109')
VENDOR=ROOT/'vendor/SolarRenderer-0ca4f5e45b062cc9e9aee506de46aabdc22e4c3d'

def check_positions(core,sample,condition,crop):
    core.crop_origin=crop[:2]
    whole=core._document(batch_for(sample,condition,0,7),torch.Generator(device=core.device).manual_seed(19))
    part=core._document(batch_for(sample,condition,0,7,crop),torch.Generator(device=core.device).manual_seed(19))
    ws,ps=whole[1].source,part[1].source
    y,x,h,w=crop
    full_positions=ws.position_ids[ws.target_video_indices].reshape(7,34,60,3)
    expected=full_positions[:,y//2:(y+h)//2,x//2:(x+w)//2].reshape(-1,3)
    assert np.array_equal(expected,ps.position_ids[ps.target_video_indices])
    assert np.array_equal(ws.position_ids[ws.audio_indices],ps.position_ids[ps.audio_indices])
    assert np.array_equal(ws.position_ids[ws.video_indices[:2040]],ps.position_ids[ps.video_indices[:2040]])
    plan,_=training_plan(part[3],window=6,teacher=True)
    return {'native_canvas_target_positions_equal':True,'audio_positions_equal':True,
            'full_frame_static_reference_equal':True,'training_rows':plan.sequence_length}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True)
    ap.add_argument('--resume',required=True);ap.add_argument('--resident',type=int,default=24)
    args=ap.parse_args();out=ROOT/'runs'/args.run;out.mkdir(parents=True,exist_ok=False)
    def log(event,**kw):
        row={'event':event,'time':time.time(),'optimizer_updates':0,**kw}
        print(json.dumps(row),flush=True)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    torch.set_num_threads(8);torch.manual_seed(20261110)
    sample=torch.load(ROOT/'data/Charge_train_3856_4081.pt',map_location='cpu',weights_only=True)
    condition=torch.load(ROOT/'data/restoration_prompt.pt',map_location='cpu',weights_only=True)
    silence=load_file(str(VENDOR/'assets/h3_silence_153_158_170.safetensors'))['silence_158f']
    checkpoint=torch.load(args.resume,map_location='cpu',weights_only=True)
    log('strict_load_start')
    model=SolarMiniMaxH3Transformer3DModel.strict_from_pretrained(str(ROOT/'models/causal_4step/transformer'),
        torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True).eval().requires_grad_(False)
    torch.npu.set_device('npu:0');torch.npu.conv.allow_hf32=False;torch.npu.matmul.allow_hf32=False
    install_attention_port();rt=TrainingRuntime(model,'npu:0',rank=16,resident=args.resident,log=log)
    rt.load_adapter(checkpoint['adapter'])
    optimizer=torch.optim.AdamW(rt.params,lr=1e-5,betas=(.9,.99),weight_decay=0,foreach=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    del checkpoint
    core=PatchCore(model,silence,torch.device('npu:0'),{'mode':'teacher','chunk':5,'window':6,'corrupt':0.0})
    records=[]
    for crop in ((16,24,32,64),(16,24,32,48)):
        core.crop_origin=crop[:2]
        positions=check_positions(core,sample,condition,crop)
        log('positions_checked',crop=crop,**positions)
        for repeat in range(3):
            optimizer.zero_grad(set_to_none=True)
            batch=batch_for(sample,condition,0,7,crop)
            torch.npu.reset_peak_memory_stats();torch.npu.synchronize();started=time.time()
            loss=core.forward_loss(batch,noise_seed=2026112000+repeat)
            torch.npu.synchronize();forward=time.time()-started;t=time.time()
            loss.backward();torch.npu.synchronize();backward=time.time()-t;t=time.time()
            assert torch.isfinite(loss)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in rt.params)
            norm=float(torch.nn.utils.clip_grad_norm_(rt.params,1.0));assert norm>0
            torch.npu.synchronize()
            record={'crop':crop,'repeat':repeat,'resident_blocks':args.resident,
                'forward_seconds':forward,'backward_seconds':backward,'validation_seconds':time.time()-t,
                'total_seconds':time.time()-started,'loss':float(loss.detach()),'grad_norm':norm,
                'peak_allocated_bytes':torch.npu.max_memory_allocated(),**positions}
            records.append(record);log('benchmark_step',**record)
            del loss,batch
    (out/'result.json').write_text(json.dumps({'scope':'forward/backward/gradient validation, no optimizer step',
        'optimizer_updates':0,'measurements':records},indent=2)+'\n')
    log('benchmark_complete')

if __name__=='__main__':main()
