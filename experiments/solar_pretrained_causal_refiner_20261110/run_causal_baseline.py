"""Finite native C5/W6 four-step inference on a paired 1088x1920 sample's first seven latents.

No optimizer is created. The result establishes a pretrained baseline before task adaptation.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch_npu
from safetensors.torch import load_file
from solarwm.backends.minimax_h3.artifacts import H3ArtifactBatch,H3VisualObservation
from solarwm.backends.minimax_h3.flow import make_shifted_schedule
from solarwm.backends.minimax_h3.model import SolarMiniMaxH3Transformer3DModel
from solarwm.backends.minimax_h3.omni_causal import H3OmniCausalCore
from solarwm.backends.minimax_h3.optional import H3_FP32_PARAMETER_NAMES
from ascend_port import install_attention_port,PagedInference

ROOT=Path('/opt/refiner-cache/h3-pretrained-causal-refiner-20261109')
VENDOR=ROOT/'vendor/SolarRenderer-0ca4f5e45b062cc9e9aee506de46aabdc22e4c3d'
SAMPLES={
    'Charge_train_3856_4081_first7.pt':'3fcca24bf47a63cb1340b9ce4ed02262b77f8fe45eedfe37938018134aadd401',
    'Caminandes3_train_168_393_first7.pt':'7448000cfcc7041e323dc21a4e408dd109754038c864cf68a051d30237a5073d',
}

class RefinerCore(H3OmniCausalCore):
    @staticmethod
    def latent_grid(batch):
        # The native layout already accepts general even spatial grids; only its
        # public-data validation is extended here. No latent resampling occurs.
        grid=tuple(int(x) for x in batch.target_latents.shape[-2:])
        if grid!=(68,120):
            raise ValueError(f'This experiment requires native 68x120 latents, got {grid}')
        return grid

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def batch_for(sample,conditioning):
    draft=sample['upscaled'].float()
    assert draft.shape==(24,7,68,120) and torch.isfinite(draft).all()
    prompt=conditioning['prompt_embeds']
    tags=conditioning['text_token_tags']
    return H3ArtifactBatch(sample_id=sample['sample_id'],start_frame=0,
        plan_fingerprint=sample['source_sha256'],
        # Inference reads this field only for shape. Do not pass GT content.
        target_latents=torch.zeros_like(draft),anchor_latents=None,
        prompt_embeds=prompt,text_token_tags=tags,source_frame_indices=torch.arange(22),
        camera_viewmats=None,camera_K=None,source_fps=24.0,
        text_prompt_embeds=prompt,text_prompt_tags=tags,dataset_source='existing_train2_first7',
        visual_observations=(H3VisualObservation(draft[:,:1],frames=None),
                             H3VisualObservation(draft,frames=tuple(range(7)))))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sample',choices=SAMPLES,required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--layout-only',action='store_true')
    args=parser.parse_args()
    started=time.time()
    out=ROOT/'runs'/args.run
    out.mkdir(parents=True,exist_ok=False)
    def log(event,**kw):
        record={'event':event,'time':time.time(),'elapsed_seconds':time.time()-started,
                'optimizer_updates':0,**kw}
        print(json.dumps(record),flush=True)
        with (out/'events.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
    path=ROOT/'data'/args.sample
    assert sha(path)==SAMPLES[args.sample]
    prompt_meta=json.loads((ROOT/'evidence/prompt.json').read_text())
    assert sha(ROOT/'data/restoration_prompt.pt')==prompt_meta['sha256']
    sample=torch.load(path,map_location='cpu',weights_only=True)
    conditioning=torch.load(ROOT/'data/restoration_prompt.pt',map_location='cpu',weights_only=True)
    batch=batch_for(sample,conditioning)
    silence=load_file(str(VENDOR/'assets/h3_silence_153_158_170.safetensors'))['silence_158f']
    torch.set_num_threads(8)
    causal={'mode':'teacher','chunk':5,'window':6,'corrupt':0.0}
    # Check the unchanged native layout before acquiring model/device resources.
    cpu_core=RefinerCore(None,silence,torch.device('cpu'),causal)
    document=cpu_core._document(batch,torch.Generator().manual_seed(20261109))
    rows=document[3]
    assert rows.frames==((0,5),(5,7)),rows.frames
    assert rows.rows_per_frame==2040
    log('layout_checked',chunks=rows.frames,rows_per_frame=rows.rows_per_frame,
        packed_rows=rows.sequence_length,spatial_latent_shape=[68,120])
    del document,cpu_core
    if args.layout_only:
        return
    assert json.loads((ROOT/'evidence/npu_attention_check.json').read_text())['status']=='passed'
    identity=json.loads((ROOT/'evidence/causal_model.json').read_text())
    model_path=Path(identity['model'])
    assert sha(model_path/'diffusion_pytorch_model.safetensors.index.json')==identity['index_sha256']
    log('strict_model_load_start',initialization='Ref2VA + released causal control + released causal 4step',
        physical_npu=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'))
    model=SolarMiniMaxH3Transformer3DModel.strict_from_pretrained(str(model_path),
        torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True).eval().requires_grad_(False)
    for name,p in model.named_parameters():
        assert p.dtype==(torch.float32 if name in H3_FP32_PARAMETER_NAMES else torch.bfloat16),(name,p.dtype)
    log('strict_model_load_complete',parameters=sum(p.numel() for p in model.parameters()))
    torch.npu.set_device('npu:0')
    torch.npu.conv.allow_hf32=False
    torch.npu.matmul.allow_hf32=False
    install_attention_port()
    with torch.no_grad():
        pager=PagedInference(model,'npu:0',log)
        core=RefinerCore(model,silence,torch.device('npu:0'),causal)
        log('pretrained_inference_start',sample=args.sample,chunk_latents=5,window_chunks=6,
            denoise_forwards_per_chunk=4,clean_commit_forwards_per_chunk=1,cache_device='cpu')
        result,_,_,_=core.stream(batch,generator=torch.Generator(device='npu:0').manual_seed(20261109),
            schedule=make_shifted_schedule(5,shift=12.0),exit_index=3,window=6,
            record=None,cache_structure=False)
        torch.npu.synchronize()
        result=result.float().cpu()
        assert result.shape==(1,24,7,68,120) and torch.isfinite(result).all()
        assert pager.calls==10,pager.calls
        torch.save({'refined':result[0],'sample_id':sample['sample_id'],'source_sha256':sample['source_sha256'],
                    'latent_slice':[0,7],'new_optimizer_updates':0},out/'pretrained_refined.pt')
        gt=sample['reference'].float()
        metrics={'draft_latent_l1':(sample['upscaled'].float()-gt).abs().mean().item(),
                 'refined_latent_l1':(result[0]-gt).abs().mean().item(),
                 'npu_peak_allocated_bytes':torch.npu.max_memory_allocated(),
                 'model_forwards':pager.calls,'new_optimizer_updates':0,
                 'scope':'first 7 latents; latent-space only; RGB/video evaluation pending'}
        (out/'result.json').write_text(json.dumps(metrics,indent=2)+'\n')
        (out/'stream_record.json').write_text(json.dumps({'chunk_latent_ranges':[[0,5],[5,7]],
            'model_forwards':pager.calls,'four_step_exit_index':3,'cache_structure':False},indent=2)+'\n')
        log('pretrained_inference_complete',**{k:v for k,v in metrics.items() if k!='new_optimizer_updates'})

if __name__=='__main__':
    main()
