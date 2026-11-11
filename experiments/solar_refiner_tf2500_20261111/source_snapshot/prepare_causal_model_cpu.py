"""Strict-load Ref2VA and merge both released causal adapters, using upstream arithmetic."""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path('/opt/refiner-cache/h3-pretrained-causal-refiner-20261109')

def main():
    import torch
    from safetensors.torch import load_file
    from solarwm.backends.minimax_h3.model import SolarMiniMaxH3Transformer3DModel
    from solarwm.backends.minimax_h3.lora import discover_h3_lora_targets, merge_lora_delta
    from solarwm.backends.minimax_h3.optional import H3_FP32_PARAMETER_NAMES
    torch.set_num_threads(8)
    def log(event,**kw):
        print(json.dumps({'event':event,'time':time.time(),**kw}),flush=True)
    assets=json.loads((ROOT/'evidence/public_assets.json').read_text())
    assert assets['status']=='verified'
    output=ROOT/'models/causal_4step/transformer'
    if output.exists():
        raise RuntimeError('Refusing to overwrite existing merged weights')
    log('strict_load_start')
    model=SolarMiniMaxH3Transformer3DModel.strict_from_pretrained(str(ROOT/'models/ref2va/transformer'),
        torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True).eval().requires_grad_(False)
    for name,p in model.named_parameters():
        assert p.dtype==(torch.float32 if name in H3_FP32_PARAMETER_NAMES else torch.bfloat16),(name,p.dtype)
    targets=discover_h3_lora_targets(model)
    log('strict_load_complete',targets=len(targets),parameters=sum(p.numel() for p in model.parameters()))
    merged=[]
    for filename in ('solarrenderer_causal_control_lora.safetensors','solarrenderer_causal_4step_lora.safetensors'):
        path=ROOT/'weights'/filename
        expected=next(x['sha256'] for x in assets['weights'] if Path(x['file']).name==filename)
        with path.open('rb') as f:
            assert hashlib.file_digest(f,'sha256').hexdigest()==expected
        state=load_file(str(path),device='cpu')
        assert set(state)=={f'{name}.lora_{ab}.weight' for name in targets for ab in ('A','B')}
        for i,name in enumerate(targets):
            pair={f'{name}.lora_{ab}.weight':state[f'{name}.lora_{ab}.weight'] for ab in ('A','B')}
            assert merge_lora_delta(model,pair,alpha=384,rank=384)==1
            if (i+1)%26==0:
                log('merge_progress',adapter=filename,modules=i+1,total=len(targets))
        merged.append({'name':filename,'sha256':expected,'rank':384,'alpha':384,'scale':1.0})
        del state
    log('save_start')
    model.save_pretrained(str(output),safe_serialization=True,max_shard_size='5GB')
    index=output/'diffusion_pytorch_model.safetensors.index.json'
    with index.open('rb') as f:
        index_sha=hashlib.file_digest(f,'sha256').hexdigest()
    record={'base_conversion':str(ROOT/'evidence/base_conversion.json'),'adapters_in_order':merged,
            'model':str(output),'index_sha256':index_sha,'target_count':len(targets),
            'merge_implementation':'upstream merge_lora_delta; FP32 delta, sequential BF16 rounding',
            'new_optimizer_updates':0,'completed_unix':time.time()}
    (ROOT/'evidence/causal_model.json').write_text(json.dumps(record,indent=2)+'\n')
    log('causal_model_ready',path=str(output))

if __name__=='__main__':
    main()
