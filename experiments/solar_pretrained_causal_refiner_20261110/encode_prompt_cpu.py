"""Encode a shared restoration instruction with the released H3 Qwen conditioner."""
import hashlib
import json
import os
import time
from pathlib import Path

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
ROOT = Path('/opt/refiner-cache/h3-pretrained-causal-refiner-20261109')
BASE = Path('/opt/refiner-cache/models/MiniMax-H3/Ref2VA')
PROMPT = 'Restore this video with sharp, natural details. Preserve the original subjects, identity, colors, lighting, composition, camera motion, and timing.'

def main():
    import torch
    from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds
    torch.set_num_threads(8)
    out = ROOT/'data/restoration_prompt.pt'
    if out.exists():
        raise RuntimeError('Refusing to overwrite a prepared prompt')
    started = time.time()
    print(json.dumps({'event':'text_encoder_loading_cpu','time':started}),flush=True)
    model, info = Qwen3VLForConditionalGeneration.from_pretrained(str(BASE/'text_encoder'), dtype=torch.bfloat16,
        local_files_only=True, output_loading_info=True, attn_implementation='sdpa')
    problems = {k:info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs') if info.get(k)}
    if problems:
        raise RuntimeError(f'Non-strict text encoder load: {problems}')
    model.eval().requires_grad_(False)
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(BASE/'tokenizer'),local_files_only=True)
    processor = Qwen3VLProcessor.from_pretrained(str(BASE/'processor'),local_files_only=True)
    tokens = tokenizer(PROMPT,add_special_tokens=False)['input_ids']
    print(json.dumps({'event':'encoding_cpu','tokens':len(tokens),'time':time.time()}),flush=True)
    with torch.no_grad():
        embeds = get_qwen3vl_prompt_embeds(model,processor,tokens,{},text_encoder_layer=50,
                                         device=torch.device('cpu'),dtype=torch.bfloat16)
    assert embeds.shape == (1,len(tokens),5120) and torch.isfinite(embeds).all()
    torch.save({'prompt':PROMPT,'prompt_embeds':embeds[0].contiguous(),'text_token_tags':torch.ones(len(tokens),dtype=torch.long),
                'encoder_layer':50,'add_special_tokens':False,'presentation':'verbatim_text_only'},out)
    with out.open('rb') as f:
        digest=hashlib.file_digest(f,'sha256').hexdigest()
    (ROOT/'evidence/prompt.json').write_text(json.dumps({'prompt':PROMPT,'path':str(out),'sha256':digest,
        'source_text_encoder':str(BASE/'text_encoder'),'token_count':len(tokens),'layer':50,
        'started_unix':started,'finished_unix':time.time()},indent=2)+'\n')
    print(json.dumps({'event':'prompt_ready','time':time.time(),'sha256':digest}),flush=True)

if __name__ == '__main__':
    main()
