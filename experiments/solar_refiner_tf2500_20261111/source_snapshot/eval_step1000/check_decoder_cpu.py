"""Real pretrained CPU reference check; no device or training modifications."""
from pathlib import Path
import json,time
import torch
from render_full import load_decoder,write,sha,HERE,ROOT
from causal_vit_decoder import CausalViTDecoder as Reference

torch.set_num_threads(8)
deadline=time.time()+7200
while not (ROOT/'weights/streamrender_vit24/receipt.json').exists():
    if time.time()>deadline:raise TimeoutError('Decoder weights not ready in two hours')
    time.sleep(30)
start=time.time();model=load_decoder('cpu')
print('Strict pretrained load succeeded',flush=True)
torch.manual_seed(1000);z=torch.randn(1,6,24,3,5)*.5
ref=model.init_state();port=model.init_state();max_error=0;frame_count=0
with torch.inference_mode():
    for i in range(0,6,2):
        a=Reference.decode_chunk(model,z[:,i:i+2],ref)
        b=model.decode_chunk(z[:,i:i+2],port)
        assert torch.equal(a,b)
        frame_count+=a.shape[2]
        assert all(x.shape[1]<=4 for x in port['k']+port['v'])
        print(json.dumps({'pair':i//2,'frames':a.shape[2],'exact':True,'seconds':time.time()-start}),flush=True)
result=dict(status='passed',strict_pretrained_load=True,layers=len(model.blocks),
    parameters=sum(p.numel() for p in model.parameters()),source_sha256=sha(Path(__file__)),
    cpu_reference_exact=True,checked_latents=6,checked_frames=frame_count,grid=[3,5],
    cache_bounded_to_four=True,npu_test_pending=True,elapsed_seconds=time.time()-start)
write(HERE/'decoder_cpu_check.json',result);print(json.dumps(result),flush=True)
