"""Tiny real architecture: dense worker vs upstream model.forward, full spatial.

No new trained adapter is enabled. This checks row-time/modality orchestration,
not full 33B NPU equivalence or production prompt quality.
"""
import json,torch
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3AttnProcessor
from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
from open_h3.geometry import pack_audio_latents,unpatchify_video_rows,MiniMaxH3Geometry
from evaluate_dense_reference import dense_velocity,check_upstream
from paged_runtime import CanonicalRows
# Reuse the already-tested tiny mixed-precision model constructor; its checks
# also ensure the condition branch has gradients but has not been updated.
from test_runtime_cpu import r,sample,x,draft

# The reference worker uses canonical_rows=False. Remove only those wrappers.
def unwrap(parent):
    for key,child in list(parent.named_children()):
        if isinstance(child,CanonicalRows):setattr(parent,key,child.module)
        else:unwrap(child)
unwrap(r.model)
r.resident=1
r.load_layer(0)
sigma=.7
with torch.no_grad():
    actual=dense_velocity(r,x,draft,sample,sigma)
    static,sp,tags=r.static_inputs(sample);ns=static.shape[1];nv=2040*x.shape[1]
    positions=torch.cat((sp,r.video_positions(list(range(x.shape[1])))))
    modalities=torch.cat((tags,torch.zeros(nv,dtype=torch.long)))
    times=torch.cat((torch.zeros(ns,dtype=torch.long),torch.ones(nv,dtype=torch.long)))
    rows=patchify_video_latents(x[None],r.model.config.patch_size).reshape(1,nv,96).bfloat16()
    r.model.transformer_blocks[0].attn.set_processor(MiniMaxH3AttnProcessor())
    raw=r.model(hidden_states=rows,audio_hidden_states=pack_audio_latents(sample['audio_latents'][None]),
        encoder_hidden_states=sample['prompt_embeds'][None],timestep=torch.tensor([1.,1.-sigma]),
        timestep_indices=times,token_tags=modalities,position_ids=positions,
        video_indices=torch.arange(ns,ns+nv),audio_indices=torch.arange(ns),text_indices=torch.empty(0,dtype=torch.long),return_dict=False)[0]
    expected=unpatchify_video_rows(raw,MiniMaxH3Geometry(x.shape[1],68,120,1))[0].float()
    d=actual-expected;maxabs=float(d.abs().max());rel=float(d.square().mean().sqrt()/expected.square().mean().sqrt())
    assert maxabs<=.03 and rel<=.005,(maxabs,rel)
print(json.dumps({'status':'passed','scope':'tiny H3 dense orchestration vs upstream model.forward; full68x120 two latents CPU',
                  'max_abs':maxabs,'relative_rms':rel,'optimizer_updates':0,'full33B_NPU_dense_parity':False}))
r.model.transformer_blocks[0].attn.set_processor(r.processors[0])
hooked=check_upstream(r,x,draft,sample,sigma)
assert hooked['max_abs']==0
print(json.dumps({'status':'passed','scope':'CPU check of paged upstream comparison hooks','max_abs':hooked['max_abs']}))
