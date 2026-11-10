"""Full-spatial tiny real H3 controls; never evidence of trained 33B quality."""
import io
import json
import sys
from types import SimpleNamespace
import torch
from torch import nn
from diffusers import MiniMaxH3Transformer3DModel
from paged_runtime import Runtime, AdapterLinear, canonicalize
from causal_attention import CausalAttention
from anchored_refiner_v1 import ResidualHead, AnchoredStreamingRefiner, align_input_projection, schedule, training_input


def tiny_runtime():
    r = Runtime.__new__(Runtime)
    r.device, r.resident = torch.device('cpu'), 0
    r.model = MiniMaxH3Transformer3DModel(num_attention_heads=1, attention_head_dim=128,
        hidden_size=32, num_layers=1, num_refiner_layers=1, ffn_dim=64,
        time_embed_hidden_dim=32, time_embed_dim=16).float().requires_grad_(False)
    for name, child in r.model.named_children():
        if name not in ('proj_in','audio_proj_in','time_proj','time_embedder','proj_out','audio_proj_out','rope'):
            child.bfloat16()
    r.host, r.processors, r.layer_params = {}, [], []
    for i, block in enumerate(r.model.transformer_blocks):
        for name in ('attn.to_q','attn.to_k','attn.to_v','attn.to_out.0'):
            parent, key = name.rsplit('.', 1)
            setattr(block.get_submodule(parent), key, AdapterLinear(block.get_submodule(name), 2))
        proc = CausalAttention()
        block.attn.set_processor(proc)
        r.processors.append(proc)
        canonicalize(block, proc)
        r.layer_params.append(tuple(p for p in block.parameters() if p.requires_grad))
        r.host[i] = [(m,k,p.detach()) for m in block.modules() for k,p in m._parameters.items()
                     if p is not None and not p.requires_grad]
        r.host[i] += [(m,k,b.detach()) for m in block.modules() for k,b in m._buffers.items() if b is not None]
    r.curves = SimpleNamespace(coords=None)
    def pre(m, args, kwargs):
        r.curves.coords = torch.zeros(len(kwargs['timestep']), 8)
    r.curves.pre = pre
    r.condition = nn.Linear(96, 32, bias=False)
    nn.init.zeros_(r.condition.weight)
    r.params = [p for g in r.layer_params for p in g]+list(r.condition.parameters())
    r.cache = None
    align_input_projection(r)
    return r


def main():
    torch.set_num_threads(2)
    torch.manual_seed(51)
    r = tiny_runtime()
    head = ResidualHead()
    cond = {'audio_latents': torch.randn(2,32,2), 'prompt_embeds': torch.empty(0,5120)}
    draft = torch.randn(1,24,3,68,120)
    a = AnchoredStreamingRefiner(r, head, cond, sample_id='cpu')
    full = a.push(draft)
    assert torch.equal(full, draft)
    assert a.audit()['max_cached_video_frames'] == 2
    b = AnchoredStreamingRefiner(r, head, cond, sample_id='cpu')
    chunk = torch.cat([b.push(draft[:, :, :1]), b.push(draft[:, :, 1:])], 2)
    assert torch.equal(full, chunk)
    # Nonzero head exposes the actual causal model path; zero-head identity alone
    # must not be mistaken for validation of its causal feature computation.
    with torch.no_grad():
        head.out_proj.weight.normal_(0, .005)
    a = AnchoredStreamingRefiner(r, head, cond, sample_id='cpu')
    first = a.push(draft[:, :, :1])
    other = draft.clone()
    other[:, :, 1:] *= -3
    b = AnchoredStreamingRefiner(r, head, cond, sample_id='cpu')
    changed = b.push(other)
    assert torch.equal(first, changed[:, :, :1])
    assert (first-draft[:, :, :1]).abs().max() <= .250001
    x = .2*draft[0,:,:1]+.8*torch.randn_like(draft[0,:,:1])
    velocity = r.offline(x, draft[0,:,:1], cond, [0], .8, train=True)
    target = draft[0,:,:1]+.1
    pred = head(x, velocity, draft[0,:,:1], .8)
    (pred-target).square().mean().backward()
    upstream = sum(float(p.grad.float().square().sum()) for p in r.params if p.grad is not None)**.5
    head_grad = sum(float(p.grad.float().square().sum()) for p in head.parameters() if p.grad is not None)**.5
    assert upstream > 0 and head_grad > 0
    buffer = io.BytesIO()
    torch.save({'adapter':r.adapter_state(),'head':head.state_dict()}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    r.load_adapter(saved['adapter'])
    head.load_state_dict(saved['head'])
    with torch.no_grad():
        reloaded = head(x, r.offline(x,draft[0,:,:1],cond,[0],.8), draft[0,:,:1],.8)
    assert torch.equal(pred.detach(), reloaded)
    # Schedule covers both clips and all67 positions, with25% start frames.
    specs = [schedule(i) for i in range(2000)]
    assert sum(s['latent_index']==0 for s in specs) == 500
    assert all(set(s['frames']) <= set(range(s['latent_index']+1)) for s in specs)
    assert all(set(s['latent_index'] for s in specs if s['sample_index']==k)==set(range(67)) for k in [0,1])
    sample = {'sample_id':'cpu','upscaled':draft[0].repeat(1,23,1,1)[:,:67],
              'reference':draft[0].repeat(1,23,1,1)[:,:67]+.1}
    spec = schedule(6)
    x0, d0, target0 = training_input(sample,spec)
    changed_sample = {**sample,'reference':sample['reference'].clone()}
    changed_sample['reference'][:,:] *= -100
    x1,d1,_ = training_input(changed_sample,spec)
    assert torch.equal(x0,x1) and torch.equal(d0,d1)
    accelerator_initialized = bool(hasattr(torch,'npu') and torch.npu.is_initialized())
    all_cpu = all(p.device.type=='cpu' for p in list(r.model.parameters())+list(head.parameters())+list(r.condition.parameters()))
    assert not accelerator_initialized and all_cpu
    print(json.dumps({'status':'passed','scope':'tiny random one-layer realH3, full68x120; CPU only',
          'initial_identity_max_abs':0,'three_latent_chunk_max_abs':0,'nonzero_head_causal_prefix_max_abs':0,
          'nonzero_upstream_gradient_norm':upstream,'nonzero_head_gradient_norm':head_grad,'reload_max_abs':0,
          'target_does_not_enter_forward_inputs':True,'all67_positions_each_train_clip':True,
          'first_latent_samples_in2000':500,'new_real_optimizer_updates':0,
          'accelerator_imported':'torch_npu' in sys.modules,
          'accelerator_initialized':accelerator_initialized,'all_model_tensors_CPU':all_cpu}),flush=True)


if __name__ == '__main__':
    main()
