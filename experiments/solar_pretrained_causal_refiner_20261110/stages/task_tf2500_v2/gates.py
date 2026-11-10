"""Check native TF visibility and paged backward against a resident real H3 block."""
import torch
from solarwm.backends.minimax_h3.model import get_parameter_dtype,H3AttentionControl
from solarwm.backends.minimax_h3.omni_causal import H3SegmentPlan, training_plan

def check_plan(core, batch):
    doc = core._document(batch, torch.Generator(device=core.device).manual_seed(17))
    rows = doc[3]
    plan, clean_span = training_plan(rows, window=6, teacher=True)
    target_width = sum(b-a for a,b in rows.target)
    clean = [(a+target_width,b+target_width) for a,b in rows.target]
    def overlaps(spans, span):
        return any(max(a,span[0]) < min(b,span[1]) for a,b in spans)
    for c, span in enumerate(rows.target):
        segment = next(s for s in plan.segments if s[:2] == span)
        keys = segment[2]
        assert overlaps(keys, span)
        assert not any(overlaps(keys, clean[j]) for j in range(c,len(clean)))
        assert not any(overlaps(keys, rows.target[j]) for j in range(c+1,len(clean)))
        assert not any(overlaps(keys, rows.structure[j]) for j in range(c+1,len(clean)))
        assert all(overlaps(keys, clean[j]) for j in range(max(0,c-5),c))
    return {'chunks': list(rows.frames), 'training_rows': plan.sequence_length,
            'clean_copy_rows': clean_span[1]-clean_span[0], 'future_or_current_GT_leak': False}

def check_backward(rt):
    model, device = rt.model, rt.device
    torch.manual_seed(20261110)
    initial = rt.adapter_state()
    # Nonzero B makes this gate exercise A gradients too; restore before training.
    with torch.no_grad():
        for a in rt.adapters[:4]:
            a.b.normal_(0,.001)
        times = torch.tensor([.25, .999],device=device,dtype=torch.float32)
        temb = model._time_condition(times,None,parameter_dtype=get_parameter_dtype)
        pos = torch.zeros((16,3),device=device,dtype=torch.float32)
        pos[:,0] = torch.arange(16,device=device)/10
        rope = model.rope(pos)
    adaln = torch.zeros(16,device=device,dtype=torch.long)
    plan = H3SegmentPlan(((0,8,((0,8),),(),None),(8,16,((0,16),),(),None)),16)
    args=(temb,adaln,rope,H3AttentionControl(attention_mask=plan,fused_prope=False,
        prope_token_indices=None,prope_frame_ids=None,cam_viewmats=None,cam_K=None,prope_kwargs=None))
    x=torch.randn(1,16,model.config.hidden_size,device=device,dtype=torch.bfloat16)
    upstream_grad=torch.randn_like(x)/100
    params=rt.layer_params[0]
    frozen=[p.detach().cpu().clone() for p in (rt.model.transformer_blocks[0].attn.to_q.base.weight,)]
    rt.load(0)
    direct_x=x.detach().requires_grad_(True)
    direct=rt.original[0](direct_x,*args)
    expected=torch.autograd.grad(direct,(direct_x,*params),upstream_grad)
    expected=[v.detach().cpu() for v in expected]
    direct_cpu=direct.detach().cpu()
    del direct,direct_x
    rt.release(0)
    replay_x=x.detach().requires_grad_(True)
    replay=rt.model.transformer_blocks[0](replay_x,*args)
    actual=torch.autograd.grad(replay,(replay_x,*params),upstream_grad)
    max_abs=[float((a.float().cpu()-b.float()).abs().max()) for a,b in zip(actual,expected)]
    forward_error=float((replay.detach().cpu()-direct_cpu).float().abs().max())
    assert forward_error==0 and max(max_abs)==0,(forward_error,max_abs)
    assert all(torch.isfinite(v).all() for v in actual)
    assert all(float(v.float().abs().sum())>0 for v in actual)
    assert torch.equal(frozen[0],rt.model.transformer_blocks[0].attn.to_q.base.weight.cpu())
    rt.load_adapter(initial)
    assert all(torch.count_nonzero(a.b)==0 for a in rt.adapters)
    rt.enable_adapters(False)
    with torch.no_grad(): base=rt.model.transformer_blocks[0](x,*args)
    rt.enable_adapters(True)
    with torch.no_grad(): zero=rt.model.transformer_blocks[0](x,*args)
    identity_error=float((zero-base).abs().max())
    assert identity_error==0
    return {'status':'passed','real_block_forward_max_abs':forward_error,
            'real_block_gradient_max_abs':max_abs,'zero_adapter_identity_max_abs':identity_error,
            'frozen_weight_unchanged':True,'optimizer_updates':0}
