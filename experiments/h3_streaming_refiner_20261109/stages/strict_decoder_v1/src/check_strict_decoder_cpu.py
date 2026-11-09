"""Random small H3 VAE CPU checks; never load pretrained weights or an accelerator."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(stage):
    import torch
    from diffusers import AutoencoderKLMiniMaxH3
    from strict_decoder_core import IncrementalVAECore
    from strict_streaming_decoder import StrictStreamingVideoDecoder
    torch.set_num_threads(2)
    torch.manual_seed(20261109)
    original = stage/'vendor/solarwm_streaming_original.py'
    assert sha(original) == '2aa52d9126b83c5fd772e505007fdaf824ef6c478295d3ec7bb50635855e35b6'
    spec = importlib.util.spec_from_file_location('pinned_solar_decoder', original)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    vae = AutoencoderKLMiniMaxH3(
        latent_channels=24, block_out_channels=(8,8,8,8), layers_per_block=1,
        spatial_downsample_factors=(2,2,2,2), temporal_downsample_factors=(1,2,2,1),
        norm_num_groups=4, decoder_num_layers=2, decoder_num_attention_heads=2,
        decoder_attention_head_dim=16, decoder_num_register_tokens=4,
        decoder_ffn_mult=2, latents_mean=tuple(.05*i-.6 for i in range(24)),
        latents_std=tuple(.5+i/24 for i in range(24))).eval().requires_grad_(False)
    assert all(p.device.type=='cpu' for p in vae.parameters())
    factory = lambda: IncrementalVAECore(vae,compile=False,compensate=False,token_budget=2600)
    model = factory()
    ref = upstream.StreamingDecoder(vae,compile=False,compensate=False,token_budget=2600)
    z = torch.randn(1,24,137,2,3)
    outputs=[]; kv_bytes=[]; slot_counts=[]; retained=[]; max_parity=0.0; projected=[]
    hook=vae.decoder.transformer_blocks[0].attn.to_k.register_forward_pre_hook(
        lambda m,args: projected.append(args[0].shape[1]))
    # Run the candidate first so its projection trace excludes the reference.
    for i in range(137):
        out=model.push(z[:,:,i:i+1]);outputs.append(out.clone())
        assert out.shape==(1,3,1 if i%5==0 else 4,32,48)
        kv_bytes.append(sum(t.numel()*t.element_size() for t in model._k+model._v))
        slot_counts.append(model.slot_passes)
        retained.append(len(model._latents))
    hook.remove()
    assert len(set(kv_bytes))==1 and max(retained)<=7
    assert all(t.device.type=='cpu' and not t.requires_grad for t in model._k+model._v)
    assert len(projected)==137
    fresh=[slot_counts[0]]+[b-a for a,b in zip(slot_counts,slot_counts[1:])]
    assert projected==[n*6+5 for n in fresh]
    assert max(fresh)==3 and model.slot_passes==190
    for i in range(137):
        other=ref.push(z[:,:,i:i+1])
        max_parity=max(max_parity,float((other-outputs[i]).abs().max()))
    assert max_parity==0
    # Reset and change every unseen latent after index66. Prefixes cannot change.
    model.reset(); changed=z.clone();changed[:,:,67:]*=-4
    future_error=0.0
    for i in range(137):
        other=model.push(changed[:,:,i:i+1])
        if i<67:future_error=max(future_error,float((other-outputs[i]).abs().max()))
    assert future_error==0
    model.reset(); offset=0; chunk_error=0.0
    for size in [1,6,60,1,66,3]:
        for other in model.stream(z[:,:,offset:offset+size]):
            chunk_error=max(chunk_error,float((other-outputs[offset]).abs().max()));offset+=1
    assert offset==137 and chunk_error==0
    assert sum(v.shape[2] for v in outputs[:67])==226
    assert sum(v.shape[2] for v in outputs)==464
    # Independent released tile splitter, full output geometry and project normalization/crop.
    ys,yl,yo=vae._split_tiles(1088,256,64);xs,xl,xo=vae._split_tiles(1920,256,64)
    assert len(ys)*len(xs)==60 and set(yl+xl)=={256}
    full=StrictStreamingVideoDecoder(vae,token_budget=2600)
    full_z=torch.randn(1,24,1,68,120)
    actual=full.push(full_z,latent_index=0)
    assert actual.shape==(1,1080,1920,3) and actual.dtype==torch.uint8
    # Check crop/denormalization against an independent upstream stream of the same random VAE.
    full_ref=upstream.StreamingDecoder(vae,compile=False,compensate=False,token_budget=2600)
    raw=full_z*torch.tensor(vae.config.latents_std).view(1,24,1,1,1)+torch.tensor(vae.config.latents_mean).view(1,24,1,1,1)
    px=full_ref.push(raw)
    pm=torch.tensor([.485,.456,.406]).view(1,3,1,1,1)
    ps=torch.tensor([.229,.224,.225]).view(1,3,1,1,1)
    expected=((px*ps+pm).clamp(0,1)[0].permute(1,2,3,0)*255).round().byte()[:,4:1084]
    assert torch.equal(actual,expected)
    stats=full.cache_stats()
    production_kv=2*36*60*(5+7*256)*32*64*4
    assert production_kv==63594823680
    for bad,index in [(full_z,0),(torch.zeros(1,24,2,68,120),1),
                      (torch.zeros(1,24,1,67,120),1),(full_z*float('nan'),1)]:
        try:full.push(bad,latent_index=index)
        except ValueError:pass
        else:raise AssertionError('Invalid input admitted')
    assert full.core.received==1
    full.reset();replay=full.push(full_z,latent_index=0);assert torch.equal(actual,replay)
    return {
        'status':'passed','scope':'random tiny actual H3 VAE decoder on CPU, not pretrained full VAE',
        'decoder_layers':2,'decoder_heads':2,'head_dim':16,'latent_positions':137,
        'long_spatial_shape':[2,3],'pinned_upstream_incremental_max_abs':max_parity,
        'future_prefix_max_abs':future_error,'chunk_reset_max_abs':chunk_error,
        'frames_for67':226,'frames_for137':464,'bounded_small_KV_bytes':kv_bytes[-1],
        'max_retained_input_latents':max(retained),'max_slots_per_push':max(fresh),
        'actual_slot_forwards':model.slot_passes,'bounded_past_slot_refreshes':model.slot_passes-137,
        'full_prefix_reprojection':False,'small_decoder_full_geometry_first_latent':stats,
        'full_geometry_RGB_shape':list(actual.shape),'full_geometry_crop_RGB_exact':True,
        'invalid_inputs_rejected':True,'reset_replay_exact':True,
        'full36layer_FP32_KV_bytes_estimated':production_kv,
        'cache_backing':'CPU; accelerator transfers not tested',
        'full_pretrained_VAE_validated':False,'accelerator_validated':False,
        'strict_decoder_accepted':False,'continuous_visual_reviewed':False,
        'optimizer_updates':0,'accelerator_forwards':0,
        'implementation_sha256':{f:sha(stage/'src'/f) for f in ['strict_decoder_core.py','strict_streaming_decoder.py','check_strict_decoder_cpu.py']},
        'diffusers_VAE_sha256':sha(inspect.getfile(AutoencoderKLMiniMaxH3)),
        'upstream_sha256':sha(original),
    }


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();stage=Path(a.stage);out=Path(a.output);assert not out.exists()
    sys.path.insert(0,str(stage/'src'));start=time.time()
    try:result=run(stage)
    except BaseException as e:
        out.with_suffix('.failure.json').write_text(json.dumps({'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()},indent=2)+'\n')
        raise
    result['seconds']=time.time()-start
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
