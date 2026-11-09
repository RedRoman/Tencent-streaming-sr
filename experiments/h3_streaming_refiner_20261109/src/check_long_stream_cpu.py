"""CPU-only long-index checks; not a full33B or streaming-VAE acceptance test."""
import argparse, hashlib, json, sys, time
from pathlib import Path
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); stage = Path(a.stage); out = Path(a.output)
    assert not out.exists()
    pins = {
        'causal_attention.py': 'f21ba96716d5cf91603e67e6d4d4d603aade47c054f81bd479ab99b341b6751f',
        'paged_runtime.py': '96c18f6d5bbecd7cd1341c0beef34c80c94f474b1bb86638a1e7af2125a7357f',
    }
    for name, digest in pins.items():
        assert sha(stage / 'src' / name) == digest
    sys.path.insert(0, str(stage / 'src'))
    import torch
    from torch import nn
    from causal_attention import CausalAttention, AttentionCall, append_cache, position_noise
    from paged_runtime import Runtime
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep
    torch.set_num_threads(2); torch.manual_seed(20261109)
    started = time.time(); frames = 137; rows = 5; static_rows = 7
    ids = (-1,) * static_rows + tuple(f for f in range(frames) for _ in range(rows))
    attn = SimpleNamespace(heads=2, to_q=nn.Linear(12,16,bias=False),
        to_k=nn.Linear(12,16,bias=False), to_v=nn.Linear(12,16,bias=False),
        norm_q=nn.RMSNorm(8), norm_k=nn.RMSNorm(8), to_out=[nn.Linear(16,12,bias=False),nn.Identity()])
    x = torch.randn(1,len(ids),12,device='cpu')
    angles = torch.outer(torch.arange(len(ids),dtype=torch.float32),
                         1 / (10000 ** (torch.arange(4,dtype=torch.float32)/4)))
    angles = torch.cat((angles,angles),-1); rope = (angles.cos(),angles.sin())
    proc = CausalAttention(chunk_size=3)
    with torch.no_grad():
        proc.call = AttentionCall(ids)
        reference = proc(attn,x,rotary_emb=rope)
        changed = x.clone(); cutoff = static_rows + 89 * rows; changed[:,cutoff:] *= -7
        changed_output = proc(attn,changed,rotary_emb=rope)
        future_error = float((reference[:,:cutoff]-changed_output[:,:cutoff]).abs().max())
        assert future_error == 0
        projected = []
        handle = attn.to_k.register_forward_pre_hook(lambda m,args: projected.append(args[0].shape[1]))
        cache = None; values = []; trace = []
        windows = [(0,static_rows,-1)] + [(static_rows+f*rows,static_rows+(f+1)*rows,f) for f in range(frames)]
        for lo,hi,f in windows:
            proc.call = AttentionCall((f,)*(hi-lo),past=cache); proc.capture=True
            values.append(proc(attn,x[:,lo:hi],rotary_emb=tuple(r[lo:hi] for r in rope)))
            cache = append_cache(cache,proc.captured,next_frame=f+1)
            assert cache.k.device.type == cache.v.device.type == 'cpu'
            assert not cache.k.requires_grad and not cache.v.requires_grad
            expected = {-1} if f<0 else {-1,0,f}
            assert set(cache.frames) == expected
            trace.append(cache.k.numel()*cache.k.element_size()+cache.v.numel()*cache.v.element_size())
        handle.remove()
        assert projected == [static_rows] + [rows]*frames, 'A previous prefix was reprojected'
        assert len(set(trace[2:])) == 1, 'Cache grows after two frames'
        error = float((torch.cat(values,1)-reference).abs().max()); assert error < 2e-6
        proc.call = AttentionCall((frames-1,)*rows,past=cache)
        try:
            proc(attn,x[:,-rows:],rotary_emb=tuple(r[-rows:] for r in rope))
        except ValueError as e:
            assert 'current or future' in str(e)
        else:
            raise AssertionError('Duplicate final frame was accepted')

    # Compare individual absolute-frame position packing to the released full
    # sequence layout, including frame67 and indices beyond two 67-frame clips.
    position_errors = {}; temporal_positions = {}
    for text_rows in (0,9):
        rt = Runtime.__new__(Runtime); rt.device=torch.device('cpu'); rt.text_count=text_rows
        # The OpenH3 wrapper rejects zero text. Call the released lower-level
        # packer directly so no dummy-token +1/-1 rounding enters the oracle.
        layout=MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags=torch.ones(text_rows,dtype=torch.long),num_latent_frames=frames,
            latent_height=68,latent_width=120,num_audio_latents=1,patch_size=(1,2,2),
            audio_channels=2,audio_tag=2,video_tag=0,keyframe_anchors=())
        expected=layout[0][layout[2]]
        indices=[0,1,4,5,6,65,66,67,68,133,134,135,136]
        got=rt.video_positions(indices)
        selected=torch.cat([expected[f*2040:(f+1)*2040] for f in indices])
        err=float((got-selected).abs().max());assert err==0
        position_errors[str(text_rows)]=err
        temporal_positions[str(text_rows)]={str(f):float(got[i*2040,0]) for i,f in enumerate(indices)}

    indices=[0,66,67,68,133,134,136];noise_hashes={}
    for f in indices:
        first=position_noise((24,1,68,120),'long-cpu',f,20261109)
        noise_hashes[str(f)]=hashlib.sha256(first.numpy().tobytes()).hexdigest()
    # Different arrival order/chunk boundaries cannot advance a global RNG.
    for group in ([133,134,136],[67,68],[0,66]):
        for f in group:
            other=position_noise((24,1,68,120),'long-cpu',f,20261109)
            assert hashlib.sha256(other.numpy().tobytes()).hexdigest()==noise_hashes[str(f)]
    assert len(set(noise_hashes.values()))==len(indices)
    result={'status':'passed','scope':'137-frame synthetic attention with5 spatial rows; full68x120 released position-layout and noise boundary checks; CPU only',
        'frames':frames,'kernel_offline_incremental_max_abs':error,'kernel_future_max_abs':future_error,
        'cache_bytes_after_two_frames':trace[2],'cache_bytes_final':trace[-1],
        'current_only_K_projection_rows':sum(projected),'full_prefix_reprojection_observed':False,
        'duplicate_final_frame_rejected':True,'position_errors_by_text_rows':position_errors,
        'position_oracle':'released build_packed_sequence, direct zero-text and9text layouts; exact comparison, no tolerance relaxation',
        'absolute_temporal_positions':temporal_positions,'noise_sha256_by_absolute_frame':noise_hashes,
        'noise_arrival_order_exact':True,'full33B_longstream_validated':False,
        'full_spatial_attention_longstream_validated':False,'strict_decoder_validated':False,
        'optimizer_updates':0,'new_accelerator_forwards':0,'source_sha256':sha(__file__),
        'runtime_source_sha256':pins,'seconds':time.time()-started}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
