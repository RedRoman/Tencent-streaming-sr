"""Pinned step1000, all 67 latents, upstream C5/W6 self-history inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path('/opt/refiner-cache/h3-pretrained-causal-refiner-20261109')
STAGE = ROOT/'stages/task_tf2500_v2'
sys.path.insert(0, str(STAGE))
import torch
import torch_npu
from safetensors.torch import load_file
from common import sha, write_json, VENDOR
from patch_core import PatchCore, batch_for
from runtime import TrainingRuntime
from ascend_port import install_attention_port
from solarwm.backends.minimax_h3.model import SolarMiniMaxH3Transformer3DModel
from solarwm.backends.minimax_h3.optional import H3_FP32_PARAMETER_NAMES
from solarwm.backends.minimax_h3.flow import make_shifted_schedule

CHECKPOINT_SHA = 'dd4b062dad6c0c9367909bbf577bc2ea98210211c3d32de400091e6f809f224c'
CONFIG_SHA = 'a652cfef6cbf68b3377c287f4f5c65858706f63812829090a07342b039c6fd45'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    (out/'chunks').mkdir()
    started = time.time(); forwards = 0
    def log(event, **kw):
        row = dict(event=event, time=time.time(), elapsed_seconds=time.time()-started,
                   checkpoint_step=1000, optimizer_updates=0, **kw)
        print(json.dumps(row), flush=True)
        with (out/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    def status(state, **kw):
        write_json(out/'status.json', dict(status=state, updated_unix=time.time(), **kw))
    try:
        status('verifying_inputs')
        config_path = STAGE/'config.json'
        assert sha(config_path) == CONFIG_SHA
        config = json.loads(config_path.read_text())
        for name, digest in config['sources'].items(): assert sha(ROOT/name) == digest, name
        checkpoint = ROOT/'runs/task_tf2500_v2/checkpoint_1000.pt'
        assert sha(checkpoint) == CHECKPOINT_SHA
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert payload['step'] == 1000 and payload['config_sha256'] == CONFIG_SHA
        sample_path = ROOT/'data/Charge_train_3856_4081.pt'
        assert sha(sample_path) == config['samples'][0]['sha256']
        sample = torch.load(sample_path, map_location='cpu', weights_only=True)
        assert sample['upscaled'].shape == sample['reference'].shape == (24,67,68,120)
        cond_path = ROOT/'data/restoration_prompt.pt'
        assert sha(cond_path) == json.loads((ROOT/'evidence/prompt.json').read_text())['sha256']
        condition = torch.load(cond_path, map_location='cpu', weights_only=True)
        batch = batch_for(sample, condition, 0, 67, inference=True)
        assert torch.count_nonzero(batch.target_latents) == 0
        assert len(batch.source_frame_indices) == 226
        silence = load_file(str(VENDOR/'assets/h3_silence_153_158_170.safetensors'))['silence_158f']
        # The released silence artifact covers only the 47-latent training clip.
        # Upstream stream() indexes the first training-length rows of each channel
        # in a full-length document. Pad each channel's UNUSED suffix so right-
        # channel offsets remain valid; never extend the audio that attention sees.
        original_silence = silence.clone()
        needed = PatchCore.audio_latents(67)
        heard = PatchCore.audio_latents(47)
        assert silence.shape == (2,32,heard)
        silence = torch.cat((silence, silence[...,-1:].expand(2,32,needed-heard)), -1)
        cpu_core = PatchCore(None, silence, torch.device('cpu'), config['causal'])
        doc = cpu_core._document(batch, torch.Generator().manual_seed(20261109))
        shifts, kept = cpu_core._frozen_positions(doc[1],doc[3],67)
        indices = torch.from_numpy(kept-doc[3].audio[0])
        packed = cpu_core.audio_rows(67)
        expected = original_silence.permute(0,2,1).reshape(1,-1,32)
        assert packed.shape == (1,2*needed,32)
        assert torch.equal(packed.index_select(1,indices),expected)
        assert indices.min()>=0 and indices.max()<packed.shape[1]
        log('full_audio_layout_verified', document_rows=2*needed, attended_rows=2*heard,
            preserved_both_channels=True, padded_rows_attended=0, chunks=doc[3].frames)
        del doc,cpu_core,packed,expected,original_silence
        torch.set_num_threads(8); torch.manual_seed(20261109)
        torch.npu.set_device('npu:0')
        torch.npu.conv.allow_hf32=False; torch.npu.matmul.allow_hf32=False
        identity = json.loads((ROOT/'evidence/causal_model.json').read_text())
        model_path = Path(identity['model'])
        assert sha(model_path/'diffusion_pytorch_model.safetensors.index.json') == identity['index_sha256']
        status('loading_model'); log('strict_model_load_start')
        model = SolarMiniMaxH3Transformer3DModel.strict_from_pretrained(str(model_path),
            torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True).eval().requires_grad_(False)
        for name,p in model.named_parameters():
            assert p.dtype == (torch.float32 if name in H3_FP32_PARAMETER_NAMES else torch.bfloat16)
        install_attention_port()
        rt = TrainingRuntime(model, 'npu:0', rank=16, resident=0, log=log)
        assert payload['parameter_names'] == [n for n,p in rt.ordered]
        rt.load_adapter(payload['adapter'])
        for name,p in rt.ordered:
            assert torch.equal(p.detach().cpu(), payload['adapter'][name])
            p.requires_grad_(False)
        del payload
        class LoggedCore(PatchCore):
            def _run(self, **kw):
                nonlocal forwards
                forwards += 1
                commits = [s[-1] for s in kw['plan'].segments if s[-1] is not None]
                log('forward_start', forward=forwards, rows=kw['plan'].sequence_length, commits=commits)
                result = super()._run(**kw)
                torch.npu.synchronize()
                log('forward_complete', forward=forwards, commits=commits)
                if commits:
                    c = commits[-1][1]
                    status('streaming', committed_chunks=c+1, total_chunks=14,
                           committed_latents=min(5*(c+1),67), total_latents=67)
                return result
        core = LoggedCore(model, silence, torch.device('npu:0'), config['causal'])
        core.crop_origin = (0,0)
        schedule = make_shifted_schedule(5, shift=12.0)
        class ChunkRecorder:
            def append(self, value):
                c,index,current,velocity,t=value
                if index != 3: return
                clean=(current+float(schedule.sigmas[index])*velocity).float().cpu()
                assert torch.isfinite(clean).all()
                path=out/'chunks'/f'chunk_{c:03d}.pt'
                torch.save({'refined':clean[0], 'latent_start':c*5,
                            'latent_end':min(c*5+5,67), 'checkpoint_step':1000},path)
                log('chunk_predicted', chunk=c, latent_start=c*5, latent_end=min(c*5+5,67),
                    sha256=sha(path), final=(c==13))
        write_json(out/'provenance.json',dict(checkpoint_sha256=CHECKPOINT_SHA, config_sha256=CONFIG_SHA,
            pretrained_index_sha256=identity['index_sha256'],sample_sha256=sha(sample_path),
            source_sha256=sha(Path(__file__)),sample=sample['sample_id'],latent_slice=[0,67],
            latent_grid=[68,120],chunk=5,window=6,denoise_calls_per_chunk=4,clean_commit_calls_per_chunk=1,
            target_content='all zeros; GT excluded from inference',history='own generated chunks',
            structural_condition='aligned draft, current and past chunks only',seed=20261109,
            cache_structure=False,cache_device='cpu',training_sample_only=True,optimizer_updates=0))
        status('streaming',committed_chunks=0,total_chunks=14,total_latents=67)
        with torch.no_grad():
            result,*_ = core.stream(batch,generator=torch.Generator(device='npu:0').manual_seed(20261109),
                schedule=schedule,exit_index=3,window=6,record=ChunkRecorder(),cache_structure=False)
        result=result[0].float().cpu()
        assert result.shape==(24,67,68,120) and torch.isfinite(result).all() and forwards==70
        parts=[torch.load(out/'chunks'/f'chunk_{c:03d}.pt',weights_only=True)['refined'] for c in range(14)]
        assert torch.equal(torch.cat(parts,1),result)
        target=out/'refined_67.pt'
        torch.save(dict(refined=result,step=1000,sample=sample['sample_id'],latent_slice=[0,67],
                        checkpoint_sha256=CHECKPOINT_SHA),target)
        metrics=dict(status='complete',checkpoint_step=1000,sample=sample['sample_id'],latent_count=67,
            old_vae_frames=226,fps=24,model_forwards=forwards,refined_sha256=sha(target),
            refined_latent_l1=float((result-sample['reference']).abs().mean()),
            draft_latent_l1=float((sample['upscaled']-sample['reference']).abs().mean()),
            elapsed_seconds=time.time()-started,peak_allocated_bytes=torch.npu.max_memory_allocated(),
            optimizer_updates=0,training_sample_only=True)
        write_json(out/'result.json',metrics); status('complete'); log('complete',result=metrics)
    except BaseException as e:
        write_json(out/'failure.json',dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc()))
        status('failed'); raise

if __name__=='__main__': main()
