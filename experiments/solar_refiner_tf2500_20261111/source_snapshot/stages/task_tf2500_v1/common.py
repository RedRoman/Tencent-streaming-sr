import hashlib
import json
from pathlib import Path
import torch
from solarwm.backends.minimax_h3.artifacts import H3ArtifactBatch, H3VisualObservation
from solarwm.backends.minimax_h3.omni_causal import H3OmniCausalCore

ROOT = Path('/opt/refiner-cache/h3-pretrained-causal-refiner-20261109')
VENDOR = ROOT/'vendor/SolarRenderer-0ca4f5e45b062cc9e9aee506de46aabdc22e4c3d'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)

def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value

class RefinerCore(H3OmniCausalCore):
    @staticmethod
    def latent_grid(batch):
        grid = tuple(int(n) for n in batch.target_latents.shape[-2:])
        if grid != (68, 120):
            raise ValueError(f'Full spatial geometry required: {grid}')
        return grid

def schedule(step):
    # Alternate both clips; every complete 24-update cycle sees all 12 starts.
    # 12-latent crops cover 0:12, 5:17, ... 55:67 without spatial resampling.
    return {'sample_index': step % 2, 'latent_start': ((step // 2) % 12)*5,
            'latent_count': 12, 'noise_seed': 2026111000+step}

def batch_for(sample, condition, start=0, frames=12, inference=False):
    draft = sample['upscaled'][:, start:start+frames].float().contiguous()
    gt = sample['reference'][:, start:start+frames].float().contiguous()
    assert draft.shape == gt.shape == (24, frames, 68, 120)
    pixels = 17*((frames-2)//5)+5
    prompt, tags = condition['prompt_embeds'], condition['text_token_tags']
    return H3ArtifactBatch(sample_id=sample['sample_id'], start_frame=0,
        plan_fingerprint=sample.get('source_sha256', sample['sample_id']),
        target_latents=torch.zeros_like(gt) if inference else gt, anchor_latents=None,
        prompt_embeds=prompt, text_token_tags=tags, source_frame_indices=torch.arange(pixels),
        camera_viewmats=None, camera_K=None, source_fps=24.0,
        text_prompt_embeds=prompt, text_prompt_tags=tags, dataset_source='train2_task_lora_v1',
        visual_observations=(H3VisualObservation(draft[:, :1], frames=None),
            H3VisualObservation(draft, frames=tuple(range(frames)))))
