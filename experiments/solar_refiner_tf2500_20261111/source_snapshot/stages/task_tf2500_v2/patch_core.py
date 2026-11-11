"""Native-scale spatial patches with full-canvas RoPE and a full-frame appearance reference.

Cropping changes visible context, not latent scale or the pretrained weight geometry.
The packed attention/teacher-forcing implementation remains upstream SolarRenderer.
"""
from dataclasses import replace
import numpy as np
import torch
from solarwm.backends.minimax_h3.artifacts import H3ArtifactBatch,H3VisualObservation
from solarwm.backends.minimax_h3.omni_causal import H3OmniCausalCore
from solarwm.backends.minimax_h3.geometry import frame_position_grid


class PatchCore(H3OmniCausalCore):
    crop_origin=(0,0)

    @staticmethod
    def latent_grid(batch):
        grid=tuple(int(n) for n in batch.target_latents.shape[-2:])
        assert grid in ((68,120),(32,64),(32,48)),grid
        return grid

    def layout(self,tags,*,anchors=('first',),observations=(),grid=(48,84),frames=47):
        layout=super().layout(tags,anchors=anchors,observations=observations,grid=grid,frames=frames)
        if grid==(68,120):return layout
        assert not anchors and len(observations)==2
        assert (observations[0].height,observations[0].width)==(68,120)
        assert observations[0].num_frames==1 and observations[0].frames is None
        assert (observations[1].height,observations[1].width)==grid
        y,x=self.crop_origin;h,w=grid
        assert y%2==x%2==0 and 0<=y<=68-h and 0<=x<=120-w
        full_grid,full_width=frame_position_grid(68,120)
        full=full_grid.reshape(34,60,2)
        patch=full[y//2:(y+h)//2,x//2:(x+w)//2].reshape(-1,2)
        source=layout.source
        position_ids=source.position_ids.copy()
        # First free observation stays full-frame; aligned structure and targets
        # use exactly the corresponding positions from the original canvas.
        video=source.video_indices
        positions=np.concatenate((full.reshape(-1,2),np.tile(patch,(2*frames,1))),axis=0)
        assert len(video)==len(positions)
        position_ids[video,1:]=positions
        # Stereo silence stays at the full canvas's left/right rotary positions.
        audio=source.audio_indices
        assert len(audio)%2==0
        position_ids[audio[:len(audio)//2],2]=full_width[0]
        position_ids[audio[len(audio)//2:],2]=full_width[-1]
        return replace(layout,source=replace(source,position_ids=position_ids),
                       position_ids=torch.from_numpy(position_ids).to(self.device))


def batch_for(sample,condition,start,frames,crop=None,inference=False):
    full_draft=sample['upscaled'][:,start:start+frames].float().contiguous()
    full_gt=sample['reference'][:,start:start+frames].float().contiguous()
    assert full_draft.shape==full_gt.shape==(24,frames,68,120)
    if crop is None:
        draft,gt=full_draft,full_gt
    else:
        y,x,h,w=crop
        assert y%2==x%2==h%2==w%2==0
        draft=full_draft[:,:,y:y+h,x:x+w].contiguous()
        gt=full_gt[:,:,y:y+h,x:x+w].contiguous()
        assert draft.shape==gt.shape==(24,frames,h,w)
    prompt,tags=condition['prompt_embeds'],condition['text_token_tags']
    pixels=17*((frames-2)//5)+5
    return H3ArtifactBatch(sample_id=sample['sample_id'],start_frame=0,
        plan_fingerprint=sample['sample_id'],target_latents=torch.zeros_like(gt) if inference else gt,
        anchor_latents=None,prompt_embeds=prompt,text_token_tags=tags,
        source_frame_indices=torch.arange(pixels),camera_viewmats=None,camera_K=None,source_fps=24.0,
        text_prompt_embeds=prompt,text_prompt_tags=tags,dataset_source='train2_native_patch_v2',
        visual_observations=(H3VisualObservation(full_draft[:,:1],frames=None),
            H3VisualObservation(draft,frames=tuple(range(frames)))))
