"""Candidate stage2000 native1080, dense comparison, and no-refiner ablation."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from encode_preview_cpu import sha, checked, encode

def validate_receipts(candidate,dense,sample_id,data_sha,checkpoint_sha):
    for r in (candidate,dense):
        assert r['status']=='full226_nativeHR_metrics_complete_visual_review_pending'
        assert r['sample_id']==sample_id and r['data_manifest_sha256']==data_sha
        assert r['raw_decode_shape']==[226,1088,1920,3] and r['crop_y']==[4,1084] and r['fps']==24
        assert r['decoder']=='frozen_official_overlap_provisional_for_quality_comparison' and r['optimizer_updates']==0
    assert candidate['arm']=='anchored_stage2000_candidate' and candidate['training_step']==4500
    assert candidate['checkpoint_sha256']==checkpoint_sha
    assert dense['arm']=='released_dense_refiner' and dense['checkpoint_sha256'] is None and dense['training_step'] is None

def main():
    p=argparse.ArgumentParser()
    for key in ['data-manifest','data-manifest-sha','candidate-pixels','candidate-pixels-sha',
                'dense-pixels','dense-pixels-sha','checkpoint-sha','sample-id','output']:
        p.add_argument('--'+key,required=True)
    a=p.parse_args();assert sha(a.data_manifest)==a.data_manifest_sha
    assert sha(a.candidate_pixels)==a.candidate_pixels_sha and sha(a.dense_pixels)==a.dense_pixels_sha
    data=json.loads(Path(a.data_manifest).read_text());sample=next(r for r in data['samples'] if r['sample_id']==a.sample_id)
    c=json.loads(Path(a.candidate_pixels).read_text());d=json.loads(Path(a.dense_pixels).read_text())
    validate_receipts(c,d,a.sample_id,a.data_manifest_sha,a.checkpoint_sha)
    assert c['no_refiner_nativeHR_metrics']==d['no_refiner_nativeHR_metrics']==sample['no_refiner_native_metrics']
    sources=[c['RGB'],d['RGB'],sample['rgb']['native_hr'],sample['rgb']['teacher_student']]
    arrays=[checked(r) for r in sources]
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    files={'candidate_native1080':encode(out/'anchored_stage2000_native1080.mp4',(1920,1080),iter(arrays[0]))}
    fontpath=next(str(p) for p in [Path('/usr/share/fonts/dejavu/DejaVuSans.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')] if p.exists())
    font=ImageFont.truetype(fontpath,23);small=ImageFont.truetype(fontpath,17)
    def frames(indices,labels,name):
        for i in range(226):
            canvas=Image.new('RGB',(2880,620),(14,19,29));draw=ImageDraw.Draw(canvas)
            for j,(index,label) in enumerate(zip(indices,labels)):
                draw.text((960*j+14,8),label,font=font,fill=(240,244,250))
                canvas.paste(Image.fromarray(arrays[index][i]).resize((960,540),Image.Resampling.LANCZOS),(960*j,42))
            draw.text((14,592),f'{a.sample_id} | {i+1:03d}/226 | 24fps | Stage2000 candidate; not accepted | Overlap decoder; strict decoder unverified',font=small,fill=(205,215,225))
            if i in (0,24,48,72,96,120,144,168,192,225):canvas.save(out/f'{name}_frame{i:03d}.jpg',quality=94)
            yield canvas
    mainlabels=['Anchored stage2000 | candidate','Original bidirectional refiner','Actual native-HR GT']
    ablationlabels=['Anchored stage2000 | candidate','Frozen upsampler | no refiner','Actual native-HR GT']
    files['main_comparison']=encode(out/'anchored_stage2000_dense_gt.mp4',(2880,620),frames([0,1,2],mainlabels,'main'))
    files['no_refiner_ablation']=encode(out/'anchored_stage2000_no_refiner_gt.mp4',(2880,620),frames([0,3,2],ablationlabels,'ablation'))
    result={'status':'full226_three_videos_technical_audit_passed','sample_id':a.sample_id,'stage_step':2000,
            'total_training_step':4500,'checkpoint_sha256':a.checkpoint_sha,'input_RGB':sources,'files':files,
            'main_columns':mainlabels,'ablation_columns':ablationlabels,'frames':226,'fps':24,
            'full226_ratios_vs_no_refiner':c['full226_ratios_vs_no_refiner'],'metrics_use_lossless_RGB':True,
            'continuous_visual_reviewed':False,'strict_streaming_decoder_validated':False,
            'accepted_refiner':None,'optimizer_updates':0,'accelerator_forwards':0}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
