"""Full226 auxiliary streaming/no-refiner/GT preview, with lossless input checks.

This diagnostic is not the required selected-streaming/dense/GT main comparison.
"""
import argparse
import json
from pathlib import Path
import time
from PIL import Image, ImageDraw, ImageFont
from encode_preview_cpu import checked, encode, sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--config-sha',required=True)
    args=parser.parse_args();started=time.time()
    assert sha(args.config)==args.config_sha
    config=json.loads(Path(args.config).read_text())
    data_sha='9ce8eebb66ad62e243dfa0eafef2961bdf72539d44ed514f03b8dc99d4f31f23'
    assert sha(config['data_manifest'])==data_sha
    data=json.loads(Path(config['data_manifest']).read_text())
    rec=next(x for x in data['samples'] if x['sample_id']==config['sample_id'])
    receipt=config['pixel_receipt'];assert sha(receipt['path'])==receipt['sha256']
    pred=json.loads(Path(receipt['path']).read_text())
    step=config['training_step']
    assert pred['sample_id']==rec['sample_id'] and pred['training_step']==step
    assert pred['checkpoint_sha256']==config['checkpoint_sha256']
    assert pred['arm']==f'strict_streaming_step{step}'
    assert pred['status']=='full226_nativeHR_metrics_complete_visual_review_pending'
    assert pred['data_manifest_sha256']==data_sha
    assert pred['raw_decode_shape']==[226,1088,1920,3] and pred['crop_y']==[4,1084] and pred['fps']==24
    assert pred['decoder']=='frozen_official_overlap_provisional_for_quality_comparison'
    assert pred['no_refiner_nativeHR_metrics']==rec['no_refiner_native_metrics']
    ratios={k:pred['nativeHR_metrics']['regions']['full226'][k]/rec['no_refiner_native_metrics']['regions']['full226'][k]
            for k in ('pixel_l1','lpips')}
    assert ratios==pred['full226_ratios_vs_no_refiner']
    failed=any(v>1.02 for v in ratios.values())
    provenance=[pred['RGB'],rec['rgb']['teacher_student'],rec['rgb']['native_hr']]
    arrays=[checked(d) for d in provenance]
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=False)
    labels=[f'Streaming TF{step} | '+('metric FAIL' if failed else 'unselected'),
            'Frozen upsampler | no refiner','Actual native-HR GT']
    fontpath=next(str(p) for p in (Path('/usr/share/fonts/dejavu/DejaVuSans.ttf'),
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')) if p.is_file())
    font=ImageFont.truetype(fontpath,19);small=ImageFont.truetype(fontpath,15)
    samples=[0,75,150,225];sheet=Image.new('RGB',(1440,320*4),(14,19,29))

    def frames():
        for i in range(226):
            canvas=Image.new('RGB',(1440,320),(14,19,29));draw=ImageDraw.Draw(canvas)
            for j,(array,label) in enumerate(zip(arrays,labels)):
                draw.text((480*j+10,5),label,font=font,fill=(240,244,250))
                canvas.paste(Image.fromarray(array[i]).resize((480,270),Image.Resampling.LANCZOS),(480*j,27))
            footer=f'{rec["sample_id"]} | {i+1:03d}/226 | 24 fps | Auxiliary diagnostic; no selected refiner; provisional overlap decoder'
            draw.text((10,300),footer,font=small,fill=(195,205,220))
            if i in samples:sheet.paste(canvas,(0,samples.index(i)*320))
            yield canvas

    video=encode(out/'stream_no_refiner_gt.mp4',(1440,320),frames())
    sheet.save(out/'contactsheet.jpg',quality=94)
    result={'status':'full226_auxiliary_video_technical_audit_passed','sample_id':rec['sample_id'],
        'training_step':step,'checkpoint_sha256':pred['checkpoint_sha256'],'columns':labels,
        'source_sha256':sha(__file__),'encoder_helper_sha256':sha(Path(__file__).with_name('encode_preview_cpu.py')),
        'config_sha256':args.config_sha,'pixel_receipt':receipt,'input_RGB':provenance,
        'data_manifest_sha256':data_sha,'frames':226,'fps':24,'full226_ratios_vs_no_refiner':ratios,
        'first_clip_metric_gate_failed':failed,'files':{'video':video,
            'contactsheet':{'path':str(out/'contactsheet.jpg'),'sha256':sha(out/'contactsheet.jpg')}},
        'contactsheet_frame_indices':samples,'continuous_visual_reviewed':False,
        'main_selected_streaming_dense_GT_comparison':False,'accepted_refiner':None,
        'strict_streaming_decoder_validated':False,'metrics_use_lossless_RGB':True,
        'optimizer_updates':0,'accelerator_forwards':0,'seconds':time.time()-started}
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
