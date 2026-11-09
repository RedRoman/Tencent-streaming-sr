"""Full226 auxiliary TF0/TF200/dense/no-refiner/GT comparison, CPU only."""
import argparse
import json
from pathlib import Path
import time
from PIL import Image, ImageDraw, ImageFont
from encode_preview_cpu import checked, encode, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    a=p.parse_args();c=json.loads(Path(a.config).read_text());started=time.time()
    assert sha(c['data_manifest'])==c['data_manifest_sha256']
    data=json.loads(Path(c['data_manifest']).read_text())
    rec=next(r for r in data['samples'] if r['sample_id']==c['sample_id'])
    records=[]
    for d,step,checkpoint in zip(c['pixels'],[0,200,None],[
        'a6b28df41737f71af1cec734291fb2a21cc3032d181d4ad4997440bfce8a267c',
        'b1bd7d862dc98e12bebdd6ea58257fbfd069d8d5173f39bebef7290c2a382ab1',None]):
        assert sha(d['path'])==d['sha256'];r=json.loads(Path(d['path']).read_text())
        assert r['sample_id']==c['sample_id'] and r['training_step']==step
        assert r['checkpoint_sha256']==checkpoint and r['data_manifest_sha256']==c['data_manifest_sha256']
        assert r['status']=='full226_nativeHR_metrics_complete_visual_review_pending'
        assert r['arm']==('released_dense_refiner' if step is None else 'strict_streaming_step'+str(step))
        assert r['raw_decode_shape']==[226,1088,1920,3] and r['crop_y']==[4,1084] and r['fps']==24
        assert r['decoder']=='frozen_official_overlap_provisional_for_quality_comparison'
        records.append(r)
    provenance=[r['RGB'] for r in records]+[rec['rgb']['teacher_student'],rec['rgb']['native_hr']]
    arrays=[checked(d) for d in provenance]
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
    labels=['Streaming TF0 | failed','Streaming TF200 | failed','Released bidirectional','Frozen upsampler | no refiner','Actual native-HR GT']
    fontpath=next(str(p) for p in [Path('/usr/share/fonts/dejavu/DejaVuSans.ttf'),
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')] if p.is_file())
    font=ImageFont.truetype(fontpath,16);small=ImageFont.truetype(fontpath,14)
    sheet=Image.new('RGB',(1920,272*4),(14,19,29));samples=[0,75,150,225]

    def frames():
        for i in range(226):
            canvas=Image.new('RGB',(1920,272),(14,19,29));draw=ImageDraw.Draw(canvas)
            for j,(a,label) in enumerate(zip(arrays,labels)):
                draw.text((j*384+8,5),label,font=font,fill=(240,244,250))
                canvas.paste(Image.fromarray(a[i]).resize((384,216),Image.Resampling.LANCZOS),(j*384,28))
            draw.text((8,249),f'{c["sample_id"]} | {i+1:03d}/226 | 24 fps | Auxiliary checkpoint comparison; no accepted refiner; provisional overlap decoder',font=small,fill=(195,205,220))
            if i in samples:sheet.paste(canvas,(0,samples.index(i)*272))
            yield canvas

    video=encode(out/'checkpoint0_200_dense_ablation_gt.mp4',(1920,272),frames())
    sheet.save(out/'contactsheet.jpg',quality=94)
    result={'status':'full226_auxiliary_video_technical_audit_passed','source_sha256':sha(__file__),
        'encoder_helper_sha256':sha(Path(__file__).with_name('encode_preview_cpu.py')),
        'columns':labels,'sample_id':c['sample_id'],'frames':226,'fps':24,'input_RGB':provenance,
        'pixel_receipts':c['pixels'],'data_manifest_sha256':c['data_manifest_sha256'],
        'files':{'video':video,'contactsheet':{'path':str(out/'contactsheet.jpg'),'sha256':sha(out/'contactsheet.jpg')}},
        'contactsheet_frame_indices':samples,'continuous_visual_reviewed':False,
        'metrics_use_lossless_RGB':True,'accepted_refiner':None,'strict_decoder_validated':False,
        'optimizer_updates':0,'accelerator_forwards':0,'seconds':time.time()-started}
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
