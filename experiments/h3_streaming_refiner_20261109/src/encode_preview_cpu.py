"""CPU-only full226 video packaging, with exact verified RGB inputs.

The three-column interim preview is step200/no-refiner/nativeHR, explicitly
distinct from the final streaming/dense/nativeHR comparison. H.264 is a viewing
copy; quantitative metrics always use the lossless arrays.
"""
import argparse, hashlib, json, subprocess, time
from fractions import Fraction
from pathlib import Path
import numpy as np
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def checked(d):
    p=Path(d['path']);assert sha(p)==d['sha256']
    if 'bytes' in d:assert p.stat().st_size==d['bytes']
    a=np.load(p,mmap_mode='r',allow_pickle=False)
    assert a.shape==(226,1080,1920,3) and a.dtype==np.uint8
    return a

def encode(path,size,frames):
    assert not path.exists()
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    cmd=[ffmpeg,'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pixel_format','rgb24',
         '-video_size',f'{size[0]}x{size[1]}','-framerate','24','-i','pipe:0','-an',
         '-c:v','libx264','-preset','medium','-crf','16','-threads','4','-pix_fmt','yuv420p',
         '-movflags','+faststart',str(path)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for frame in frames:proc.stdin.write(np.asarray(frame,dtype=np.uint8).tobytes())
        proc.stdin.close();assert proc.wait()==0
    finally:
        if proc.poll() is None:proc.terminate();proc.wait()
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-count_frames','-show_entries','stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration',
        '-show_entries','frame=best_effort_timestamp_time','-of','json',str(path)]))
    s=probe['streams'][0];assert (s['width'],s['height'])==size and int(s['nb_read_frames'])==226
    assert Fraction(s['avg_frame_rate'])==24 and abs(float(s['duration'])-226/24)<1e-5
    pts=[float(f['best_effort_timestamp_time']) for f in probe['frames']]
    assert len(pts)==226 and all(abs(t-i/24)<1e-5 for i,t in enumerate(pts))
    subprocess.run([ffmpeg,'-v','error','-xerror','-i',str(path),'-f','null','-'],check=True)
    return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'stream':s,
            'all226_pts_checked':True,'full_decode_error_check_passed':True}

def main():
    p=argparse.ArgumentParser()
    for n in ('data-manifest','data-manifest-sha','sample-id','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--prediction');p.add_argument('--prediction-sha')
    a=p.parse_args();started=time.time();assert sha(a.data_manifest)==a.data_manifest_sha
    data=json.loads(Path(a.data_manifest).read_text());rec=next(x for x in data['samples'] if x['sample_id']==a.sample_id)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    arrays=[checked(rec['rgb']['teacher_student']),checked(rec['rgb']['native_hr'])]
    labels=['Frozen upsampler | no refiner','Actual native-HR GT']
    provenance=[rec['rgb']['teacher_student'],rec['rgb']['native_hr']]
    result={};kind='baseline'
    if a.prediction:
        assert a.prediction_sha and sha(a.prediction)==a.prediction_sha
        pred=json.loads(Path(a.prediction).read_text())
        assert pred['sample_id']==a.sample_id and pred['training_step']==200
        assert pred['checkpoint_sha256']=='b1bd7d862dc98e12bebdd6ea58257fbfd069d8d5173f39bebef7290c2a382ab1'
        assert pred['data_manifest_sha256']==a.data_manifest_sha and pred['fps']==24 and pred['crop_y']==[4,1084]
        arrays.insert(0,checked(pred['RGB']));labels.insert(0,'Streaming refiner | TF step200')
        provenance.insert(0,pred['RGB']);kind='step200_engineering_preview'
        result['step200_1080p']=encode(out/'step200_1080p.mp4',(1920,1080),iter(arrays[0]))
    fontpath=next(str(p) for p in (Path('/usr/share/fonts/dejavu/DejaVuSans.ttf'),
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')) if p.is_file())
    titlefont=ImageFont.truetype(fontpath,25);smallfont=ImageFont.truetype(fontpath,21)
    width=960*len(arrays);height=620
    def panels():
        for i in range(226):
            canvas=Image.new('RGB',(width,height),(14,19,29));draw=ImageDraw.Draw(canvas)
            for j,(frames,label) in enumerate(zip(arrays,labels)):
                draw.text((j*960+20,10),label,font=titlefont,fill=(240,244,250))
                im=Image.fromarray(frames[i]).resize((960,540),Image.Resampling.LANCZOS)
                canvas.paste(im,(j*960,44))
            footer=f'{a.sample_id}   |   {i+1:03d}/226   |   24 fps   |   '+('ENGINEERING PREVIEW - not accepted; dense refiner comparison pending' if a.prediction else 'Frozen-upsampled baseline / native-HR reference - no new refiner output')
            draw.text((20,591),footer,font=smallfont,fill=(195,205,220))
            if i in (0,24,48,72,96,120,144,168,192,225):canvas.save(out/f'frame_{i:03d}.jpg',quality=93)
            yield canvas
    result['comparison']=encode(out/(kind+'_comparison.mp4'),(width,height),panels())
    report={'status':'full226_video_technical_audit_passed','kind':kind,'sample_id':a.sample_id,
            'columns':labels,'input_RGB':provenance,'data_manifest_sha256':a.data_manifest_sha,
            'files':result,'fps':24,'frames':226,'seconds':time.time()-started,
            'metrics_use_lossless_RGB_not_H264':True,'continuous_visual_reviewed':False,
            'accepted_refiner':None,'strict_streaming_decoder_validated':False,'optimizer_updates':0}
    (out/'video_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'files':result}),flush=True)

if __name__=='__main__':main()
