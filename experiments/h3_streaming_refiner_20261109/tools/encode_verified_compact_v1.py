"""CPU-only playback derivatives; never substitutes for lossless evaluation."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--video-receipt',required=True)
    p.add_argument('--video-receipt-sha',required=True)
    p.add_argument('--sample-id',required=True)
    p.add_argument('--stage-step',type=int,choices=[1000,2000],required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    assert sha(a.video_receipt)==a.video_receipt_sha
    rec=json.loads(Path(a.video_receipt).read_text())
    assert rec['status']=='full226_three_videos_technical_audit_passed'
    assert rec['sample_id']==a.sample_id and rec['stage_step']==a.stage_step
    assert rec['frames']==226 and rec['fps']==24
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    import imageio_ffmpeg
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    result={}
    for kind,width in [('candidate_native1080',960),('main_comparison',1440),('no_refiner_ablation',1440)]:
        source=rec['files'][kind];src=Path(source['path'])
        assert sha(src)==source['sha256'] and src.stat().st_size==source['bytes']
        assert source['all226_pts_checked'] and source['full_decode_error_check_passed']
        dst=out/(src.stem+'_preview.mp4')
        subprocess.run([ffmpeg,'-v','error','-n','-i',str(src),'-vf',f'scale={width}:-2','-c:v','libx264','-threads','4','-preset','fast','-crf','29','-pix_fmt','yuv420p','-movflags','+faststart',str(dst)],check=True)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration','-show_entries','frame=best_effort_timestamp_time','-of','json',str(dst)],text=True))
        stream=probe['streams'][0];pts=[float(x['best_effort_timestamp_time']) for x in probe['frames']]
        assert int(stream['nb_read_frames'])==226 and Fraction(stream['avg_frame_rate'])==24
        assert len(pts)==226 and all(abs(t-i/24)<1e-5 for i,t in enumerate(pts))
        assert abs(float(stream['duration'])-226/24)<1e-5
        subprocess.run([ffmpeg,'-v','error','-xerror','-i',str(dst),'-f','null','-'],check=True)
        result[dst.name]={'kind':kind,'source_sha256':source['sha256'],'sha256':sha(dst),'bytes':dst.stat().st_size,'stream':stream,'all226_pts_checked':True,'full_decode_passed':True}
    report={'status':'three_full226_playback_derivatives_verified','sample_id':a.sample_id,'stage_step':a.stage_step,'source_receipt_sha256':a.video_receipt_sha,'files':result,'display_only_downscaled':True,'metrics_from_full_resolution_lossless_RGB':True,'continuous_visual_reviewed':False,'accelerator_forwards':0,'optimizer_updates':0}
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    with tarfile.open(out/'preview_bundle.tar','w') as t:
        for dst in sorted(out.glob('*.mp4')):t.add(dst,arcname=dst.name)
        t.add(out/'result.json',arcname='result.json')
    print(json.dumps({'bundle':str(out/'preview_bundle.tar'),'sha256':sha(out/'preview_bundle.tar'),'bytes':(out/'preview_bundle.tar').stat().st_size,'files':result}),flush=True)

if __name__=='__main__':main()
