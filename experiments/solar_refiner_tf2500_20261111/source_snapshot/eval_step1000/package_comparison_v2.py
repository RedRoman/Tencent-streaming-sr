"""CPU-only comparison of exactly the same latent decoded by two frozen decoders."""
import json,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from render_full_v2 import Writer,write,ROOT,sha

root=ROOT/'media/step1000_full67_v2';out=root/'comparison';out.mkdir(exist_ok=False)
deadline=time.time()+8*3600
try:
    while not all((root/m/'result.json').exists() for m in ['causal','original']):
        assert time.time()<deadline,'Timed out waiting for both decoders'
        p=root/'pipeline_status.json'
        if p.exists():assert json.loads(p.read_text())['status']!='failed','Decoder pipeline failed'
        time.sleep(30)
    results=[json.loads((root/m/'result.json').read_text()) for m in ['causal','original']]
    assert results[0]['prediction_sha256']==results[1]['prediction_sha256']
    files=[root/'original/refiner_original_preview.npy',root/'causal/refiner_causal_preview.npy',
           root/'original/decoded_gt_original_preview.npy']
    labels=['Step1000 | original VAE','Step1000 | causal ViT24','Decoded GT | original VAE']
    arrays=[np.load(p,mmap_mode='r') for p in files]
    assert all(a.shape==(226,360,640,3) for a in arrays)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
    writer=Writer(out/'step1000_two_decoders.mp4',w=1920,h=410)
    sheet=Image.new('RGB',(1920,1230))
    for i in range(226):
        canvas=Image.new('RGB',(1920,410),(16,20,28));d=ImageDraw.Draw(canvas)
        for j,(a,label) in enumerate(zip(arrays,labels)):
            canvas.paste(Image.fromarray(a[i]),(640*j,31));d.text((640*j+8,4),label,font=font,fill='white')
        d.text((8,390),f'Charge train | same step1000 latent | {i+1}/226 frames | 24fps',font=font,fill=(205,210,220))
        writer.push(np.asarray(canvas)[None])
        if i in (0,112,225):sheet.paste(canvas,(0,(0 if i==0 else 1 if i==112 else 2)*410))
    receipt=writer.finish();sheet.save(out/'contactsheet.jpg',quality=95)
    write(out/'result.json',dict(status='complete',frames=226,fps=24,prediction_sha256=results[0]['prediction_sha256'],
        comparison=receipt,training_sample_only=True,GT_scope='original VAE decoded target latent, not native RGB',
        original_refiner=str(root/'original/refiner_original1080p.mp4'),
        causal_refiner=str(root/'causal/refiner_causal1080p.mp4'),continuous_visual_reviewed=False))
    print('Comparison ready',flush=True)
except BaseException as e:
    write(out/'failure.json',dict(error=str(e),traceback=traceback.format_exc()));raise
