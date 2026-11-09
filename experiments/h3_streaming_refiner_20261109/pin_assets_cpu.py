"""Read-only full asset SHA inventory; never initializes an accelerator."""
import argparse,hashlib,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);args=p.parse_args()
r=Path(args.root);out=Path(args.output);out.mkdir(exist_ok=False,parents=True)
files=list(sorted((r/'merged_transformer').glob('*')))
expected={'assets/lora/open_h3_regenerate_2k_lora.safetensors':'8e36a0912ab72f7f3b00f41e3c7c97c7dfdec21e66c88ab22f08b9b0c681bb04','assets/lora/sharp_curve_branches.safetensors':'26ff1c4ea50fc5b0cea252912b67ec308a65d534cef026ef56f415ae92102e4e','assets/lora/adaln_t_table.bin':'c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7','upscaler_production67_limiteddev_20261107/teacher/updates/best_step_2000.pt':'7b9f43c17462fb2d0554645bc7889dc7abdebc052ee9f18f6780624de912e0b5'}
files += [r/n for n in expected]
records={}
for file in files:
    if not file.is_file():continue
    before=file.stat();key=str(file.relative_to(r))
    with file.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    after=file.stat();assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
    if key in expected:assert digest==expected[key]
    records[key]={'bytes':after.st_size,'sha256':digest,'mtime_ns':after.st_mtime_ns,'inode':after.st_ino}
    print(json.dumps({'event':'asset_hashed','file':key,**records[key]}),flush=True)
    (out/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
(out/'complete.json').write_text(json.dumps({'status':'complete','files':records,'all_declared_release_SHA_match':True,'completed_unix':time.time()},indent=2)+'\n')
