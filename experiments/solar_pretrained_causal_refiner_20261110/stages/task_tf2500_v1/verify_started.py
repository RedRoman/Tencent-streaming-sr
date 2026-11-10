"""CPU-only audit that the first saved update changed weights and includes full resume state."""
import hashlib
import json
from pathlib import Path
import torch

torch.set_num_threads(4)
root=Path('/opt/refiner-cache/h3-pretrained-causal-refiner-20261109')
run=root/'runs/task_tf2500_v1'
saved=[]
for step in (0,1):
    p=run/f'checkpoint_{step:04d}.pt'
    receipt=json.loads(p.with_suffix('.json').read_text())
    with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
    saved.append(torch.load(p,map_location='cpu',weights_only=True))
before,after=saved
assert after['step']==1 and before['step']==0
assert after['parameter_names']==before['parameter_names']
assert len(after['optimizer']['state'])==400
assert all(float(v['step'])==1 for v in after['optimizer']['state'].values())
changed=[]
max_delta=0.0
for name in after['parameter_names']:
    a,b=before['adapter'][name],after['adapter'][name]
    assert torch.isfinite(b).all()
    delta=float((a-b).abs().max())
    if delta:changed.append(name)
    max_delta=max(delta,max_delta)
assert len(changed)==200 and all(n.endswith('.b') for n in changed)
events=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
assert any(e['event']=='live_adapter_optimizer_reload_verified' for e in events)
update=next(e for e in events if e['event']=='optimizer_update' and e['step']==1)
launch=json.loads((root/'launches/task_tf2500_v1/launch.json').read_text())
record={'status':'real_update_and_resume_verified','step':1,'changed_adapter_tensors':len(changed),
    'adapter_max_abs_delta':max_delta,'optimizer_parameter_states':400,'first_update':update,
    'launch':launch,'config_sha256':after['config_sha256'],
    'checkpoint_sha256':json.loads((run/'checkpoint_0001.json').read_text())['sha256'],
    'scope':'Startup correctness only; no RGB/generalization claim'}
(root/'evidence/training_started_v1.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
