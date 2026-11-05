"""Plot measured CMN training history; no fitted or invented points."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();s=json.loads((a.run/'summary.json').read_text())
if s['status']!='completed':raise ValueError('Plot requires completed training')
r=[json.loads(x) for x in (a.run/'steps.jsonl').read_text().splitlines() if x.strip()]
steps=s['completed_updates']
if [v['step'] for v in r]!=list(range(1,steps+1)):raise ValueError('Noncontiguous training log')
fig,axes=plt.subplots(1,3,figsize=(15,4.3),constrained_layout=True)
window=min(100,steps);x=np.array([v['step'] for v in r])
for ax,key,title in [(axes[0],'latent_mse','Train latent MSE'),(axes[1],'ewarp_l1','Train eWarp L1')]:
    y=np.array([v[key] for v in r]);ax.plot(x[window-1:],np.convolve(y,np.ones(window)/window,mode='valid'),color='#146c67')
    ax.set_title(f'{title} ({window}-update mean)');ax.set_xlabel('Optimizer updates');ax.grid(alpha=.2)
val=s['validation'];axes[2].plot([v['step'] for v in val],[v['mean']['latent_mse'] for v in val],color='#146c67',marker='o',markersize=3,label='CMN')
axes[2].axhline(s['bilinear_dev']['mean']['latent_mse'],color='#888888',linestyle='--',label='Bilinear')
best=next(v for v in val if v['step']==s['best_step'])
axes[2].scatter([best['step']],[best['mean']['latent_mse']],marker='*',s=90,color='#d57e1d',label='Selected by dev MSE')
axes[2].set_title('Validation latent MSE (16 clips)');axes[2].set_xlabel('Optimizer updates');axes[2].grid(alpha=.2);axes[2].legend(frameon=False,fontsize=8)
fig.suptitle(f'H3 CMN upscaler: {steps:,} updates, frozen H3 latent pairs',fontsize=14)
a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=160)
print(a.output)
