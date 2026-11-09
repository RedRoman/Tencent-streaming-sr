"""Exact TF schedule and restricted CPU restore of this project's AdamW state."""
import hashlib
from pathlib import Path

UPSAMPLER_SHA='7b9f43c17462fb2d0554645bc7889dc7abdebc052ee9f18f6780624de912e0b5'
ADAPTER200_SHA='b1bd7d862dc98e12bebdd6ea58257fbfd069d8d5173f39bebef7290c2a382ab1'
OPTIMIZER200_SHA='3b770ffc6e5fcabb7e78cf3481105531fe6b0372ea898706d8f8af1a16dc1ef5'


def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def schedule(completed_updates,sample_count=2):
    if completed_updates<0 or sample_count!=2:raise ValueError('Unexpected TF schedule')
    step=completed_updates
    pos=(step//sample_count*17)%67
    return {'sample_index':step%sample_count,'latent_index':pos,
            'history':sorted(set([0,max(0,pos-1),pos]))[:-1],
            'frames':sorted(set([0,max(0,pos-1),pos])),
            'sigma':[.909375,6/7,2/3][(step//sample_count)%3],
            'noise_seed':20261109+step//134}


def _cpu_format(value):
    # This exact project's saved optimizer uses ND. Reject other layouts.
    if value!=2:raise ValueError('Unexpected legacy optimizer storage format')
    return value


def _cpu_rebuild(storage,offset,size,stride,requires_grad,hooks,npu_format):
    import torch
    if storage._untyped_storage.device.type!='cpu' or requires_grad or hooks or npu_format!=2:
        raise ValueError('Unexpected legacy optimizer tensor')
    return torch.empty(0,dtype=storage.dtype).set_(storage,offset,size,stride)


def load_cpu(path,digest):
    import torch
    if sha(path)!=digest:raise ValueError('Resume payload SHA mismatch')
    # Map only these two known legacy globals to CPU-only functions. The original
    # NPU rebuild function and torch_npu are never imported by this reader.
    with torch.serialization.safe_globals([
        (_cpu_rebuild,'torch_npu.utils.storage._rebuild_npu_tensor'),
        (_cpu_format,'torch_npu.npu._format.Format')]):
        return torch.load(path,map_location='cpu',weights_only=True)


def validate_bundle(adapter,private,step):
    import torch
    if adapter['step']!=step or private['step']!=step:raise ValueError('Resume step mismatch')
    if adapter['format']!='streaming-refiner-adapter-v1' or adapter['upsampler_sha256']!=UPSAMPLER_SHA:
        raise ValueError('Frozen model binding changed')
    names=list(adapter['adapter']);opt=private['optimizer'];groups=opt['param_groups']
    if len(names)!=401 or len(groups)!=1:raise ValueError('Trainable group changed')
    group=groups[0];ids=group['params']
    if ids!=list(range(401)) or set(opt['state'])!=set(ids):raise ValueError('Optimizer parameter order changed')
    expected={'lr':1e-4,'betas':(.9,.99),'eps':1e-8,'weight_decay':0,'foreach':False,
              'amsgrad':False,'maximize':False,'capturable':False,'differentiable':False}
    for k,v in expected.items():
        if group.get(k)!=v:raise ValueError('Optimizer configuration changed: '+k)
    count=0
    for name,index in zip(names,ids):
        p=adapter['adapter'][name];state=opt['state'][index]
        if p.dtype!=torch.float32 or p.device.type!='cpu' or not bool(torch.isfinite(p).all()):
            raise ValueError('Invalid trainable tensor')
        if set(state)!={'step','exp_avg','exp_avg_sq'} or float(state['step'])!=step:
            raise ValueError('Incomplete optimizer state')
        for key in ('exp_avg','exp_avg_sq'):
            v=state[key]
            if v.shape!=p.shape or v.dtype!=p.dtype or v.device.type!='cpu' or not bool(torch.isfinite(v).all()):
                raise ValueError('Invalid optimizer moment')
        if not bool((state['exp_avg_sq']>=0).all()):raise ValueError('Negative second moment')
        count+=p.numel()
    if count!=20586496:raise ValueError('Trainable count changed')
    return {'step':step,'parameter_tensors':len(names),'parameter_elements':count,
            'all_optimizer_steps_match':True,'all_moments_finite':True,
            'optimizer_configuration_preserved':True,'parameter_order_sha256':hashlib.sha256('\n'.join(names).encode()).hexdigest()}


def cpu_tree(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value
