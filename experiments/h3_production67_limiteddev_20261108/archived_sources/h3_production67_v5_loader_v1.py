"""Candidate strict CPU v5 warm-start for the reviewed explicit-target wrapper.

No model factory, optimizer, device selection, inference, decoder, download or
training launcher is provided. Runtime Torch paths remain unexecuted in the
local author preparation. Pure metadata checks can run without importing Torch.
"""
from __future__ import annotations
import contextlib, hashlib, importlib, inspect, json, math, os, re, stat, sys
from pathlib import Path

V5_SHA = 'c7aaaa396d877a84c3fed2a5536189b7ed5e61226a82ef1a9e6be1e831f7f914'
TENSOR_MANIFEST_SHA = '071b3fd8b71dfde906b1f75f250d79c8d8df32f24b0c0c194b83387bf1705a75'
EXPECTED_TENSORS, EXPECTED_PARAMETERS = 322, 345280216
FROZEN_SOURCES = {
 'causal_upscaler.py':'8032ebd29e558d2cad7f31cb91618022892270a5d4027052f251475b04f98677',
 'h3_production67_geometry_v1.py':'147765ef79549bc31fd2b9e5026d5983be2328ee12ee04743f9202543b4af383',
 'h3_conv3d_explicit_target_v1.py':'8c5bcbd1702a252266c68cbdf131e40fa7b56366deaeef17c53a9b84dbd5b3c6',
 'train_upscaler_trial.py':'ae0f5d0b185600ebc1b674977b1b338607aee5aab801a4ad4cd1e8e1b6710a71',
}
UPSTREAM_SOURCE_SHAS = {
 'e54959b609c5732711fd8e87b5e6cbe4e80977379d988df175afa1c7745f4521',
 'f5945b500ddb38afee23c377034bf15394ba64748511a7e9608953040c4be437',
}
ADAPTER_EXPECTED = dict(version=1, normalization='D(U(N(native_normalized_H3_latent)))', initialization='aligned_past', temporal_padding='left_only_zero', groupnorm='per_frame',
 spatial_scale=2.0, temporal_chunking=False, dtype='float32', saved_weights_already_initialized=True,
 reload_must_not_reinitialize_weights=True)


def require(value, message):
    if not value: raise ValueError(message)


def sha_stream(handle):
    h=hashlib.sha256();handle.seek(0)
    for block in iter(lambda:handle.read(8 << 20),b''):h.update(block)
    return h.hexdigest()


def sha_file(path):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'Expected regular original file')
    with path.open('rb') as h:return sha_stream(h)


def load_tensor_manifest(path):
    path=Path(path);require(path.is_file() and not path.is_symlink() and path.stat().st_size<1<<20,'Invalid tensor manifest file')
    raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==TENSOR_MANIFEST_SHA,'Pinned v5 tensor manifest SHA differs')
    doc=json.loads(raw)
    require(doc['status']=='passed' and doc['variant']=='v5' and doc['original_checkpoint_sha256']==V5_SHA and
            doc['original_checkpoint_sha256_after']==V5_SHA and doc['tensor_count']==EXPECTED_TENSORS and
            doc['parameter_count']==EXPECTED_PARAMETERS and doc['all_tensors_bitwise_equal'] is True and
            doc['tensor_dtype_unchanged'] is True,'Wrong published v5 tensor provenance')
    schema=doc['checks'];require(len(schema)==EXPECTED_TENSORS,'Expected322 original tensor names')
    total=0
    for name,row in schema.items():
        require(isinstance(name,str) and name and not name.startswith('model.'),'Only original unprefixed tensor names')
        require(isinstance(row['shape'],list) and all(type(n) is int and n>0 for n in row['shape']),'Invalid recorded shape')
        require(type(row['numel']) is int and math.prod(row['shape'])==row['numel'] and row['dtype']=='float32' and
                row['bitwise_equal'] is True and re.fullmatch('[0-9a-f]{64}',row['tensor_sha256']),'Invalid tensor metadata')
        total+=row['numel']
    require(total==EXPECTED_PARAMETERS,'Recorded original tensor element count differs')
    return schema


def validate_inventory(inventory,schema):
    """Metadata-only validation; never substitutes for actual Tensor checks."""
    require(isinstance(inventory,dict) and set(inventory)==set(schema),'Missing/extra/renamed original tensor keys')
    for name,row in schema.items():
        actual=inventory[name]
        require(isinstance(actual['shape'],(tuple,list)) and all(type(n) is int for n in actual['shape']) and list(actual['shape'])==row['shape'] and actual['dtype']=='float32' and type(actual['numel']) is int and actual['numel']==row['numel'],
                'Original tensor shape/dtype/numel differs: '+name)
    return {'tensor_count':len(schema),'numel':sum(x['numel'] for x in inventory.values()),'scope':'metadata_only_not_loaded_tensor_validation'}


def validate_saved_metadata(saved):
    require(isinstance(saved,dict) and type(saved.get('step')) is int and saved['step']==1800 and saved.get('state_dict_format')=='original_upscaler','Wrong v5 selected step or state format')
    cfg=saved.get('adapter_config');require(isinstance(cfg,dict) and cfg==ADAPTER_EXPECTED,'Wrong already-aligned adapter provenance')
    for key in ('temporal_chunking','saved_weights_already_initialized','reload_must_not_reinitialize_weights'):
        require(type(cfg[key]) is bool,'Adapter flags must be actual booleans')
    require(saved.get('model_config')==dict(kind='h3_causal_conv3d',latent_channels=24,spatial_scale=2,adapter_config=cfg),'Wrong original model schema')
    c=saved.get('contract',{}).get('config',{})
    require(type(c.get('steps')) is int and c['steps']==2000 and type(c.get('seed')) is int and c['seed']==20261105 and c.get('dtype')=='fp32','Wrong v5 run lineage')
    return cfg


def verify_source_bundle(paths):
    require(isinstance(paths,dict) and set(paths)==set(FROZEN_SOURCES),'Require exact four frozen wrapper/adapter/geometry/activation sources')
    return {name:{'path':str(Path(paths[name]).resolve()),'sha256':digest} for name,digest in FROZEN_SOURCES.items()
            if _verify_one(paths[name],digest)}


def _verify_one(path,expected):
    require(sha_file(path)==expected,'Frozen source differs: '+str(path));return True


def _import_verified(name,source_paths):
    """Caller supplies existing import environment; do not modify sys.path."""
    filename=name+'.py';module=importlib.import_module(name)
    require(Path(inspect.getfile(module)).resolve()==Path(source_paths[filename]).resolve() and
            sha_file(inspect.getfile(module))==FROZEN_SOURCES[filename],'Imported module differs from reviewed file')
    return module


@contextlib.contextmanager
def _verified_original_file(path):
    """SHA pinned BEFORE pickle loading, then same opened file rehashed afterward."""
    path=Path(path);require(path.is_file() and not path.is_symlink(),'Original v5 checkpoint missing/nonregular; partial or symlink is not accepted')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as handle:
        before=os.fstat(handle.fileno());require(stat.S_ISREG(before.st_mode),'Checkpoint must be regular')
        require(sha_stream(handle)==V5_SHA,'Original complete v5 checkpoint SHA differs; do not deserialize')
        handle.seek(0)
        yield handle
        after=os.fstat(handle.fileno())
        require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) and sha_stream(handle)==V5_SHA,
                'Checkpoint changed during deserialization')


def _tensor_inventory(state,torch):
    require(isinstance(state,dict),'Expected original tensor mapping')
    result={}
    for name,tensor in state.items():
        require(isinstance(tensor,torch.Tensor) and tensor.layout==torch.strided and tensor.device.type=='cpu' and
                tensor.dtype==torch.float32,'Require actual dense CPU FP32 tensors: '+str(name))
        result[name]={'shape':list(tensor.shape),'dtype':'float32','numel':tensor.numel()}
    return result


def _verify_tensor_values(state,schema,torch):
    require(sys.byteorder=='little','Recorded FP32 tensor byte hashes require little-endian host')
    validate_inventory(_tensor_inventory(state,torch),schema)
    for name,tensor in state.items():
        require(bool(torch.isfinite(tensor).all()),'Nonfinite initial tensor: '+name)
        digest=hashlib.sha256(tensor.detach().contiguous().numpy().tobytes(order='C')).hexdigest()
        require(digest==schema[name]['tensor_sha256'],'Tensor values differ from original v5 export audit: '+name)


def _copy_strict_into_original(model,state,schema,torch):
    # All schema/value checks happen before any model parameter copy.
    _verify_tensor_values(state,schema,torch)
    validate_inventory(_tensor_inventory(model.state_dict(),torch),schema)
    require(sum(p.numel() for p in model.parameters())==sum(x['numel'] for x in schema.values()),'Original model parameter count differs')
    ids={n:id(p) for n,p in model.named_parameters()}
    require(getattr(model,'_causal_upscaler_adapter',None) is None,'Require original unwrapped architecture, not an already-wrapped model')
    require(not hasattr(model,'_production67_v5_load_verified'),'This loader does not support double initialization/load')
    result=model.load_state_dict(state,strict=True,assign=False)
    require(not result.missing_keys and not result.unexpected_keys,'Unexpected strict load report')
    require(ids=={n:id(p) for n,p in model.named_parameters()},'Loading replaced Parameter objects')
    _verify_tensor_values(model.state_dict(),schema,torch)
    # Provenance flag only. Never call initialize_aligned_past_ or shift kernels.
    model._aligned_past_initialized=True
    model._production67_v5_load_verified=True
    return ids


def enable_checked_block_checkpointing(student,source_paths):
    """Reuse the historical helper once; this is not decoder checkpointing."""
    import torch
    model=student.model
    require(getattr(model,'_production67_v5_load_verified',False) is True,'Load exact v5 first')
    require(getattr(student,'_production67_block_checkpointing',False) is False,'Block checkpointing already installed')
    require(student.stream_state()['live_token'] is False,'No active inference stream allowed')
    require(all(not m.training for m in model.modules()),'Keep all dropout/stochastic training modes disabled')
    blocks=[*model.in_blocks,*model.out_blocks]
    require(blocks and len({id(b) for b in blocks})==len(blocks),'Expected distinct original in/out blocks')
    require(all('forward' not in b.__dict__ for b in blocks),'Block already has a forward override; refuse double checkpoint wrapping')
    ids={n:id(p) for n,p in model.named_parameters()};names=tuple(model.state_dict());original=[b.forward for b in blocks]
    helper=_import_verified('train_upscaler_trial',source_paths)
    helper.enable_activation_checkpointing(model)
    require(ids=={n:id(p) for n,p in model.named_parameters()} and names==tuple(model.state_dict()),'Checkpoint helper changed parameter identity/names')
    require(all(b.forward is not old for b,old in zip(blocks,original)),'Checkpoint helper did not wrap every block')
    student._production67_block_checkpointing=True
    return {'installed_blocks':len(blocks),'helper_sha256':FROZEN_SOURCES['train_upscaler_trial.py'],
            'use_reentrant':False,'preserve_rng_state':False,'dropout':'eval/disabled','actual_recompute_gradient_parity_passed':False,
            'decoder_checkpointing_installed':False,'caller_must_not_mix_stream_or_mutate_weights_between_forward_backward':True}


def load_initial_v5_cpu(checkpoint_path,*,node,original_model,tensor_manifest_path,source_paths,activation_checkpointing=False):
    """CPU-only strict warm-start; original architecture construction is external.

    No optimizer/RNG is restored. The trusted fixed pickle is accepted only after
    its complete byte SHA. No fallback to a partial file or public export exists.
    """
    require(type(activation_checkpointing) is bool,'Explicit bool activation flag required')
    bundle=verify_source_bundle(source_paths);schema=load_tensor_manifest(tensor_manifest_path)
    import torch
    require(isinstance(original_model,torch.nn.Module),'Expected original model module')
    require(sha_file(inspect.getfile(type(original_model))) in UPSTREAM_SOURCE_SHAS,'Original architecture source fingerprint differs')
    require(sha_file(node.__file__) in UPSTREAM_SOURCE_SHAS,'Original node/normalization source differs')
    require(all(p.device.type=='cpu' and p.dtype==torch.float32 for p in original_model.parameters()),'Construct unchanged CPU FP32 architecture first')
    # weights_only=False follows the fixed historical tensor_load format; exact
    # known v5 SHA is mandatory before this trusted pickle is deserialized.
    with _verified_original_file(checkpoint_path) as handle:
        saved=torch.load(handle,map_location='cpu',weights_only=False)
    config=validate_saved_metadata(saved);state=saved.get('model_state_dict')
    _copy_strict_into_original(original_model,state,schema,torch)
    _import_verified('causal_upscaler',source_paths)
    geometry_module=_import_verified('h3_production67_geometry_v1',source_paths)
    wrapper=_import_verified('h3_conv3d_explicit_target_v1',source_paths)
    student=wrapper.ExplicitTargetCausalStudent(node,original_model,config,geometry=geometry_module.production67(),source_checkpoint_sha256=V5_SHA)
    cp=enable_checked_block_checkpointing(student,source_paths) if activation_checkpointing else {'enabled':False,'decoder_checkpointing_installed':False}
    _verify_tensor_values(student.original_state_dict(),schema,torch)
    verify_source_bundle(source_paths)
    receipt={'scope':'actual_CPU_load_only_no_forward_or_optimizer','original_checkpoint_sha256':V5_SHA,'original_checkpoint_file_verified':True,
        'selected_step':1800,'tensor_count':EXPECTED_TENSORS,'parameter_count':EXPECTED_PARAMETERS,
        'strict_names_shapes_dtype_values_passed':True,'parameter_identity_preserved':True,'aligned_initialization_called':False,
        'effective_input':[1,24,67,44,80],'explicit_target':[67,68,120],'total_latent_tokens':67,
        'historical_adapter_scale':2.0,'historical_scale_is_not_current_geometry':True,'frozen_sources':bundle,
        'block_checkpointing':cp,'optimizer_or_rng_restored':False,'decoder_gradient_or_memory_preflight_passed':False,
        'actual345M_forward_stream_HF32_or_training_passed':False,'formal_training_allowed':False}
    return student,receipt
