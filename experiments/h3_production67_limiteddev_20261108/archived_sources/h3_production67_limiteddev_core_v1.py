"""User-authorized limited-dev paired2000 over reviewed production67 numerical path.

No factory, launcher, data generation or device-mode mutation. Imported helpers
remain byte-pinned. Formal execution requires a separately reviewed actual-data
audit, fresh paired preflight and external admission; this module issues none.
"""
from __future__ import annotations
import copy
import hashlib
import inspect
import json
import math
import os
import random
import time
from pathlib import Path
import torch
import h3_production67_train_step_v1 as core
import h3_production67_deterministic_restore_core_v1 as old
import h3_production67_limiteddev_data_v1 as data

SCHEMA = 'h3-production67-limiteddev-paired2000-checkpoint-v1'
PROTOCOL_SCHEMA = 'h3-production67-limiteddev-paired2000-protocol-v1'
SEED, UPDATES, SAVE_EVERY, VALIDATE_EVERY = 20261107, 2000, 100, 200
CORE_SHA = '32bd9154fdbae7b9e5cb8ac16029014da3dae9227d276a3e90d5f37ea7877997'
DATA_SHA = 'fe2a6e20c3a4779a73a947001f792896576d9d43b22cf29eddcaea9d3a9fcc75'
OLD_SHA = '837ed1ef87126dc6a6387208012502a193b0d6cbdf54cd1418982e8af91b3126'
PROVIDER_SHA = '79af08a57ec3c3c8743631a49aac66ecb0628a3eb1bd90e7a486862d73ccd752'
MODE = {'deterministic_algorithms': True, 'warn_only': False}
PRECISION = {'dtype': 'float32', 'conv_hf32': False, 'matmul_hf32': False,
             'float32_matmul_precision': 'highest'}
# Actual aliases, not reimplementations of numerical/state helpers.
require, file_sha, canonical_sha, atomic_json = old.require, old.file_sha, old.canonical_sha, old.atomic_json
cpu_tree, exact_tree, capture_rng, restore_rng = old.cpu_tree, old.exact_tree, old.capture_rng, old.restore_rng
optimizer_counters, deterministic_mode, check_frozen, precision = old.optimizer_counters, old.deterministic_mode, old.check_frozen, old.precision
METRICS = ('hr_pixel_l1', 'hr_lpips', 'teacher_pixel_l1', 'teacher_lpips',
           'teacher_motion_l1', 'teacher_delta_l1')


def check_sources():
    for module, name, sha in ((core, 'h3_production67_train_step_v1.py', CORE_SHA),
                             (old, 'h3_production67_deterministic_restore_core_v1.py', OLD_SHA)):
        path = Path(inspect.getfile(module)).resolve()
        require(path == Path(__file__).resolve().with_name(name) and file_sha(path) == sha,
                'Frozen numerical helper origin/SHA changed')
    require(Path(inspect.getfile(data)).resolve() == Path(__file__).resolve().with_name('h3_production67_limiteddev_data_v1.py') and
            file_sha(inspect.getfile(data)) == DATA_SHA, 'Dataset adapter origin/SHA changed')
    return {'train_step': CORE_SHA, 'state_helpers': OLD_SHA,
            'dataset_adapter': file_sha(inspect.getfile(data)), 'paired2000_core': file_sha(__file__)}


def build_paired_schedule(sample_ids, recipe):
    check_sources()
    require(sample_ids and len(set(sample_ids)) == len(sample_ids), 'Unique ordered train IDs required')
    result = recipe.paired_schedule(sample_ids, core.Production67ScheduleSampler(),
        steps=UPDATES, seed=SEED, lr=1e-5, min_lr=1e-6, warmup=100, full_every=4)
    require([r['step'] for r in result] == list(range(1, UPDATES + 1)), 'Schedule continuity differs')
    for i, row in enumerate(result):
        require(row['sample_id'] in sample_ids and row['spec']['region']['spatial']['spatial_mode'] ==
                ('full_spatial' if i % 4 == 0 else 'stable_tile_interior'), 'Sample/full-spatial schedule differs')
    return result


def build_selection_plan(dev_ids):
    """Fixed sparse selection regions; explicitly not the final full226 dev gate."""
    require(dev_ids and len(dev_ids) == len(set(dev_ids)), 'Unique fixed dev order required')
    full = core.Production67ScheduleSampler().full_specs
    return {'scope': 'fixed_nonblend_regions_for_selection_not_full226_quality_gate',
            'samples': [{'sample_id': sid, 'region': copy.deepcopy(full[i % len(full)]),
                         'lpips_packed_positions': list(range(len(full[i % len(full)]['rgb_frame_indices'])))}
                        for i, sid in enumerate(dev_ids)]}


def candidate_protocol(dataset, schedule, selection_plan, *, evaluator_source_sha256, initial_tensor_schema_sha256):
    loader = core.strict_loader()
    for sha in (evaluator_source_sha256, initial_tensor_schema_sha256):
        loader.sha_value(sha)
    return {'schema': PROTOCOL_SCHEMA, 'status': 'explicit_user_authorized_limiteddev_2train_3dev',
            'formal_training_allowed': False, 'source_sha256': check_sources(),
            'registry_sha256': dataset['descriptor']['sha256'],
            'cohort_science_sha256': copy.deepcopy(dataset['cohort_science_sha256']),
            'generation_mode_semantics': copy.deepcopy(dataset['generation_mode_semantics']),
            'train_order': [r['sample_id'] for r in data.rows_for(dataset, 'train')],
            'dev_order': [r['sample_id'] for r in data.rows_for(dataset, 'dev')],
            'test_count_required': 0, 'dev_count_required': 3, 'train_count_required': 2, 'test_access': False,
            'schedule_sha256': canonical_sha(schedule), 'selection_plan_sha256': canonical_sha(selection_plan),
            'selection_evaluator_sha256': evaluator_source_sha256,
            'initial_tensor_schema_sha256': initial_tensor_schema_sha256,
            'initializer_checkpoint_sha256': core.V5_SHA, 'engineering_initializer_allowed': False,
            'arms': ['control', 'teacher'], 'seed': SEED, 'updates_per_arm': UPDATES,
            'save_every': SAVE_EVERY, 'selection_every': VALIDATE_EVERY,
            'precision': PRECISION, 'mode': MODE, 'direct_pixel_scope': '166_nonblend_of_226',
            'blend_frames_in_final_evaluation': 60,
            'selection': 'common_step0_teacher_score_HR_guard_best0_allowed',
            'full226_dev_gate_and_test_access': 'separate_not_implemented_here'}


def checked_external(desc):
    require(set(desc) == {'path', 'bytes', 'sha256'}, 'Exact external receipt descriptor required')
    p = Path(desc['path'])
    require(p.is_absolute() and p.resolve() == p and p.is_file() and
            p.stat().st_size == desc['bytes'] and file_sha(p) == desc['sha256'], 'External receipt identity differs')
    return json.loads(p.read_bytes())


def validate_admission(protocol, dataset, admission):
    require(admission['scope']=='explicit_user_authorized_production67_limiteddev_2train_3dev' and admission['protocol_sha256']==canonical_sha(protocol),'Exact new limited-dev protocol required')
    require(admission['physical_NPU_locks_held'] is True and admission['first_step_admission']=='full_spatial_loss_backward_then_finite_nonzero_clip_then_optimizer','Device/same-graph admission missing')
    data.validate_audit(dataset,checked_external(admission['payload_audit']))
    baseline=checked_external(admission['common_step0_dev'])
    require(baseline['schema']=='h3-production67-limiteddev-common-step0-v1' and baseline['initializer_checkpoint_sha256']==core.V5_SHA and baseline['registry_sha256']==protocol['registry_sha256'] and baseline['selection_plan_sha256']==protocol['selection_plan_sha256'] and baseline['selection_evaluator_sha256']==protocol['selection_evaluator_sha256'] and baseline['dev_order']==protocol['dev_order'] and len(baseline['observation']['samples'])==3 and baseline['fresh_v5_identity']['all_initial_values_match_frozen_v5'] is True,'Actual shared fresh-v5 3dev baseline missing')
    require(protocol['test_access'] is False and protocol['dev_count_required']==3 and protocol['train_count_required']==2 and protocol['test_count_required']==0,'Limited protocol changed')
    return copy.deepcopy(baseline['metrics'])


def selection_metrics(result, *, protocol, step, recipe):
    require(result['step'] == step and result['registry_sha256'] == protocol['registry_sha256'] and
            result['selection_plan_sha256'] == protocol['selection_plan_sha256'] and
            [x['sample_id'] for x in result['samples']] == protocol['dev_order'], 'Dev evaluation provenance/order differs')
    rows = [x['metrics'] for x in result['samples']]
    require(rows and all(set(x) == set(METRICS) and all(math.isfinite(float(v)) and float(v) >= 0
            for v in x.values()) for x in rows), 'Dev metrics missing, extra or nonfinite')
    return recipe.selection_fields({k: sum(float(x[k]) for x in rows) / len(rows) for k in METRICS})


def validate_selection(state, updates, *, recipe):
    require(set(state) == {'baseline', 'history', 'best_step', 'best_score', 'best_checkpoint'}, 'Selection schema differs')
    steps = [h['step'] for h in state['history']]
    require(steps == [0] + list(range(VALIDATE_EVERY, updates + 1, VALIDATE_EVERY)), 'Selection history is incomplete')
    require(state['history'][0]['metrics'] == state['baseline'] and state['best_step'] in steps and
            math.isfinite(float(state['best_score'])), 'Initial/best selection identity differs')
    require(state['best_checkpoint']['step'] == state['best_step'] and
            state['best_checkpoint']['initializer_checkpoint_sha256'] == core.V5_SHA, 'Best checkpoint lineage differs')
    p = Path(state['best_checkpoint']['path'])
    require(p.is_absolute() and p.resolve() == p and p.is_file() and
            p.stat().st_size == state['best_checkpoint']['bytes'] and
            file_sha(p) == state['best_checkpoint']['sha256'], 'Best checkpoint bytes missing or changed')
    best = None
    for item in state['history']:
        metrics = item['metrics']
        require(all(k in metrics and math.isfinite(float(metrics[k])) and float(metrics[k]) >= 0
                    for k in METRICS), 'Selection contains invalid base metrics')
        require(metrics == recipe.selection_fields({k: metrics[k] for k in METRICS}),
                'Derived selection scores differ from frozen recipe')
        guard = bool(recipe.hr_guard(metrics, state['baseline']))
        require(item['guard_pass'] is guard, 'History guard differs from frozen recipe')
        if best is None or (guard and metrics['selection_score'] < best['metrics']['selection_score']):
            best = item
    require(best['guard_pass'] is True and best['step'] == state['best_step'] and
            best['metrics']['selection_score'] == state['best_score'],
            'Best is not the earliest minimum of guard-passing history')


def save_best(path, *, model, step, arm, protocol_sha, metrics):
    path = Path(path); require(not path.exists(), 'Never overwrite selected weights')
    value = {'schema': 'h3-production67-limiteddev-selected-weights-v1', 'arm': arm, 'step': step,
             'protocol_sha256': protocol_sha, 'initializer_checkpoint_sha256': core.V5_SHA,
             'metrics': copy.deepcopy(metrics), 'model': cpu_tree(model.state_dict())}
    with path.open('xb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': file_sha(path),
            'step': step, 'arm': arm, 'protocol_sha256': protocol_sha, 'initializer_checkpoint_sha256': core.V5_SHA}


def observe_selection(state, metrics, *, step, model, arm, protocol_sha, output, recipe):
    require(step == 0 if state is None else step == state['history'][-1]['step'] + VALIDATE_EVERY,
            'Selection observations must be continuous including step0')
    if state is None:
        state = {'baseline': copy.deepcopy(metrics), 'history': [], 'best_step': 0,
                 'best_score': metrics['selection_score'], 'best_checkpoint': None}
    else:
        state = copy.deepcopy(state)
    guard = bool(recipe.hr_guard(metrics, state['baseline']))
    state['history'].append({'step': step, 'metrics': copy.deepcopy(metrics), 'guard_pass': guard})
    if step == 0 or (guard and metrics['selection_score'] < state['best_score']):
        state.update(best_step=step, best_score=metrics['selection_score'],
                     best_checkpoint=save_best(Path(output) / f'best_step_{step:04d}.pt', model=model,
                         step=step, arm=arm, protocol_sha=protocol_sha, metrics=metrics))
    validate_selection(state, step, recipe=recipe)
    return state


def save_checkpoint(path, *, model, optimizer, device, updates, arm, protocol_sha,
                    schedule_sha, registry_sha, selection, durable_steps, recipe):
    path = Path(path); require(not path.exists(), 'Never overwrite formal checkpoints')
    require(arm in ('control', 'teacher') and type(updates) is int and 0 <= updates <= UPDATES,
            'Wrong arm/update counter')
    optimizer_counters(optimizer, model.parameters(), updates); deterministic_mode()
    validate_selection(selection, updates, recipe=recipe)
    require([r['step'] for r in durable_steps] == list(range(1, updates + 1)), 'Durable step prefix incomplete')
    require(selection['best_checkpoint']['arm'] == arm and selection['best_checkpoint']['protocol_sha256'] == protocol_sha,
            'Selection belongs to another arm/protocol')
    value = {'schema': SCHEMA, 'arm': arm, 'completed_updates': updates,
             'protocol_sha256': protocol_sha, 'schedule_sha256': schedule_sha, 'registry_sha256': registry_sha,
             'initializer_checkpoint_sha256': core.V5_SHA, 'mode': MODE, 'precision': PRECISION,
             'model': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()),
             'rng': capture_rng(device), 'selection': copy.deepcopy(selection),
             'durable_steps': copy.deepcopy(durable_steps), 'durable_steps_sha256': canonical_sha(durable_steps)}
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('xb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    result = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': file_sha(path),
              'completed_updates': updates, 'arm': arm, 'protocol_sha256': protocol_sha,
              'schedule_sha256': schedule_sha, 'registry_sha256': registry_sha}
    atomic_json(path.with_suffix('.receipt.json'), result)
    return result


def restore_checkpoint(receipt, *, model, optimizer, device, arm, protocol_sha, schedule_sha, registry_sha, recipe):
    deterministic_mode()
    p = Path(receipt['path'])
    require(p.is_absolute() and p.resolve() == p and p.is_file() and p.stat().st_size == receipt['bytes'] and
            file_sha(p) == receipt['sha256'], 'Checkpoint identity differs; do not deserialize')
    saved = torch.load(p, map_location='cpu', weights_only=False)
    require(saved['schema'] == SCHEMA and saved['initializer_checkpoint_sha256'] == core.V5_SHA and
            saved['mode'] == MODE and saved['precision'] == PRECISION, 'Engineering/foreign checkpoint is not a formal resume')
    for k, v in {'arm': arm, 'protocol_sha256': protocol_sha, 'schedule_sha256': schedule_sha, 'registry_sha256': registry_sha}.items():
        require(saved[k] == receipt[k] == v, 'Mixed arm/protocol/schedule/dataset resume')
    updates = saved['completed_updates']
    require(type(updates) is int and 0 <= updates < UPDATES and updates == receipt['completed_updates'], 'Bad resume counter')
    validate_selection(saved['selection'], updates, recipe=recipe)
    require(saved['selection']['best_checkpoint']['arm'] == arm and saved['selection']['best_checkpoint']['protocol_sha256'] == protocol_sha,
            'Selection belongs to another arm/protocol')
    require([r['step'] for r in saved['durable_steps']] == list(range(1, updates + 1)) and
            canonical_sha(saved['durable_steps']) == saved['durable_steps_sha256'], 'Durable prefix differs')
    expected = model.state_dict()
    require(set(saved['model']) == set(expected) and all(v.shape == expected[k].shape and
            v.dtype == expected[k].dtype == torch.float32 and bool(torch.isfinite(v).all())
            for k, v in saved['model'].items()), 'Checkpoint model schema invalid')
    ids = {n: id(v) for n, v in model.named_parameters()}
    model.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
    require(ids == {n: id(v) for n, v in model.named_parameters()}, 'Resume replaced parameters')
    optimizer_counters(optimizer, model.parameters(), updates)
    require(exact_tree(model.state_dict(), saved['model']) and exact_tree(optimizer.state_dict(), saved['optimizer']),
            'Live restored model/optimizer not exact')
    restore_rng(saved['rng'], device)
    require(file_sha(p) == receipt['sha256'], 'Checkpoint changed while restoring')
    return updates, copy.deepcopy(saved['selection']), copy.deepcopy(saved['durable_steps']), {
        'model_bitwise_exact': True, 'optimizer_bitwise_exact': True, 'CPU_and_selected_device_RNG_exact': True,
        'checkpoint': copy.deepcopy(receipt), 'selection_state_exact': True, 'durable_prefix_exact': True}


def load_bound_example(dataset, sample_id):
    pair, teacher, contract = data.load_bundles(dataset, sample_id, split='train')
    loader = core.strict_loader(); values = loader.torch_arrays(pair)
    example = {'draft': values['draft'].unsqueeze(0), 'reference': values['reference_hr'].unsqueeze(0),
               'teacher': loader.torch_teacher(teacher).unsqueeze(0), 'flow': values['flow'], 'valid': values['flow_valid'],
               'pair_bundle': pair, 'teacher_bundle': teacher, 'contract': contract,
               'scope': 'candidate_checked_cache_not_training_READY'}
    core.validate_example_tensors(example)
    return example


def run_arm(*, student, provider, lpips_fn, recipe, device, dataset, schedule, selection_plan,
            evaluate_dev, output, arm, protocol, admission, initial_tensor_schema, resume_receipt=None):
    """Fresh originalv5 arm or same-protocol new checkpoint, never engineering init.

    evaluate_dev is a SHA-pinned caller-supplied no-grad sparse-region evaluator;
    it must use the exact selection_plan and data.rows_for(dataset,'dev'). A new
    full226 evaluator and final16dev/test gate are deliberately outside this core.
    """
    check_sources(); device = torch.device(device); prec = precision(device); deterministic_mode()
    require(arm in ('control', 'teacher'), 'Unknown arm')
    require(schedule == build_paired_schedule([r['sample_id'] for r in data.rows_for(dataset, 'train')], recipe),
            'Shared materialized2000 schedule differs')
    require(selection_plan == build_selection_plan([r['sample_id'] for r in data.rows_for(dataset, 'dev')]),
            'Frozen selection regions differ')
    require(protocol == candidate_protocol(dataset, schedule, selection_plan,
            evaluator_source_sha256=protocol['selection_evaluator_sha256'],
            initial_tensor_schema_sha256=canonical_sha(initial_tensor_schema)), 'Exact candidate protocol differs')
    baseline = validate_admission(protocol, dataset, admission)
    require(file_sha(inspect.getsourcefile(evaluate_dev)) == protocol['selection_evaluator_sha256'], 'Dev evaluator source changed')
    require(file_sha(inspect.getsourcefile(type(provider))) == PROVIDER_SHA, 'Decoder provider source changed')
    require(provider.allow_full_spatial_grad is True and provider.checkpoint_decode is True,
            'Full spatial/checkpointed decode must be explicitly admitted')
    require(student.source_checkpoint_sha256 == core.V5_SHA and
            student.model_config['kind'] == 'h3_causal_conv3d_explicit_target_v1' and
            getattr(student, '_production67_block_checkpointing', False) is True, 'Fresh reviewed v5 wrapper required')
    student.eval()
    require(all(p.requires_grad and p.dtype == torch.float32 and p.device == device for p in student.parameters()), 'Wrong student dtype/device/trainability')
    require(len(tuple(student.parameters())) == 322 and sum(p.numel() for p in student.parameters()) == 345280216, 'Wrong actual student size')
    st = student.original_state_dict(); require(set(st) == set(initial_tensor_schema), 'Initial v5 schema differs')
    for name, t in st.items():
        c = t.detach().cpu().contiguous()
        require(list(c.shape) == initial_tensor_schema[name]['shape'] and
                hashlib.sha256(memoryview(c.numpy()).cast('B')).hexdigest() == initial_tensor_schema[name]['tensor_sha256'], 'Fresh v5 value differs: ' + name)
    check_frozen(provider, lpips_fn)
    output = Path(output); require(output.is_absolute(), 'Absolute exclusive attempt output required'); output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(SEED); random.seed(SEED); torch.npu.manual_seed(SEED)
    optimizer = core.fresh_adamw(student.parameters(), lr=1e-5); optimizer_counters(optimizer, student.parameters(), 0)
    protocol_sha, sched_sha = canonical_sha(protocol), canonical_sha(schedule)
    start, selection, durable, restored = 0, None, [], None
    if resume_receipt is not None:
        start, selection, durable, restored = restore_checkpoint(resume_receipt, model=student.model, optimizer=optimizer,
            device=device, arm=arm, protocol_sha=protocol_sha, schedule_sha=sched_sha, registry_sha=protocol['registry_sha256'], recipe=recipe)
        require(selection['baseline'] == baseline, 'Resumed selection baseline differs from common v5')
        require(all({k: r[k] for k in ('step', 'sample_id', 'spec', 'lr')} == schedule[i]
                    for i, r in enumerate(durable)), 'Saved durable sample/spec/LR prefix differs')
    status_path = output / 'attempt.json'
    state = {'schema': 'h3-production67-limiteddev-paired2000-attempt-v1', 'status': 'running', 'arm': arm,
             'protocol_sha256': protocol_sha, 'schedule_sha256': sched_sha, 'registry_sha256': protocol['registry_sha256'],
             'precision': prec, 'mode': MODE, 'completed_updates': start, 'optimizer_updates_this_attempt': 0,
             'first_full_spatial_backward_admission': 'not_run' if start == 0 else 'inherited_saved_prefix',
             'optimizer_step_started': False, 'resume': restored, 'admission': copy.deepcopy(admission),
             'checkpoints': [], 'steps': [], 'selection': selection, 'full226_dev_quality_gate_passed': False}
    atomic_json(status_path, state)
    before_frozen = [(id(t), t._version) for m in (provider.vae, lpips_fn) for t in list(m.parameters()) + list(m.buffers())]

    def evaluate(step):
        rng = capture_rng(device)
        with torch.no_grad():
            result = evaluate_dev(student=student, provider=provider, lpips_fn=lpips_fn, dataset=dataset,
                                  selection_plan=selection_plan, step=step)
        restore_rng(rng, device)
        deterministic_mode(); require(precision(device) == prec, 'Dev evaluation changed mode/precision')
        folder=output/'dev_observations';folder.mkdir(exist_ok=True)
        with (folder/f'step{step:04d}.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
        return selection_metrics(result, protocol=protocol, step=step, recipe=recipe)

    try:
        if resume_receipt is None:
            # Exact322 fresh-v5 values checked above; reuse the one shared actual baseline.
            metrics0=copy.deepcopy(baseline)
            with (output/'common_step0_reference.json').open('x') as f:json.dump(admission['common_step0_dev'],f,indent=2)
            selection = observe_selection(None, metrics0, step=0, model=student.model, arm=arm,
                protocol_sha=protocol_sha, output=output, recipe=recipe)
            state['selection'] = selection
            state['checkpoints'].append(save_checkpoint(output / 'checkpoint_step_0000.pt', model=student.model,
                optimizer=optimizer, device=device, updates=0, arm=arm, protocol_sha=protocol_sha,
                schedule_sha=sched_sha, registry_sha=protocol['registry_sha256'], selection=selection, durable_steps=durable, recipe=recipe))
            atomic_json(status_path, state)
        with (output / 'steps.jsonl').open('x') as log:
            for row in schedule[start:UPDATES]:
                deterministic_mode(); require(precision(device) == prec, 'Precision changed before update')
                example = load_bound_example(dataset, row['sample_id'])
                example = core.move_example(example, device=device)
                for g in optimizer.param_groups:
                    g['lr'] = row['lr']
                torch.npu.synchronize(device); torch.npu.reset_peak_memory_stats(device); beg = time.monotonic()
                loss,terms,pixels=core.loss_for_example(student,example,row['spec']['region'],lpips_packed_positions=row['spec']['lpips_packed_positions'],temporal_scope='nonblend_only',decode_region=provider.decode_region,lpips_fn=lpips_fn,recipe=recipe,arm=arm)
                metrics={'loss':float(loss.detach()),**{k:float(v.detach()) for k,v in {**terms,**pixels}.items()}}
                optimizer.zero_grad(set_to_none=True);loss.backward()
                params=tuple(student.parameters());require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params),'Missing/nonfinite gradient before optimizer')
                norm=torch.nn.utils.clip_grad_norm_(params,1.,foreach=False,error_if_nonfinite=True);require(float(norm)>0,'Zero gradient before optimizer')
                state['first_full_spatial_backward_admission']='passed_actual_same_graph' if row['step']==1 else state['first_full_spatial_backward_admission']
                state['optimizer_step_started']=True;state['optimizer_step_pending_update']=row['step'];atomic_json(status_path,state)
                optimizer.step();torch.npu.synchronize(device);state['optimizer_step_started']=False
                state['completed_updates']=row['step'];state['optimizer_updates_this_attempt']+=1;state['last_confirmed_optimizer_update']=row['step'];atomic_json(status_path,state)
                update={'gradient_norm_before_clip':float(norm),'optimizer_updates':1,'formal_training_preflight_passed':False}
                optimizer_counters(optimizer,student.parameters(),row['step']);check_frozen(provider,lpips_fn);require(precision(device)==prec,'Precision changed during updates');deterministic_mode()
                rec={**copy.deepcopy(row),**metrics,**update,'seconds':time.monotonic()-beg,'peak_allocated':torch.npu.max_memory_allocated(device),'peak_reserved':torch.npu.max_memory_reserved(device)}
                log.write(json.dumps(rec,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno());state['steps'].append(rec)
                del loss,terms,pixels,example
                durable.append(rec)
                if row['step'] % VALIDATE_EVERY == 0:
                    selection = observe_selection(selection, evaluate(row['step']), step=row['step'],
                        model=student.model, arm=arm, protocol_sha=protocol_sha, output=output, recipe=recipe)
                    state['selection'] = selection
                if row['step'] % SAVE_EVERY == 0 or row['step'] == UPDATES:
                    state['checkpoints'].append(save_checkpoint(output / f'checkpoint_step_{row["step"]:04d}.pt',
                        model=student.model, optimizer=optimizer, device=device, updates=row['step'], arm=arm,
                        protocol_sha=protocol_sha, schedule_sha=sched_sha, registry_sha=protocol['registry_sha256'],
                        selection=selection, durable_steps=durable, recipe=recipe))
                atomic_json(status_path, state)
        after_frozen = [(id(t), t._version) for m in (provider.vae, lpips_fn) for t in list(m.parameters()) + list(m.buffers())]
        require(before_frozen == after_frozen, 'Frozen decoder/LPIPS changed')
        require(state['completed_updates'] == UPDATES and state['optimizer_updates_this_attempt'] == UPDATES - start,
                'Actual update count differs')
        validate_selection(selection, UPDATES, recipe=recipe); deterministic_mode(); require(precision(device) == prec, 'Completion precision changed')
        state['status'] = 'completed2000_awaiting_separate_full226_dev_gate'
        atomic_json(status_path, state)
        return state
    except BaseException as e:
        state['partial_optimizer_update_possible'] = state.get('optimizer_step_started', False)
        state.update(status='failed_no_automatic_retry', exception_type=type(e).__name__, exception=str(e))
        atomic_json(status_path, state)
        raise
