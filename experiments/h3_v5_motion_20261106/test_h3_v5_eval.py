#!/usr/bin/env python3
"""CPU-only evaluator unit checks; no checkpoint/data load, no NPU access.

Loads selected pure functions from their AST so the evaluator's runtime imports,
main(), checkpoint checks and tensor datasets are never executed. This is NOT
--stage check and does not verify two completed training runs. Use --without-torch
for local stdlib checks; the full command requires an existing CPU torch install.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names), 'Missing pure function')
    require(all(not n.decorator_list for n in nodes), 'Selected function has decorator')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=Path(__file__).parent)
    parser.add_argument('--without-torch', action='store_true')
    args = parser.parse_args()
    evaluator = args.experiment / 'evaluate_h3_v5_motion.py'
    helper = args.experiment / 'h3_v5_motion.py'
    ns = dict(require=require, math=math, statistics=statistics, hashlib=hashlib,
              LABELS=('v5', 'control', 'motion', 'reference'), ARMS=('control', 'motion'), THRESHOLD=.05)
    extract(evaluator, ['frame_regions', 'transition_pair_groups', 'aggregate', 'aggregate_transition_groups', 'gate', 'run_test'], ns)
    regions = ns['frame_regions']()
    require({k: len(v) for k, v in regions.items()} ==
            dict(full120=120, prefix102=102, tail18=18, blend=30, nonblend=90), 'Region lengths')
    require(set(regions['blend']).isdisjoint(regions['nonblend']) and
            set(regions['blend']) | set(regions['nonblend']) == set(range(120)), 'Blend coverage')
    expected_pairs = dict(full120=119, prefix102=101, tail18=17, blend=24, nonblend=83)
    require({k: sum(b-a == 1 for a, b in zip(v, v[1:])) for k, v in regions.items()} == expected_pairs,
            'Region pairing must respect actual RGB gaps')
    pair_groups = ns['transition_pair_groups']()
    touching = {tuple(pair) for pair in pair_groups['blend_touching_pairs']}
    nonblend = {tuple(pair) for pair in pair_groups['nonblend_only_pairs']}
    require(len(touching) == 36 and len(nonblend) == 83 and touching.isdisjoint(nonblend) and
            touching | nonblend == {(i, i+1) for i in range(119)}, 'Transition partition36+83=119')
    require((16, 17) in touching and (21, 22) in touching and (15, 16) in nonblend,
            'Blend entry/exit pairs must be included')
    blend_internal = {(a, b) for a, b in touching if a in regions['blend'] and b in regions['blend']}
    require(len(touching-blend_internal) == 12, 'Exactly12 blend entry/exit transitions')
    base = dict(global_score=.14, pixel_l1=.1, lpips=.2, motion_l1=.3, delta_l1=.4)
    good = dict(base, motion_l1=.29, delta_l1=.39)
    def decision(motion, control=None):
        means = dict(v5=copy.deepcopy(base), control=copy.deepcopy(control or good), motion=motion)
        return ns['gate']({'aggregate': {'regions': {'full120': {'mean': means}}}})
    require(decision(good)['test_unlocked'], 'Strict improvements pass')
    require(not decision(base)['test_unlocked'], 'Unchanged baseline must fail strict improvements')
    for key, factor in [('global_score', 1.01), ('pixel_l1', 1.02), ('lpips', 1.02)]:
        at_limit = dict(good, **{key: base[key]*factor})
        require(decision(at_limit)['test_unlocked'], key+' exact inclusive threshold')
        beyond = dict(good, **{key: math.nextafter(base[key]*factor, math.inf)})
        require(not decision(beyond)['test_unlocked'], key+' excess must fail')
    for key in ('motion_l1', 'delta_l1'):
        require(not decision(dict(good, **{key: base[key]}))['test_unlocked'], key+' requires strict improvement')
    require(decision(good, base)['test_unlocked'] and not decision(good, base)['arms']['control']['passed'],
            'Control failure does not reject passing motion')
    require(not decision(base, good)['test_unlocked'] and decision(base, good)['arms']['control']['passed'],
            'Passing control must not unlock failing motion')
    # Equal clip means: ROI size is deliberately different, and must not become a clip weight.
    entries = {}
    for ident, value, roi in [('small', 1., .01), ('large', 9., .99)]:
        entries[ident] = {'variants': {label: {'regions': {'full120':
            dict(pixel_l1=value, motion_roi_fraction=roi, motion_l1=value)}} for label in ns['LABELS'][:-1]},
            'temporal_transition_groups': {group: {'pair_indices': pairs, 'variants': {label:
                dict(motion_l1=value, delta_l1=value, motion_roi_fraction=roi) for label in ns['LABELS'][:-1]}}
                for group, pairs in pair_groups.items()}}
    aggregate = ns['aggregate'](entries, ['full120'])
    require(aggregate['regions']['full120']['mean']['motion']['motion_l1'] == 5., 'Equal-clip ROI aggregation')
    require(aggregate['regions']['full120']['mean']['motion']['motion_roi_fraction'] == .5, 'Equal-clip coverage')
    transition_aggregate = ns['aggregate_transition_groups'](entries, pair_groups)
    require(all(group['mean']['motion']['motion_l1'] == 5. for group in transition_aggregate['groups'].values()),
            'Transition aggregation must weight clips equally')
    # A failed gate raises before even contract verification, device selection or a split loader.
    forbidden_calls = []
    def forbidden(*_args, **_kwargs):
        forbidden_calls.append(True)
        raise RuntimeError('Forbidden post-gate operation reached')
    ns.update(verify_contract=forbidden, load_split=forbidden, select_device=forbidden)
    for dev in ({'gate': {'test_unlocked': False}, 'status': 'dev_gate_failed'},
                {'gate': {'test_unlocked': True}, 'status': 'dev_gate_failed'}):
        try:
            ns['run_test'](None, dev)
        except AssertionError:
            pass
        else:
            raise AssertionError('Failed dev gate did not block test stage')
    require(not forbidden_calls, 'Failed gate touched later operations')
    try:
        ns['run_test'](None, {'gate': {'test_unlocked': True}, 'status': 'dev_passed'})
    except RuntimeError as error:
        require(str(error) == 'Forbidden post-gate operation reached', 'Unexpected pass path')
    else:
        raise AssertionError('Passing gate did not proceed to verification')
    require(len(forbidden_calls) == 1, 'Unexpected pass-path operations')
    checks = ['region coverage and actual-time adjacency', '36+83 transition partition including12 blend boundary pairs', 'five dev-gate conditions and inclusive/strict boundaries',
              'motion-only test unlock', 'equal-clip aggregation', 'failed gate blocks all post-gate operations']
    if not args.without_torch:
        import torch
        torch.set_num_threads(2)
        ns['torch'] = torch
        extract(helper, ['_sequence_pair', '_time_indices', 'rgb_motion_losses'], ns)
        extract(evaluator, ['target_mask_metadata', 'rgb_metrics', 'rgb_transition_metrics'], ns)
        def fake_lpips(_model, output, reference, _device, _batch):
            return (output.float()-reference.float()).abs().mean((1, 2, 3)).div(255).tolist()
        ns['lpips_frames'] = fake_lpips
        # One changing pixel among four. All 119 real adjacent pairs, including118->119, must count.
        reference = torch.zeros(120, 2, 2, 3, dtype=torch.uint8)
        reference[1::2, 0, 0, :] = 26
        metadata = ns['target_mask_metadata'](reference, list(range(120)))
        require(metadata['adjacent_pairs'] == 119 and metadata['motion_pair_pixels'] == 119 and
                metadata['total_pair_pixels'] == 476, 'Full120 GT ROI mask')
        ref = reference.float().permute(3, 0, 1, 2)[None]/255.
        pred = ref + .1
        motion = ns['rgb_motion_losses'](pred, ref, list(range(120)))
        require(abs(float(motion['motion_l1'])-.1) < 1e-6 and abs(float(motion['delta_l1'])) < 1e-7,
                'Pair-weighted endpoint ROI / constant-offset delta')
        require(float(motion['adjacent_pair_count']) == 119 and float(motion['motion_roi_fraction']) == .25,
                'All full120 pairs included')
        # A real RGB118->119 pair is valid even when raw indices skip19->23.
        last_pair = ns['rgb_motion_losses'](pred[:, :, -2:], ref[:, :, -2:], [118, 119], [19, 23])
        require(float(last_pair['adjacent_pair_count']) == 1, 'Last RGB pair was incorrectly excluded')
        # Equality at the floating helper threshold is included; static pixels stay excluded.
        threshold_target = torch.zeros(1, 3, 2, 1, 2)
        threshold_target[:, :, 1, 0, 0] = .05
        threshold_result = ns['rgb_motion_losses'](threshold_target+.1, threshold_target, [0, 1])
        require(float(threshold_result['motion_roi_fraction']) == .5, '>=.05 boundary')
        output = reference.clone(); output[-1, 0, 0, :] = 0
        metrics = ns['rgb_metrics'](output, reference, None, torch.device('cpu'), regions)
        direct = output.float()/255. - reference.float()/255.
        expected_temporal = float((direct[1:]-direct[:-1]).square().mean())
        require(abs(metrics['regions']['full120']['temporal_difference_mse']-expected_temporal) < 1e-10,
                'Pixel temporal MSE differs from direct actual-adjacent definition')
        for name, count in expected_pairs.items():
            require(metrics['regions'][name]['adjacent_pair_count'] == count, name+' RGB pairing')
            require(all(math.isfinite(v) for v in metrics['regions'][name].values()), name+' nonfinite value')
        transitions = ns['rgb_transition_metrics'](output, reference, pair_groups)
        weighted_mse = sum(transitions[name]['metrics']['temporal_difference_mse']*len(pairs)
                           for name, pairs in pair_groups.items())/119
        require(abs(weighted_mse-expected_temporal) < 1e-10, 'Partition must recover all119-pair MSE')
        # Direct group formula independently selects full-timeline endpoints, never subset neighbors.
        rgb_error = output.float()/255. - reference.float()/255.
        target_rgb = reference.float()/255.
        for name, pairs in pair_groups.items():
            left = [a for a, b in pairs]; right = [b for a, b in pairs]
            mask = (target_rgb[right]-target_rgb[left]).abs().mean(-1) >= .05
            err = rgb_error[right]-rgb_error[left]
            endpoint_roi = ((rgb_error[left].abs()+rgb_error[right].abs())*mask[..., None]).sum()/(2*3*mask.sum()).clamp_min(1)
            rec = transitions[name]['metrics']
            require(abs(rec['motion_l1']-float(endpoint_roi)) < 1e-7 and
                    abs(rec['delta_l1']-float(err.abs().mean())) < 1e-7 and
                    abs(rec['temporal_difference_mse']-float(err.square().mean())) < 1e-10,
                    'Direct transition endpoint formula differs: '+name)
            require(rec['adjacent_pair_count'] == len(pairs) and rec['motion_pair_pixels'] == len(pairs) and
                    rec['motion_roi_fraction'] == .25, 'Fixed target transition ROI coverage')
        # Distortion exactly at blend entry/exit is visible in touching-pair diagnostics.
        boundary_output = reference.clone(); boundary_output[17, 1, 1, :] = 255; boundary_output[22, 1, 1, :] = 255
        boundary = ns['rgb_transition_metrics'](boundary_output, reference, pair_groups)
        require(boundary['blend_touching_pairs']['metrics']['delta_l1'] > 0., 'Boundary artifact not detected')
        exact = ns['rgb_metrics'](reference, reference, None, torch.device('cpu'), regions)
        for rec in exact['regions'].values():
            for key in ('pixel_l1', 'lpips', 'motion_l1', 'delta_l1', 'temporal_difference_mse'):
                require(rec[key] == 0., 'Exact-output metric is nonzero: '+key)
        require(len(metrics['lpips_per_frame']) == 120 and all(v.device.type == 'cpu' for v in (ref, pred)), 'CPU-only coverage')
        checks += ['CPU120-frame fixed target ROI and endpoint normalization', 'last RGB pair despite raw-token gap',
                   '>=.05 threshold', 'pixel temporal MSE and region adjacency', 'explicit transition endpoint ROI/delta/MSE and boundary detection', 'exact-output zero errors']
    print(json.dumps(dict(status='passed', scope='Pure-function unit checks only; no training-run context/checkpoint/data/NPU/real LPIPS verified',
        torch_checks='skipped' if args.without_torch else 'passed_cpu', checks=checks,
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (evaluator, helper)}), indent=2))


if __name__ == '__main__':
    main()
