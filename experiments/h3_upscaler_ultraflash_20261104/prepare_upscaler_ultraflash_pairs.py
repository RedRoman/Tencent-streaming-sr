"""Plan, shard-encode, and finalize a larger fixed H3 causal-upscaler dataset.

This only prepares paired data; it does not implement a paper's training loss.
The three legacy clips are isolated. Known visual-issue clips are excluded from
all new splits. Workers share an immutable plan and never overwrite each other.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from data_utils import encode_pixels, load_video_vae, released_decoded_frames, select_device
from prepare_upscaler_5s_pairs import (
    FPS, SOURCE_FRAMES, LATENT_FRAMES, HEIGHT, WIDTH, VAE_CLIP_LENGTH,
    VAE_PADDED_INPUT_FRAMES, VAE_INTERNAL_PAD_FRAMES, RAW_DECODED_FRAMES,
    count_source_frames, read_first_120_strict, save_json, sha256)


LEGACY_IDS = ('VIDGEN5S_0336361', 'VIDGEN5S_0380588', 'VIDGEN5S_0515402')
SPLIT_COUNTS = {'train': 128, 'dev': 16, 'test': 16, 'legacy': 3}
SPLIT_ORDER = ('train', 'dev', 'test', 'legacy')


def quota(total, groups):
    """Capacity-aware largest-remainder allocation, at least one per category."""
    categories = sorted(category for category, values in groups.items() if values)
    if total > sum(len(groups[category]) for category in categories):
        raise ValueError('Insufficient distinct sources for requested split')
    minimum = 1 if total >= len(categories) else 0
    result = {category: minimum for category in categories}
    remaining = total-sum(result.values())
    capacities = {category: len(groups[category])-result[category] for category in categories}
    available = sum(capacities.values())
    if remaining:
        exact = {category: remaining*capacities[category]/available for category in categories}
        for category in categories:
            result[category] += int(exact[category])
        leftover = total-sum(result.values())
        ranking = sorted(categories, key=lambda category: (-(exact[category]-int(exact[category])), category))
        for category in ranking:
            if leftover and result[category] < len(groups[category]):
                result[category] += 1
                leftover -= 1
        if leftover:
            raise AssertionError('Quota allocation failed')
    return result


def deterministic_rank(row, seed):
    text = f'{seed}/{row["category"]}/{row["sample_id"]}/{row["source_video_sha256"]}'
    return hashlib.sha256(text.encode()).hexdigest()


def plan_only(args):
    if args.dataset is None:
        raise ValueError('--dataset is required for --plan-only')
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory for a new immutable plan')
    dataset = args.dataset.resolve()
    manifest = dataset / 'dataset.jsonl'
    source_rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    rows, excluded = {}, []
    for row in source_rows:
        sample_id = row['id']
        if not isinstance(sample_id, str) or Path(sample_id).name != sample_id or sample_id in ('.', '..'):
            raise ValueError('Invalid source sample ID')
        if sample_id in rows:
            raise ValueError(f'Duplicate ID in source manifest: {sample_id}')
        media = row['media']
        source = (dataset / row['relative_video_path']).resolve()
        if not source.is_relative_to(dataset) or not source.is_file():
            raise ValueError(f'{sample_id}: source is missing or outside the selected dataset')
        if source.stat().st_size != media['bytes']:
            raise ValueError(f'{sample_id}: source size differs from verified manifest')
        if media['frames'] < SOURCE_FRAMES or float(media['fps']) != FPS or not media['full_decode_passed']:
            raise ValueError(f'{sample_id}: source metadata does not support the strict 120-frame contract')
        digest = media['sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError(f'{sample_id}: missing or invalid source SHA256')
        decision = row.get('collection_decision', {})
        rows[sample_id] = {
            'sample_id': sample_id, 'source_video_path': str(source),
            'source_video_sha256': digest, 'source_id': row.get('source', {}).get('id'),
            'category': str(row.get('source', {}).get('category') or 'unknown'),
            'visual_status': decision.get('visual_status'),
            'historical_issue': bool(decision.get('user_restoration')) or decision.get('visual_status') == 'visual_issue',
            'source_frames_available': media['frames'], 'source_fps': media['fps']}
    if any(identifier not in rows for identifier in LEGACY_IDS):
        raise ValueError('All three existing legacy clips must be present')
    legacy_hashes = {rows[identifier]['source_video_sha256'] for identifier in LEGACY_IDS}
    if len(legacy_hashes) != len(LEGACY_IDS):
        raise ValueError('Legacy clips contain duplicate source SHA256')
    candidates, seen = [], set(legacy_hashes)
    for identifier in sorted(rows):
        row = rows[identifier]
        if identifier in LEGACY_IDS:
            continue
        reason = None
        if row['historical_issue']:
            reason = 'historical_visual_issue_excluded_from_all_new_splits'
        elif row['source_video_sha256'] in seen:
            reason = 'duplicate_source_sha256_or_legacy_alias'
        if reason:
            excluded.append({'sample_id': identifier, 'reason': reason})
        else:
            candidates.append(row)
            seen.add(row['source_video_sha256'])
    groups = defaultdict(list)
    for row in candidates:
        groups[row['category']].append(row)
    for category in groups:
        groups[category].sort(key=lambda row: (deterministic_rank(row, args.split_seed), row['sample_id']))
    selected, allocations = [], {}
    # Allocate blind test first, then dev, then train, all from the same seeded
    # category order. Neither captions nor model scores enter this decision.
    for split in ('test', 'dev', 'train'):
        allocations[split] = quota(SPLIT_COUNTS[split], groups)
        for category, count in allocations[split].items():
            for row in groups[category][:count]:
                selected.append({**row, 'split': split})
            groups[category] = groups[category][count:]
    for identifier in LEGACY_IDS:
        selected.append({**rows[identifier], 'split': 'legacy'})
    for remaining in groups.values():
        excluded.extend({'sample_id': row['sample_id'], 'reason': 'not_selected_by_fixed_seed_and_quota'} for row in remaining)
    selected.sort(key=lambda row: (SPLIT_ORDER.index(row['split']), row['sample_id']))
    if len({row['source_video_sha256'] for row in selected}) != len(selected):
        raise AssertionError('Source SHA256 overlaps across planned splits')
    for row in selected:
        actual = sha256(row['source_video_path'])
        if actual != row['source_video_sha256']:
            raise ValueError(f'{row["sample_id"]}: source SHA256 does not match the verified collection')
    plan = {
        'schema_version': 1, 'source_dataset': str(dataset),
        'source_manifest_sha256': sha256(manifest), 'source_manifest_count': len(source_rows),
        'split_seed': args.split_seed, 'posterior_seed': args.posterior_seed,
        'num_shards': args.num_shards,
        'selection': 'Category stratification with minimum-one largest-remainder quotas; SHA256(seed/category/id/source_sha) rank; test then dev then train',
        'historical_issue_policy': 'Exclude known restored/visual_issue clips from train, dev and test',
        'split_counts': SPLIT_COUNTS, 'category_allocations': allocations,
        'legacy_ids': list(LEGACY_IDS), 'entries': selected, 'excluded': excluded,
        'source_frames': SOURCE_FRAMES, 'fps': FPS, 'latent_frames': LATENT_FRAMES,
        'decoded_frames': SOURCE_FRAMES, 'raw_decoded_frames': RAW_DECODED_FRAMES,
        'vae_internal_pad_frames': VAE_INTERNAL_PAD_FRAMES, 'vae_padded_input_frames': VAE_PADDED_INPUT_FRAMES,
        'base_model': args.base_model, 'dtype': 'float32',
        'limitations': ['These are generated-source development videos, not a real-world video benchmark.',
                       'Only exact source SHA256 duplicates are excluded; semantic near-duplicates are not identified.',
                       'Dev/test labels are fixed before training; legacy clips are not a new blind test.']}
    args.output.mkdir(parents=True)
    save_json(args.output / 'plan.json', plan)
    print(json.dumps({'plan': str(args.output / 'plan.json'), 'plan_sha256': sha256(args.output / 'plan.json'),
                      'counts': dict(Counter(row['split'] for row in selected)),
                      'category_allocations': allocations, 'excluded_reasons': dict(Counter(row['reason'] for row in excluded))}), flush=True)


def load_plan(args):
    path = args.plan or args.output / 'plan.json'
    if path.resolve().parent != args.output.resolve():
        raise ValueError('Worker/finalize output must be the directory containing its immutable plan')
    plan = json.loads(path.read_text())
    if plan['num_shards'] != args.num_shards:
        raise ValueError('--num-shards differs from immutable plan; workers cannot change ownership')
    if plan['split_counts'] != SPLIT_COUNTS or len(plan['entries']) != sum(SPLIT_COUNTS.values()):
        raise ValueError('Unexpected planned split sizes')
    if len({row['sample_id'] for row in plan['entries']}) != len(plan['entries']):
        raise ValueError('Duplicate sample IDs in plan')
    if len({row['source_video_sha256'] for row in plan['entries']}) != len(plan['entries']):
        raise ValueError('Duplicate source SHA256 in plan')
    return plan, sha256(path)


def encode_shard(args):
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError('--shard-index must be in [0, --num-shards)')
    plan, plan_hash = load_plan(args)
    if args.base_model != plan['base_model']:
        raise ValueError('Worker base-model path differs from immutable plan')
    assigned = [row for index, row in enumerate(plan['entries']) if index % args.num_shards == args.shard_index]
    shard_name = f'shard-{args.shard_index:03d}-of-{args.num_shards:03d}'
    directory = args.output / 'shards' / shard_name
    directory.parent.mkdir(exist_ok=True)
    # Exclusive mkdir prevents two workers claiming the same shard, even if they
    # were launched on different devices. Partial shards are never overwritten.
    directory.mkdir()
    proof_directory = directory / 'source_proof'
    proof_directory.mkdir()
    (args.output / 'samples').mkdir(exist_ok=True)
    state = {'status': 'preflighting', 'plan_sha256': plan_hash, 'shard_index': args.shard_index,
             'num_shards': args.num_shards, 'expected_samples': len(assigned), 'records': []}
    save_json(directory / 'summary.json', state)
    started = time.perf_counter()
    try:
        # No clip-specific source work can change allocation or substitute data.
        for row in assigned:
            if sha256(row['source_video_path']) != row['source_video_sha256']:
                raise ValueError(f'{row["sample_id"]}: source changed since plan creation')
            if (args.output / 'samples' / f'{row["sample_id"]}.pt').exists():
                raise FileExistsError(f'{row["sample_id"]}: sample already exists; refusing overwrite')
        device = select_device(args.device)
        vae = load_video_vae(args.base_model, device)
        if vae.config.clip_length != VAE_CLIP_LENGTH or released_decoded_frames(LATENT_FRAMES) != RAW_DECODED_FRAMES:
            raise ValueError('Unexpected VAE temporal geometry')
        state['status'] = 'encoding'
        save_json(directory / 'summary.json', state)
        for row in assigned:
            then = time.perf_counter()
            sample_id, source = row['sample_id'], Path(row['source_video_path'])
            proof = {'sample_id': sample_id, 'source_video_sha256': row['source_video_sha256'],
                     'plan_sha256': plan_hash, 'full_source_count': count_source_frames(source, proof_directory, sample_id)}
            save_json(proof_directory / f'{sample_id}.json', proof)
            if not proof['full_source_count']['has_at_least_120_source_and_fps_frames']:
                raise ValueError(f'{sample_id}: cannot supply 120 source-timed frames; no fallback permitted')
            hr_pixels, proof['strict_reader'] = read_first_120_strict(source, proof_directory, sample_id)
            proof['passed'] = True
            save_json(proof_directory / f'{sample_id}.json', proof)
            lr_pixels = F.avg_pool2d(hr_pixels.permute(0, 3, 1, 2).float(), 2, 2)
            lr_pixels = lr_pixels.round().clamp(0, 255).byte().permute(0, 2, 3, 1).contiguous()
            draft = encode_pixels(vae, lr_pixels, device, plan['posterior_seed'])
            reference = encode_pixels(vae, hr_pixels, device, plan['posterior_seed'])
            if tuple(draft.shape) != (24, LATENT_FRAMES, 12, 21) or tuple(reference.shape) != (24, LATENT_FRAMES, 24, 42):
                raise ValueError(f'{sample_id}: unexpected encoded latent geometry')
            metadata = {
                **row, 'id': sample_id, 'plan_sha256': plan_hash,
                'source_frames': SOURCE_FRAMES, 'source_frames_encoded': SOURCE_FRAMES,
                'output_frames': SOURCE_FRAMES, 'decoded_frames': SOURCE_FRAMES,
                'latent_frames': LATENT_FRAMES, 'raw_decoded_frames': RAW_DECODED_FRAMES,
                'vae_internal_pad_frames': VAE_INTERNAL_PAD_FRAMES, 'vae_padded_input_frames': VAE_PADDED_INPUT_FRAMES,
                'vae_internal_padding': 'Released VAE repeats last input to multiple of17 frames',
                'decode_tail_crop_frames': RAW_DECODED_FRAMES-SOURCE_FRAMES,
                'fps': float(FPS), 'draft_pixels': [WIDTH//2, HEIGHT//2], 'target_pixels': [WIDTH, HEIGHT],
                'posterior': 'sample', 'posterior_seed': plan['posterior_seed'],
                'normalization': '(raw-vae.latents_mean)/vae.latents_std', 'base_model': args.base_model,
                'pairing': 'Same strict120 source RGB frames; HR672x384 cover-area resize/crop; exact2x2 area-average uint8 LR; FP32 VAE, all37 latents retained',
                'source_proof_path': str((proof_directory / f'{sample_id}.json').relative_to(args.output)),
                'external_time_padding': False, 'external_slowdown': False,
                'new_training': False, 'flow_computed': False}
            target = args.output / 'samples' / f'{sample_id}.pt'
            temporary = target.with_suffix(f'.pt.{shard_name}.tmp')
            if target.exists() or temporary.exists():
                raise FileExistsError(target)
            torch.save({'draft': draft, 'reference': reference, 'source_pixels': hr_pixels,
                        'metadata': metadata}, temporary)
            # The fixed plan allocates each ID once; exclusive shard ownership
            # guarantees another legitimate worker cannot target this filename.
            temporary.replace(target)
            record = {'sample_id': sample_id, 'split': row['split'],
                      'path': str(target.relative_to(args.output)), 'sha256': sha256(target),
                      'source_video_sha256': row['source_video_sha256'], 'plan_sha256': plan_hash,
                      'bytes': target.stat().st_size, 'draft_shape': list(draft.shape),
                      'reference_shape': list(reference.shape), 'source_pixel_shape': list(hr_pixels.shape),
                      'seconds': time.perf_counter()-then}
            state['records'].append(record)
            save_json(directory / 'summary.json', state)
            print(json.dumps({'shard': shard_name, 'encoded': sample_id, **record}), flush=True)
            del draft, reference, hr_pixels, lr_pixels
        manifest = directory / 'manifest.jsonl'
        manifest.write_text(''.join(json.dumps(record)+'\n' for record in state['records']))
        state.update(status='completed', seconds=time.perf_counter()-started,
                     manifest_sha256=sha256(manifest), total_bytes=sum(record['bytes'] for record in state['records']))
        save_json(directory / 'summary.json', state)
        print(json.dumps({'shard_completed': shard_name, 'samples': len(state['records']), 'seconds': state['seconds']}), flush=True)
    except Exception as error:
        state.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        save_json(directory / 'summary.json', state)
        raise


def finalize(args):
    plan, plan_hash = load_plan(args)
    for name in (*[f'{split}.jsonl' for split in SPLIT_ORDER], 'READY.json'):
        if (args.output / name).exists():
            raise FileExistsError('Finalized manifests already exist; refusing overwrite')
    expected = {row['sample_id']: row for row in plan['entries']}
    records, seen = [], set()
    for shard_index in range(args.num_shards):
        name = f'shard-{shard_index:03d}-of-{args.num_shards:03d}'
        directory = args.output / 'shards' / name
        state = json.loads((directory / 'summary.json').read_text())
        if state['status'] != 'completed' or state['plan_sha256'] != plan_hash:
            raise RuntimeError(f'{name}: shard is not ready for this plan')
        if state['shard_index'] != shard_index or state['num_shards'] != args.num_shards:
            raise ValueError(f'{name}: shard identity mismatch')
        manifest = directory / 'manifest.jsonl'
        if sha256(manifest) != state['manifest_sha256']:
            raise ValueError(f'{name}: manifest hash mismatch')
        items = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        assigned = {row['sample_id'] for index, row in enumerate(plan['entries']) if index % args.num_shards == shard_index}
        if {row['sample_id'] for row in items} != assigned or len(items) != len(assigned):
            raise ValueError(f'{name}: missing, duplicate or incorrectly assigned samples')
        for row in items:
            identifier = row['sample_id']
            planned = expected[identifier]
            if identifier in seen or row['split'] != planned['split'] or row['source_video_sha256'] != planned['source_video_sha256']:
                raise ValueError(f'{identifier}: sample/split/source identity mismatch')
            if row['plan_sha256'] != plan_hash:
                raise ValueError(f'{identifier}: encoded under another plan')
            path = args.output / row['path']
            if path.resolve() != (args.output / 'samples' / f'{identifier}.pt').resolve():
                raise ValueError(f'{identifier}: unexpected pair path')
            if not path.is_file() or path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
                raise ValueError(f'{identifier}: pair file missing or hash mismatch')
            seen.add(identifier)
            records.append(row)
    if seen != set(expected) or dict(Counter(row['split'] for row in records)) != SPLIT_COUNTS:
        raise ValueError('Not all planned samples are ready')
    if len({row['source_video_sha256'] for row in records}) != len(records):
        raise ValueError('Source SHA256 overlap across finalized splits')
    manifests = {}
    for split in SPLIT_ORDER:
        items = sorted((row for row in records if row['split'] == split), key=lambda row: row['sample_id'])
        path = args.output / f'{split}.jsonl'
        temporary = path.with_suffix('.jsonl.tmp')
        temporary.write_text(''.join(json.dumps(row)+'\n' for row in items))
        temporary.replace(path)
        manifests[split] = {'path': path.name, 'samples': len(items), 'sha256': sha256(path)}
    ready = {'status': 'ready', 'plan_sha256': plan_hash, 'num_shards': args.num_shards,
             'samples': len(records), 'total_bytes': sum(row['bytes'] for row in records),
             'manifests': manifests, 'source_sha256_unique': True,
             'posterior_seed': plan['posterior_seed'], 'split_seed': plan['split_seed']}
    save_json(args.output / 'READY.json', ready)
    print(json.dumps(ready), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--plan-only', action='store_true')
    mode.add_argument('--finalize', action='store_true')
    p.add_argument('--dataset', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', type=Path)
    p.add_argument('--base-model', required=True)
    p.add_argument('--device', default='npu:0')
    p.add_argument('--split-seed', type=int, default=20261104)
    p.add_argument('--posterior-seed', type=int, default=42)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=3)
    args = p.parse_args()
    if args.num_shards < 1:
        p.error('--num-shards must be positive')
    torch.set_num_threads(4)
    if args.plan_only:
        plan_only(args)
    elif args.finalize:
        finalize(args)
    else:
        encode_shard(args)


if __name__ == '__main__':
    main()
