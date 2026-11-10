"""Post-hoc descriptive analysis of existing decoder receipts; stdlib only.

No model, RGB, cache, or optimizer payload is read. No inference is performed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math


def main(root):
    root = Path(root)
    result_path = root / 'validation/strict_decoder_pretrained_v1/result.json'
    trace_path = root / 'validation/strict_decoder_pretrained_v1/long_stream.json'
    result_bytes = result_path.read_bytes()
    assert hashlib.sha256(result_bytes).hexdigest() == '079d0052455865ce5375295188c93221c3fa333c576491e4c9e80610af4fc592'
    result = json.loads(result_bytes)
    trace = json.loads(trace_path.read_text())['cache_audit_trace']
    strict = result['nativeHR_metrics']['frame_metrics']
    official = result['official_overlap_no_refiner_nativeHR_metrics']['frame_metrics']
    assert len(strict) == len(official) == 226
    keys = ('l1', 'mse', 'lpips')
    assert all(math.isfinite(row[k]) and row[k] >= 0 for arm in (strict, official) for row in arm for k in keys)
    rows = []
    frame_start = previous_refreshes = 0
    for index, audit in enumerate(trace[:67]):
        assert audit['latent_count'] == index + 1
        frame_end = audit['pixel_frames']
        assert frame_end - frame_start == (1 if index % 5 == 0 else 4)
        refresh = audit['refreshed_past_slots'] - previous_refreshes
        previous_refreshes = audit['refreshed_past_slots']
        rows.extend({'frame': frame, 'latent': index, 'frame_in_latent': frame-frame_start,
                     'refreshed_past_slots_on_push': refresh} for frame in range(frame_start, frame_end))
        frame_start = frame_end
    assert [r['frame'] for r in rows] == list(range(226))
    groups = {
        'full226': rows,
        'excluding_first_frame': rows[1:],
        'excluding_first7_latents': [r for r in rows if r['latent'] >= 7],
        'refresh_push_after_initial_window': [r for r in rows if r['latent'] >= 7 and r['refreshed_past_slots_on_push'] > 0],
        'ordinary_push_after_initial_window': [r for r in rows if r['latent'] >= 7 and r['refreshed_past_slots_on_push'] == 0],
    }
    for phase in range(5):
        groups[f'latent_mod5_{phase}'] = [r for r in rows if r['latent'] >= 7 and r['latent'] % 5 == phase]
    summaries = {}
    for name, selected in groups.items():
        frames = [r['frame'] for r in selected]
        metrics = {}
        for k in keys:
            a = sum(strict[f][k] for f in frames)/len(frames)
            b = sum(official[f][k] for f in frames)/len(frames)
            assert b > 0
            metrics[k] = {'strict_mean': a, 'official_mean': b, 'ratio_of_means': a/b,
                          'improved_frames': sum(strict[f][k] < official[f][k] for f in frames),
                          'worsened_frames': sum(strict[f][k] > official[f][k] for f in frames)}
        summaries[name] = {'frame_count': len(frames), 'frame_indices': frames,
                           'latent_indices': sorted({r['latent'] for r in selected}), 'metrics': metrics}
    expected = result['full226_ratios_vs_official_no_refiner']
    for k, original_key in [('l1', 'pixel_l1'), ('mse', 'mse'), ('lpips', 'lpips')]:
        assert abs(summaries['full226']['metrics'][k]['ratio_of_means']-expected[original_key]) < 1e-12
    # These two sets form a disjoint partition; no selected frame is dropped.
    refresh = set(summaries['refresh_push_after_initial_window']['frame_indices'])
    ordinary = set(summaries['ordinary_push_after_initial_window']['frame_indices'])
    assert not refresh & ordinary and refresh | ordinary == set(range(22, 226))
    net_share = {}
    for k in keys:
        total = sum(a[k]-b[k] for a, b in zip(strict, official))
        assert total > 0
        net_share[k] = (strict[0][k]-official[0][k])/total
    out = {
        'status': 'existing_receipts_only_posthoc_descriptive_analysis_complete',
        'sample_id': result['sample_id'], 'input_arm': result['input_arm'],
        'result_sha256': hashlib.sha256(result_bytes).hexdigest(),
        'long_stream_receipt_sha256': hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'metric_definition': 'Ratio of arithmetic mean per-frame error against the same nativeHR target; lower is better.',
        'group_definition': 'Frame-to-latent mapping and refresh events from actual recorded pixel_frames and cumulative refreshed_past_slots; phase is latent index modulo5, excluding first7 latents.',
        'groups': summaries,
        'first_frame_share_of_full226_signed_net_excess_error': net_share,
        'conclusions': [
            'Excluding frame0 leaves mean L1 and LPIPS error worse than official overlap; the aggregate regression is not confined to the first frame.',
            'Excluding all first7 latents (22 frames) still leaves higher mean L1 and LPIPS error; the regression is not confined to the initial window.',
            'Both refresh-emitting and ordinary-emitting groups regress on average. Ordinary pushes have larger ratios here; this does not identify a causal effect of refresh.',
        ],
        'limits': [
            'Post-hoc descriptive groups on one clip, not new preregistered acceptance gates.',
            'Frames are temporally dependent; no independence, significance, causal attribution, or generalization claim.',
            'Refresh phase is confounded with input content, available history, frame position and official overlap behavior.',
            'No intervention separates missing future context, slot/window positioning, blending or refresh effects.',
            'No refiner in either decoder arm. This is not a refiner quality comparison.',
            'No new inference, continuous visual review, decoder compensation, training or threshold change.',
        ],
        'new_accelerator_forwards': 0, 'new_optimizer_updates': 0,
        'strict_decoder_accepted': False, 'continuous_visual_reviewed': False,
    }
    target = root/'evidence/strict_decoder_existing_metrics_diagnostic_v1.json'
    payload = json.dumps(out, indent=2)+'\n'
    if target.exists():
        assert target.read_text() == payload, 'Preserve prior receipt; choose a new version for changed analysis.'
    else:
        target.write_text(payload)
    print(json.dumps({name: {'frames': v['frame_count'], 'ratios': {k: m['ratio_of_means'] for k,m in v['metrics'].items()}} for name,v in summaries.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    main(parser.parse_args().root)
