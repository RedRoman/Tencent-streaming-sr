"""Read-only evaluation audit and CPU delivery packaging. No accelerator imports."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path('<AUTHORIZED_CACHE>/h3-streaming-refiner-20261109')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sample', choices=['ElephantsDream_dev_13070_13295', 'WingIt_dev_300_525'], required=True)
    args = p.parse_args()
    sample = args.sample
    short = 'ElephantsDream' if sample.startswith('Elephants') else 'WingIt300'
    out = ROOT / 'evaluations/anchored_step2000_v1' / sample
    launch_path = ROOT / f'launches/anchored_step2000_{sample}_v1/launch.json'
    launch = json.loads(launch_path.read_text())
    assert launch['status'] == 'complete' and launch['exit_code'] == 0 and launch['child_reaped']
    assert not any(Path('/proc', str(launch[k])).exists() for k in ['parent_pid', 'child_pid'])
    root_result = json.loads((out / 'result.json').read_text())
    assert root_result['status'] == 'anchored_stage2000_full226_video_ready_visual_review_pending'
    video = json.loads((out / 'videos/result.json').read_text())
    assert video['frames'] == 226 and video['fps'] == 24 and video['stage_step'] == 2000
    assert video['checkpoint_sha256'] == '5e95b0826db51003968597b8ffbc6159f42d0c784ad8c0a10843734fae3dc485'
    for item in video['files'].values():
        assert sha(item['path']) == item['sha256']
        assert Path(item['path']).stat().st_size == item['bytes']
        assert item['all226_pts_checked'] and item['full_decode_error_check_passed']
    latent = json.loads((out / 'candidate_latent/result.json').read_text())
    assert latent['pipeline_shape'] == [1, 24, 67, 68, 120]
    assert latent['model_forwards_full67'] == 268
    assert latent['trained_gates_passed'] and latent['checkpoint_reload_first4_max_abs'] == 0
    assert max(latent['cache_bytes_trace']) == 6930022400
    pixel = json.loads((out / 'candidate_pixels/result.json').read_text())
    prev_path = ROOT / 'evaluations/anchored_step1000_v1' / sample / 'candidate_pixels/result.json'
    prev = json.loads(prev_path.read_text())
    current, base, previous = (pixel['nativeHR_metrics'], pixel['no_refiner_nativeHR_metrics'], prev['nativeHR_metrics'])
    keys = ['pixel_l1', 'mse', 'lpips', 'delta_l1', 'temporal_difference_mse']
    ratios = lambda a, b: {k: a[k] / b[k] for k in keys}
    frame_counts = {}
    for key in ['pixel_l1', 'mse', 'lpips']:
        a = current['lpips_per_frame'] if key == 'lpips' else [x[key] for x in current['frame_metrics']]
        b = base['lpips_per_frame'] if key == 'lpips' else [x[key] for x in base['frame_metrics']]
        assert len(a) == len(b) == 226
        frame_counts[key] = {'improved': sum(x < y for x, y in zip(a, b)), 'worse': sum(x > y for x, y in zip(a, b)), 'equal': sum(x == y for x, y in zip(a, b))}
    evidence = ROOT / 'evidence'
    completion_path = evidence / f'anchored_eval2000_{short}_complete_v1.json'
    analysis_path = evidence / f'anchored_stage2000_{short}_full226_metrics_analysis_v1.json'
    assert not completion_path.exists() and not analysis_path.exists()
    receipts = ['result.json', 'checkpoint_binding.json', 'shared_input_gate.json', 'candidate_latent/result.json', 'candidate_latent/trained_gates.json', 'candidate_pixels/result.json', 'videos/result.json']
    hashes = {rel: sha(out / rel) for rel in receipts if (out / rel).exists()}
    completion = {'observed_at': datetime.now(timezone.utc).isoformat(), 'sample_id': sample, 'stage_step': 2000, 'launch': launch, 'launch_sha256': sha(launch_path), 'parent_and_child_absent': True, 'receipts_sha256': hashes, 'full67_full226_three_videos_verified': True, 'continuous_visual_reviewed': False, 'strict_decoder_validated': False, 'model_accepted': False, 'optimizer_updates': 0}
    analysis = {'sample_id': sample, 'stage_step': 2000, 'metrics_source_sha256': hashes['candidate_pixels/result.json'], 'stage1000_metrics_source_sha256': sha(prev_path), 'ratios_vs_no_refiner': ratios(current['regions']['full226'], base['regions']['full226']), 'stage2000_over_stage1000_error_ratios': ratios(current['regions']['full226'], previous['regions']['full226']), 'tail18_ratios_vs_no_refiner': ratios(current['regions']['tail18'], base['regions']['tail18']), 'per_frame_comparison_vs_no_refiner': frame_counts, 'nativeHR_full226': current['regions']['full226'], 'no_refiner_nativeHR_full226': base['regions']['full226'], 'continuous_visual_reviewed': False, 'accepted_refiner': None, 'metrics_from_lossless_native1080_RGB': True}
    completion_path.write_text(json.dumps(completion, indent=2) + '\n')
    analysis_path.write_text(json.dumps(analysis, indent=2) + '\n')
    bundle = evidence / f'anchored_stage2000_{short}_delivery_receipts_v1.tar.gz'
    assert not bundle.exists()
    with tarfile.open(bundle, 'w:gz') as tar:
        for rel in receipts:
            f = out / rel
            if f.exists(): tar.add(f, arcname=str(f.relative_to(ROOT)))
        for f in [launch_path, completion_path, analysis_path]: tar.add(f, arcname=str(f.relative_to(ROOT)))
    print(json.dumps({'completion': str(completion_path), 'analysis': analysis, 'receipt_bundle': str(bundle), 'receipt_bundle_bytes': bundle.stat().st_size, 'receipt_bundle_sha256': sha(bundle)}))


if __name__ == '__main__':
    main()
