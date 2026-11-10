"""Aggregate already verified full-frame receipts; no inference or training."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = [('WingIt_dev_3586_3811', 'WingIt3586'), ('ElephantsDream_dev_13070_13295', 'ElephantsDream13070'), ('WingIt_dev_300_525', 'WingIt300')]
rows = []
for step in [1000, 2000]:
    for sample, short in SAMPLES:
        path = ROOT / f'evaluations/anchored_step{step}_v1' / sample / 'candidate_pixels/result.json'
        if step == 1000 and short == 'WingIt3586':
            path = ROOT / 'evidence/anchored_eval1000_WingIt3586_full226_pixels_v1.json'
        record = json.loads(path.read_text())
        current, base = record['nativeHR_metrics'], record['no_refiner_nativeHR_metrics']
        assert current['frames'] == base['frames'] == 226
        keys = ['pixel_l1', 'mse', 'lpips', 'delta_l1', 'temporal_difference_mse']
        ratios = {k: current['regions']['full226'][k] / base['regions']['full226'][k] for k in keys}
        counts = {}
        for metric, field in [('pixel_l1', 'l1'), ('mse', 'mse'), ('lpips', 'lpips')]:
            a, b = ([x[field] for x in m['frame_metrics']] for m in [current, base])
            counts[metric] = {'improved': sum(x < y for x, y in zip(a, b)), 'worse': sum(x > y for x, y in zip(a, b))}
        folder = ROOT / f'media/anchored_stage{step}_{short}_preview'
        receipt_path = folder / 'result.json'
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            for name, item in receipt['files'].items():
                assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == item['sha256']
        else:
            assert step == 1000 and short == 'WingIt3586'
            receipt_path = ROOT / 'publication/results/h3_streaming_refiner_20261109/evidence_sha_mapping.json'
            mapping = json.loads(receipt_path.read_text())
            files = list(folder.glob('*.mp4'))
            assert len(files) == 3
            for file in files:
                assert hashlib.sha256(file.read_bytes()).hexdigest() == mapping[file.name]['original_sha256']
        rows.append({'stage_step': step, 'total_training_updates_across_recipes': 2500 + step, 'sample_id': sample, 'short': short, 'source_metric_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'ratios_vs_no_refiner': ratios, 'per_frame_counts': counts, 'playback_directory': str(folder), 'playback_receipt_sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest()})
summary = {'status': 'six_full226_video_sets_complete_technical_audit_local_playback_verified', 'frames_per_video': 226, 'fps': 24, 'video_sets': 6, 'videos': 18, 'rows': rows, 'continuous_visual_reviewed': False, 'strict_decoder_validation': 'running; pretrained small-spatial nine-latent parity passed exactly0, full-spatial137 pending', 'real33B_long137': 'running exactstage1000 only; looped authorized67draft, not137unique video', 'model_accepted': False, 'automatic_stage2000_promotion': False, 'optimizer_updates_since_stage2000_training_endpoint': 0}
(ROOT / 'evidence/anchored_six_video_sets_full226_summary_v1.json').write_text(json.dumps(summary, indent=2) + '\n')
header = '''# 固定 1000 / 2000 步完整视频评估

两个检查点、三个开发样例共六组视频全部完成，每组包含候选单路、候选／原版双向 refiner／真实 GT，以及候选／无 refiner／真实 GT，共 18 份视频。所有视频均为完整 226 帧、24fps，已检查逐帧时间戳和完整解码。指标来自 1920×1080 无损 RGB，而非轻量播放版。

所有比较固定相同 teacherstep2000 upsampler、无文本／静音条件、噪声、样例及裁剪几何。1000 步与 2000 步分别为新 anchored 阶段步数；累计更新为 3500 / 4500，跨两种配方。

表中为候选误差／无 refiner 误差，低于 1 更好。时间差指标比较生成帧间变化与真实 GT 帧间变化的误差，不能直接等同于视觉闪烁结论。

| 样例 | 新阶段步数 | L1 比 | LPIPS 比 | 时序差 L1 比 | 时序差 MSE 比 | LPIPS 改善／退化帧数 |
|---|---:|---:|---:|---:|---:|---:|
'''
lines = [header]
for row in rows:
    v, c = row['ratios_vs_no_refiner'], row['per_frame_counts']['lpips']
    lines.append(f"| {row['short']} | {row['stage_step']} | {v['pixel_l1']:.6f} | {v['lpips']:.6f} | {v['delta_l1']:.6f} | {v['temporal_difference_mse']:.6f} | {c['improved']}／{c['worse']} |\n")
lines.append('''
两个检查点在三个样例上的平均 L1 都优于无 refiner，但两个 WingIt 的 LPIPS 均在全部 226 帧退化。2000 相对 1000 在三例上的 L1 分别变差约 2.72%、3.58%、2.91%；LPIPS 则分别改善约 0.56%、0.57%、1.10%。这仍是混合结果，不能自动把 2000 步提升为最佳或宣称训练后零劣化。

2000 步三个样例各查看第 0、144 帧的主对比与无 refiner 消融。原版双向输出中可见与 GT 不同的新增／改写细节：WingIt3586 的绳线、背景纹理，ElephantsDream 的管线纹理和墙面斑点，WingIt300 的仪表盘按钮、刻度及机身装饰。候选和无 refiner 的主体布局更接近 GT；候选细节仍偏软，不能仅凭保留内容证明它优于无 refiner。以上仅是抽帧观察，不是全片连续审片，也不能证明所有帧都没有幻觉。

目前画质视频仍使用 official overlap decoder。另行启动的预训练严格 decoder 验证已通过真实全模型九个小空间 latent 的精确 parity，137 个全空间循环 latent 压力测试仍在进行。真实 33B refiner 的独立 137 latent 测试也在运行；固定 1000 步、循环授权 67 latent，仅用于增量 KV／缓存／恢复工程验证。它们不替代真实长视频画质、2000 步长流、全长离线 parity 或实时吞吐验收。24fps 为视频播放帧率，生成不是实时。

连续视觉审片、严格 decoder 完整结果与长流完整结果仍未通过；没有已验收的最佳 refiner。训练按新阶段 2000 步自然结束，没有新增训练、没有重启评估。
''')
report = ''.join(lines)
(ROOT / 'evidence/ANCHORED_SIX_VIDEO_SETS_FULL226_V1.md').write_text(report)
index = ['# 完整视频产物索引\n\n所有链接均为本地已校验的完整 226 帧播放版。1000 / 2000 均指 anchored 新阶段步数。主对比列顺序：候选／原版双向 refiner／真实 GT；消融列顺序：候选／无 refiner／真实 GT。候选尚未通过模型验收。\n\n', '| 样例 | 新阶段步数 | 主三列对比 | 无 refiner 消融 | 候选单路 |\n|---|---:|---|---|---|\n']
for row in rows:
    folder = Path(row['playback_directory'])
    step = row['stage_step']
    paths = [folder / f'anchored_stage{step}_{kind}_preview.mp4' for kind in ['dense_gt', 'no_refiner_gt', 'native1080']]
    # Earliest preview retains the same source stem; assert every delivered link exists.
    for p in paths: assert p.exists(), p
    links = [f'[{label}](<{p}>)' for label, p in zip(['播放主对比', '播放消融', '播放候选'], paths)]
    index.append(f"| {row['short']} | {step} | {' | '.join(links)} |\n")
index.append(f'\n[完整 226 帧指标与评估限制](<{ROOT / "evidence/ANCHORED_SIX_VIDEO_SETS_FULL226_V1.md"}>)\n')
(ROOT / 'VIDEO_DELIVERY_INDEX.md').write_text(''.join(index))
print(json.dumps({'sets': len(rows), 'videos': len(rows) * 3, 'report': 'evidence/ANCHORED_SIX_VIDEO_SETS_FULL226_V1.md', 'index': 'VIDEO_DELIVERY_INDEX.md'}))
