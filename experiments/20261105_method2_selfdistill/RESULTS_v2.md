# v2 结果：像素合同蒸馏（单变量：仅换 loss）

**日期**：2026-11-05 ｜ 训练 3000 步完成，eval 完成 ｜ 机器 32193 npu:6

## 结论（单变量归因）

像素 loss 有正效应但不足以独立解决问题——**数据量（2 clip）是当前主要瓶颈**。

## 四方对比（holdout VIDGEN5S_0380588，PSNR vs decoded reference）

| 方案 | PSNR |
|---|---|
| teacher（双向，不可流式） | 30.11 dB |
| method-1（Codex 对齐-过去权重改造） | 25.31 dB |
| **v2 student（本方案）** | **24.04 dB** |
| v1 student（纯 latent MSE） | 22.24 dB |

全部 clip（teacher / v2 student）：
- VIDGEN5S_0336361 (train): 23.87 / 22.39
- VIDGEN5S_0515402 (train): 22.83 / 21.89
- VIDGEN5S_0380588 (holdout): 30.11 / 24.04

训练 clip 上 v2 略低于 v1：PSNR 对感知改善不敏感，v2 优化 LPIPS/纹理而非逐像素对齐；
真实判据是解码后观感（videos/ 目录三个 clip 的 reference/student/teacher mp4）。

## 训练要点

- 3000 步 ~4.5s/步；consistency_max_abs 4.77e-7 passed（像素 loss 未破坏因果性）。
- causality：future_cut 全 0.0；prefix_cut/streaming ≤4.8e-7，passed。
- 训练 clip LPIPS 0.36 → 0.040（充分拟合）；holdout latent_mse 0.295→0.301（持平微涨，
  v2 下 latent 指标系统性偏高属预期——student 被像素目标驱动偏离 teacher latent）。
- 形态：训练 clip 拟合极好 + holdout 落后 = 数据瓶颈证据，触发预案中的扩数据决策。

## 下一轮决策（按 PLAN 预设规则）

1. **扩数据优先**：回到 vidgen1m-h3-5s-selected-493 池子多取真实 clip（构造脚本现成，
   只有 prepare 时间成本），辅以 reference latent bilinear 合成对。
2. loss 保持 v2 合同不动（eWarp/IQA 仍推迟，保持归因清晰）。
3. 若扩数据后仍不达 method-1 的 25.31：再考虑架构（causal 3D conv memory）或
   UltraFlash-pretrained init 变体。
4. 并行：v3 overlap 滑窗 teacher（零训练）实测 GroupNorm 近似误差，见 PLAN_v3。

## 文件

- train_summary_v2.json / summary_v2_eval.json：训练与评测完整指标
- student_pixel_v2_20261105.log / student_pixel_v2_eval.log：原始日志
- videos/：3 clip × reference/student/teacher mp4（120 帧 24fps）
