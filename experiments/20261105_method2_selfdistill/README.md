# Method-2：因果 latent upsampler 自蒸馏（独立方案，进行中）

**日期**：2026-11-05 ｜ **机器**：ModelArts 32193（昇腾 910B3，npu:6）｜ **状态**：v1 完成并评测（负结果已保留），v2 训练中

独立于对齐-过去权重改造（用户的 method-1）的平行方案：从头训一个因果 student，
蒸馏冻结的 Open-H3 双向 latent upsampler（teacher）的输出，不需要 GT 配对。

## 方案
- Teacher：`minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`（fp32 冻结），输入 draft latent (24,37,12,21) → 输出 (24,37,24,42)。
- Student：CausalStudentUpscaler ~15M 参数。bilinear 残差基线 + head conv→PixelShuffle(2) + 6×CausalMemBlock（拼接过去帧记忆，逐帧因果）+ zero-init 输出。支持 parallel / sequential(带缓存) 双前向，一致性校验。
- 训练对：真实管线配对（draft = 低清生成视频的 H3 latent；reference = 同内容高清生成视频的 H3 latent，仅用于评测）。**无 bilinear downsample 合成**——draft 天然就是真实 LR。
- UltraFlash 的 bilinear-downsample 造对方法不适用/不需要：我们的 LR 是生成器真输出，非合成退化；差异在于真实配对稀缺 vs 合成对可无限生成。

## v1（纯 latent MSE + 0.1 temporal）——负结果
- 3000 步，~0.45s/步，一致性 4.8e-7 通过（因果性无泄漏）。
- 训练 clip 蒸馏保真极好（latent_mse 0.0042），但 holdout 上畸变明显：PSNR 22.24 dB vs teacher 30.11 dB、用户 method-1 25.31 dB；val latent_mse 训练中 0.251→0.280 恶化 → 2-clip 过拟合 + 无感知约束。
- 结论：畸变主因怀疑是 loss 缺像素/感知约束。

## v2（进行中）——单变量改动：仅换 loss
- 引入 songhua 跑法（h3_frozen_mapper672_x2_flashvaed_20261021）的 decode 后像素合同，adapted 到 H3 latent 空间（他的权重不可直接复用：mapper 输出 96ch Wan 空间 → FlashVAED；我们是 24ch H3 空间 → H3 VAE）。
- Loss = latent MSE + 0.1 temporal + 1.0 pixel L1 + 1.0 LPIPS(alex) + 0.33 wavelet(LH/HL/HH)，像素项在冻结 H3 VAE decode 输出上计算、梯度穿过 decoder 只更新 student。每步随机 5-latent-帧窗口（解码 17 帧）控制成本，~4.5s/步。
- songhua 完整合同中的 eWarp(RAFT×0.1) 与 IQA reward 刻意推迟到下一轮，保持单变量归因。
- 进度日志：`logs/student_pixel_v2_20261105.log`（step 460 时 LPIPS 0.36→0.13，持续下降）。

## 目录
- `code/train_student_distill.py` — v1 训练（架构、双前向一致性、验证与 summary）
- `code/train_student_pixel_v2.py` — v2 训练（v1 + 像素合同）
- `code/eval_student_5s.py` — 5s 全量评测：teacher/student/reference latent、vs-reference 与 vs-teacher 指标、因果性校验（future_cut / prefix_cut / 逐帧流式一致性）、解码 mp4
- `results/train_summary.json` — v1 训练总结
- `results/eval/summary_v1_student_holdout.json` — v1 holdout 评测
- `results/videos/` — 3 个训练 clip 的 teacher/student/reference 对比 mp4（v1）
- `logs/` — 训练与评测原始日志

## 已知环境要点（32193）
- `ASCEND_OPP_PATH=/cache/hanbin/ascend_private/cann-8.5.2/opp` 必需（系统 opp 损坏）。
- vendored diffusers：`/home/ma-user/workspace/zzjrua/vdn-minimax-h3/diffusers/src`（AutoencoderKLMiniMaxH3），其 dependency_versions_table.py 的 peft 已补丁 0.17→0.14。
- H3 VAE decode 必须先 `vae.disable_tiling()`（T<5 块触发 torch.cat 空列表 bug）；T=5 窗口 decode+fwd ≈1s。
- LPIPS 需 per-image (N,3,H,W)；DWTForward 不接受 5D。
