# v5 — H3 因果 Conv3d hybrid upscaler

2026-11-06 正式命名。v5 指向已完成训练和 16 条 test 像素评估的同一份模型；本次命名没有重新训练或修改权重。

- 模型：原版 H3 的 345,280,216 参数 Conv3d upscaler，经左侧时间卷积 padding、逐帧 GroupNorm 和历史缓存改造。
- 权重：`best_step_001800.pt`，SHA256 `c7aaaa396d877a84c3fed2a5536189b7ed5e61226a82ef1a9e6be1e831f7f914`；本地保存，未上传仓库。
- 训练：从旧 causal eWarp 初始化，本轮完成 2000 updates，使用包含 step0 的 16 条 dev 选择 step1800。此前训练历史与中断尝试分别见原始实验记录。
- 版本对应的代码与结果提交：`b498f98fb42c4d85519c1d3e972e881b049e78c0`。

| 全部 16 条 test，参考为冻结 VAE 解码 HR | PSNR ↑ | LPIPS ↓ | 像素时间差分 MSE ↓ |
|---|---:|---:|---:|
| 旧 causal eWarp | 24.3314 | 0.200394 | 0.00239366 |
| **v5** | **25.5670** | **0.131721** | **0.00190879** |
| 原版双向 teacher | 26.4083 | 0.131143 | 0.00162947 |

每条视频完整 120 帧、24 fps、5 秒。相对旧 causal 的 PSNR、LPIPS 和像素时间差分指标均在 16 条上改善。原版 teacher 整体仍更好；latent eWarp 增加 8.96% 的退步仍成立，不能用不同的像素时间差分指标掩盖。

相邻帧变化大的区域仍有伪影，人脸、文字、细毛和密集线条仍有失真。数据已复用，不是盲测；分辨率为 672×384 HR、336×192 LR 的合成缩小输入，不代表 2K 或真实生成模型输出。因果性范围仅为固定 latent 到放大 latent，H3 decoder 仍非因果。默认 HF32 下 163 latent、2×2 空间的 stream/full 数值一致性失败证据保留，不能声称所有形状与精度下逐数值等价。没有进行 DMD、Self Forcing、refiner 或 decoder 训练。

- [训练代码](../../experiments/h3_conv3d_hybrid_20261105/README.md)
- [训练与原四例完整证据](../../results/h3_conv3d_hybrid_20261105/README.md)
- [全部 16 条 test 指标、26 段新增视频与审计](../../results/h3_conv3d_test16_20261105/README.md)
- [机器可读版本绑定](version.json)

后续优化从这份冻结的 v5 权重出发，在独立实验目录记录；v5 的原始代码、数据合同和结果 manifest 保持原样。
