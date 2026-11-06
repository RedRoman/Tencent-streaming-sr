# 诊断源码进度快照

这四份已执行诊断源码的公开副本只做路径脱敏，原始与公开 SHA 见 ../original_public_mapping.json。内嵌合同/源码 SHA 仍指向原运行字节，不能把公开副本 SHA 冒充原运行 SHA。源码依赖已有 experiment helpers、固定数据/权重与运行环境；它们是供审阅的快照，不是无需配置即可运行的独立包。`${EXPERIMENT_ROOT}`、`${H3_BASE_MODEL}`、`${SCRATCH}` 等是公开路径占位符，复现时须通过既有 CLI 参数/环境设置或显式路径适配提供实际位置。

- conversion_factors：只完成四例 latent 阶段，未执行该诊断的像素解码阶段。
- lr_roundtrip：完成两例 LR 解码，与 rawLR 和缓存 HR384 roundtrip 对照。
- temporal_alignment：完成八例 CPU latent 时移扫描，零最佳偏移不排除混帧/模糊。
- complete_h3_zero_aligned_clip：完成固定动画一例的 step0 对比，无训练。
- drafts/resolution v2：**尚未执行**，仅静态独立审查和14项合成结构检查。不能称真实384重放、768编码/预测或画质改善已通过。
