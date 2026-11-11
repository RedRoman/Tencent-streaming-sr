# SolarRenderer 预训练块因果任务微调 v1

用户于 2026-11-10 授权启动新路线，确认目标为 2500 个真实 optimizer update，每 100 步保存完整状态。这是独立实验，使用 31051 物理 NPU 6，不修改旧路线的训练或评估。

## 初始化与可训练部分

底座为已严格转换的 MiniMax-H3 Ref2VA 33B，依次合并公开的 SolarRenderer causal control 和 causal 4step LoRA。两份预训练增量已成为冻结底座的一部分，直接参与每次前向。新增 rank-16、alpha-16 任务 LoRA，覆盖全部 50 个 DiT block 的 Q/K/V/out 共 200 个线性层；A 随机小值，B 为零，初始增量为零。只训练这些 FP32 低秩参数。文本 token refiner、原生条件入口、输入输出头、VAE 和上采样器均冻结。没有新增随机条件投影。

优化器 AdamW，学习率 1e-5，前 50 步线性 warmup，betas=(0.9,0.99)，weight decay=0，梯度范数裁剪为 1。原始公开预训练文件不被覆盖。冻结已合并权重不等于输出行为永远不变；任务 LoRA 会改变整模型行为。

## 数据与目标

使用原有 train2 的 Charge / Caminandes3 配对 latent，按原文件 SHA-256 校验。每个样本完整 67 个 latent，GT 来自原始高分辨率视频；draft 来自已有上采样器。样本按步交替；从 0、5、10、…、55 取 12-latent 窗口，24 步覆盖两样本的所有起点。保持 [24,12,68,120]，对应 1088×1920 codec canvas，不缩放空间尺寸。窗口局部时间从 0 开始。

沿用 SolarRenderer 原生 C5/W6 teacher-forcing 文档与 forward_loss：每块单独抽取 shift=12 的噪声时间；当前块读静态条件、可见 draft 结构、过去块的干净 GT 和当前带噪块；当前及未来 GT 都不可见。损失为 data-ward velocity 的无权重 MSE，目标 GT-noise。没有旧路线的 draft L1 项。draft 首 latent 提供静态参考，draft 全窗口提供对齐结构，使用已有真实 Qwen 编码恢复提示词和已编码静音。

12 latent 分为 5+5+2 三块。配置允许最多六块窗口，本轮实际只覆盖最多两个历史块；不能把它表述为已经在六块完整历史上微调。推理时使用自己生成的历史，与 TF 存在暴露偏差；本轮未实现 self-forcing 或 DMD。四步生成能力来自预训练初始化，FM 微调后是否保持，需要单独评估。

## 运行与验证

单卡使用逐层冻结权重搬运、CPU 激活保存和真实反向重计算，保持原生 DiT forward、位置编码与分段可见性。原推理 paging 不用于训练。训练前对真实 H3 block 做直接反向与重计算反向比较，检查零增量与冻结权重不变，检查原生 TF plan 不读当前/未来 GT。首个完整空间更新额外保存 step1 并实际重载 adapter/AdamW/RNG 后继续。

保存 step0、step1 和每 100 步完整 checkpoint，包括 adapter、参数顺序、AdamW、CPU/NPU RNG、下一步样本/时间窗口/噪声 seed、配置哈希和预训练身份；原子落盘、读回检查并记录 SHA-256。SIGTERM/SIGINT 请求在当前更新后保存退出。非有限梯度或损失直接失败，保留上一完整 checkpoint 与错误记录，不自动重启。

在 100/500/1000/1500/2000/2500 步用固定 seed 做 Charge 开头 7 latent 的四步自生成历史评估，保存输出 latent 和 L1。它只是训练样本回放，不是 dev 或 RGB 质量结论。RGB、完整 226 帧、未见样本和长历史评估仍需后续完成。两训练片段不足以证明泛化。

## 启动状态

实现与资源检查进行中；只有远端 events.jsonl 中的 optimizer_update 才计为完成训练步数。启动成功后将补充运行目录和证据。
