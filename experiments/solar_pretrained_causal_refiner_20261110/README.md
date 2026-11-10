# SolarRenderer 预训练块因果 refiner

这是当前侧对话的独立路线 `task_tf2500_v2`：尽量复用预训练权重，用小规模任务LoRA适配refiner任务。与[逐latent anchored路线](../open_h3_anchored_refiner_20261110/)分别记录，不混用权重或步数。

截至 **2026-11-10T09:27:44.766968+00:00**，已确认累计 **410/2500** 次更新；最新完整检查点为step400。原整帧v1在step12保存退出，v2完整恢复adapter、AdamW和RNG后继续累计，不是另加2500步。当前没有RGB或未见样本画质结论。

## 预训练与训练设计

- MiniMax-H3 Ref2VA 33B底座依次合并SolarRenderer发布的causal control与causal 4step权重；原始权重冻结但参与前向。旧Open H3任务adapter没有混入本路线。
- 在50个DiT block的Q/K/V/out上新增rank16、alpha16任务LoRA，共40,140,800个FP32可训练参数。AdamW学习率1e-5，前50个累计更新warmup，梯度裁剪阈值1。
- 沿用原生块因果teacher-forced flow-matching velocity MSE，C5/W6。块内双向、块间因果；这不是逐latent严格因果，也不是新DMD/self-forcing训练。
- 常规每步7个latent（5+2），从68×120画布取32×64原生空间裁块，不缩放。draft和GT同位置裁切，整帧draft首latent作静态参考。裁块RoPE仍使用完整画布位置；24个冻结DiT层常驻NPU。
- 每100步改用完整12×68×120更新，并保存检查点；整帧前释放常驻层，之后恢复。裁块减小了时空上下文，周期整帧训练不能保证画质与原方案等价。W6是可配置窗口，当前短训练片段实际只覆盖1至2个历史块。
- 100/500/1000/1500/2000/2500步做固定Charge训练短片段的四步完整空间回放。VAE不参与训练，也尚未集成StreamRender causal decoder。

## 当前证据

step100的完整空间更新、保存及之后切回裁块均已完成。该训练短片段的latent L1：draft **0.195546**，未微调预训练输出 **0.191529**，step100输出 **0.187991**。它不等于RGB观感、长视频一致性或泛化结论。

[当前进度、整帧步和检查点](evidence/training_snapshot.json) · [参数更新与恢复证据](evidence/training_started_v2.json) · [裁块基准](evidence/patch_benchmark_v2.json) · [冻结配置](stages/task_tf2500_v2/config.json) · [训练入口](stages/task_tf2500_v2/train.py)

两份训练样本是Charge与Caminandes3，各67 latent。来源与native-HR裁剪信息见 [配对记录](evidence/source_pairs.json)，当前文本编码见 [prompt记录](evidence/prompt.json)。配对记录中的旧condition字段仅描述上游配对阶段，不是本路线实际使用的恢复prompt。样本数量小，Charge来源记录保留原generation-mode未绑定的限制。

SolarRenderer代码固定为 [0ca4f5e](https://github.com/Enderfga/SolarRenderer/tree/0ca4f5e45b062cc9e9aee506de46aabdc22e4c3d)，模型固定为 [c5512be](https://huggingface.co/Enderfga/SolarRenderer/tree/c5512be825f297ed8db390cbeda0289fb0c24779)。公开初始化与SHA见 [资产记录](evidence/public_assets.json)。上游代码和大模型权重未复制进本目录。

## 复现与范围

运行时为torch 2.7.1加torch_npu、CANN 9.0.1、配套修订版Diffusers 0.40.0.dev0。环境启动脚本见 `with_runtime.sh`，资产准备与Ascend注意力适配脚本保留在根目录。`stages/task_tf2500_v1` 保留前12步整帧配方，`stages/task_tf2500_v2` 为当前配方。

历史证据文件中的step16、候选配置的“Awaiting”及DESIGN启动估时属于产生时快照；当前进度以上方时间戳和training_snapshot为准，裁块取舍已经接受，训练正在进行。

本目录是正在运行实验的代码与证据快照。仅把私有环境路径改成 `/opt/...` 示例路径，并为原文件与公开文件分别保留SHA-256。配置中的原始哈希仍用于追溯真实训练，不代表脱敏文件的新哈希；`PUBLIC_MANIFEST.json` 记录当前公开文件哈希。

在另一台机器复现时，需自行准备被许可使用的预训练模型、对应运行时与数据，恢复环境路径并重新生成部署清单及配置绑定，然后重新运行适用检查。不要直接跳过完整性断言或把这份快照当成一键启动包。训练权重、AdamW/RNG张量、latent缓存和数据集没有提交进Git；这里只保存其校验信息。

本快照的整理与发布没有修改、停止或重启远端训练。历史试验与失败证据仍保留在原分支目录。

