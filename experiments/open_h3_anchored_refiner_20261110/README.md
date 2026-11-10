# Open H3：以 draft 为锚点的逐 latent 流式 refiner

这是主对话当前正在训练的 `anchored_refiner_v1`。原TF训练已经完成2500步；本阶段从该检查点的LoRA、条件投影和全部401组AdamW状态接续，计划新增2000步。两个阶段的目标不同，累计4500步不等于同一配方训练4500步。

截至 **2026-11-10T09:28:52.011739+00:00**，本阶段已确认 **324/2000** 次更新，连同前序TF累计 **2824** 次。更新连续，记录的loss和梯度均有限；最新完整检查点为阶段第300步。当前尚未选出通过完整视频验收的模型。

## 设计

- 使用原Open H3冻结底座，接续TF2500的rank-8 LoRA及条件投影；保留旧优化器状态。新增FP32逐点残差头，结构48→96→24，末层零初始化。
- 输出锚定同一冻结teacher step2000 upsampler的draft，残差逐元素限制在±0.25。真实模型检查已验证阶段第0步输出与draft完全相同；训练后不保证所有样例零退化。
- 当前输入与历史来自draft、噪声及小扰动，GT只作当前目标。训练损失为native-HR x0 MSE加0.1倍残差MSE。旧参数学习率1e-4，新头1e-3。
- 推理逐latent进行，3次去噪加1次clean cache commit，使用真实增量KV。训练采用截断历史；不称为已完成self-forcing或DMD。
- 每100个阶段更新保存，500/1000/2000做短开发集诊断。完整67-latent、226帧视频和严格decoder验收仍需单独完成。

## 来源与验证

训练使用两条工程样本，开发集三条。输入/输出latent为24×67×68×120，codec画布1088×1920，显示裁剪为1080×1920。冻结upsampler SHA-256：`7b9f43c17462fb2d0554645bc7889dc7abdebc052ee9f18f6780624de912e0b5`。

原TF2500的三个开发样例均未通过保真门槛。完整五样例诊断发现自生成历史与首latent保真问题，相关负结果见 [诊断](evidence/TF2500_full_history_analysis_v2.json)。这只是新方案的动机，不能当作新模型已改善的证据。

[当前进度和检查点](evidence/training_snapshot.json) · [真实模型检查](evidence/anchored_real_gates_v1.json) · [保存恢复检查](evidence/anchored_roundtrip_v1.json) · [冻结配置](stages/anchored_refiner_v1/config/anchored_refiner_v1.json) · [训练入口](stages/anchored_refiner_v1/src/train_anchored_refiner_v1.py)

依赖同仓库既有Open H3运行时与冻结资产；前序训练、诊断及历史代码在 [原实验目录](../h3_streaming_refiner_20261109/)。本目录的stage是原运行stage的独立公开快照。

## 复现与范围

本目录是正在运行实验的代码与证据快照。仅把私有环境路径改成 `/opt/...` 示例路径，并为原文件与公开文件分别保留SHA-256。配置中的原始哈希仍用于追溯真实训练，不代表脱敏文件的新哈希；`PUBLIC_MANIFEST.json` 记录当前公开文件哈希。

在另一台机器复现时，需自行准备被许可使用的预训练模型、对应运行时与数据，恢复环境路径并重新生成部署清单及配置绑定，然后重新运行适用检查。不要直接跳过完整性断言或把这份快照当成一键启动包。训练权重、AdamW/RNG张量、latent缓存和数据集没有提交进Git；这里只保存其校验信息。

本快照的整理与发布没有修改、停止或重启远端训练。历史试验与失败证据仍保留在原分支目录。

