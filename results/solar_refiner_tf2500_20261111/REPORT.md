# SolarRenderer task refiner：累计2500步完整报告

本次发布固定 `task_tf2500_v2` 累计2500步候选任务LoRA，包含源码快照、推理加载器、CPU导出审计、连续训练记录、已完成评估和全部已知限制。训练已正常结束；**模型质量尚未验收**。这份报告不把另一条Open H3 anchored路线的结果混入SolarRenderer。

## 结论与交付范围

真实更新连续覆盖1–2500。训练样例首7 latent的误差持续下降；step1000完整67 latent自生成历史回放也改善。以上是积极的拟合证据，不代表未见数据、原生HR像素保真、完整2500步视频或实时部署通过。

Release提供160,606,248字节的FP32任务adapter，400个张量、40,140,800个参数；不是独立完整33B模型。训练checkpoint仅作为SHA来源保留，未上传optimizer、RNG、数据或KV缓存。完整原权重与训练状态未修改。本次发布没有NPU前向或optimizer更新，也不改变独立两步蒸馏任务。

## 1. 路线和固定依赖

- MiniMax-H3 Ref2VA约33B底座，依次合并SolarRenderer causal control和causal 4step。两个上游adapter均rank384/alpha384，沿用FP32 delta计算与逐次BF16舍入顺序。
- SolarRenderer源码固定 `0ca4f5e45b062cc9e9aee506de46aabdc22e4c3d`；上游权重版本 `c5512be825f297ed8db390cbeda0289fb0c24779`。
- 50个DiT block的Q/K/V/out各新增rank16/alpha16任务LoRA。冻结底座仍参与前向，任务分支保持FP32。不能把任务delta先并入BF16权重后声称逐位等价。
- C5/W6：每块最多5个latent，块内双向、块间因果，四次去噪后一次clean KV提交。推理使用自身生成历史，结构条件只来自当前与过去块；这是块因果，不是逐latent严格因果。
- 训练为原生teacher-forced flow-matching velocity MSE；不称本阶段已完成DMD或self-forcing。目标GT只在训练监督/允许的历史中使用，step1000推理目标内容全零、不输入GT。
- 冻结upsampler SHA `7b9f43c17462fb2d0554645bc7889dc7abdebc052ee9f18f6780624de912e0b5`，本发布不替换它。

## 2. 数据和条件

两条授权训练样例：Charge_train_3856_4081与Caminandes3_train_168_393，各24×67×68×120 latent，对应226帧24fps。codec画布1920×1088，显示时裁上下各4像素得到1920×1080。训练数据和nativeHR原片未公开。

Charge nativeHR来自3840×1608原片中心裁剪；Caminandes3原片即1920×1080。配对来源与像素SHA详见[来源收据](evidence/source_pairs.json)。Charge来源保留 `original_generation_mode_unbound` 限制；不能因payload校验通过就抹去原generation-mode未绑定的局限。

实际恢复提示词由原Qwen3-VL第50层编码，28个token，prompt tensor SHA `253b1ebea968a41fee598952b162dac8a5900c3dc18c1a7cdc2fa51d60da440a`。来源配对记录中的旧 `no_text_known_silence_full226` 是上游配对阶段字段，不是本路线实际文本条件。推理静音行按上游布局保留；67latent推理只对不被注意力读取的尾部补齐，双声道有效行相等检查记录在events中。

## 3. 训练配方、恢复和终点

原v1完整12latent整帧训练完成step1–12后保存退出。v2从精确step12恢复全部400组任务参数/AdamW以及CPU/NPU RNG，继续step13–2500；不是另加2500步。

常规更新取7latent、32×64原生空间裁块，不缩放；draft与GT同位置裁切，draft首latent保留整帧外观参考，RoPE仍取完整68×120画布位置。常规24个冻结层常驻NPU，每100步切换完整12×68×120更新，再保存。v2共2463裁块更新、25整帧更新。短片训练实际覆盖1–2个历史块，弱于完整W6历史覆盖。

AdamW学习率1e-5，前50累计步warmup，梯度裁剪1。记录的2500个loss和grad norm全部有限。最后一步是完整画布更新，loss=0.206633329；不同裁块/整帧与噪声条件的loss不能直接横比为画质涨跌。

v2作业墙钟20.304小时；裁块均值26.763秒、中位26.628秒，整帧均值233.831秒。张量分配峰值约42.404GB与49.229GB（十进制；非整卡所有内存）。原v1整帧单步约210秒，裁块是速度/上下文取舍，不是数学等价加速。

最终退出时间 `2026-11-11T02:31:30.762409+00:00`，exit0、child reaped，原父子已确认退出。checkpoint 482,164,949字节，SHA `aec47799fbdf3425b8d11828653eaccc0349faa011919a8bd2fbe9fd9a9de422`。CPU导出时检查全部400组AdamW的step均2500、矩有限，CPU RNG实际恢复一致；终点NPU RNG只核保存字节，不冒称重新执行NPU恢复。下一步日程保留用于审计，不启动下一步。

## 4. 已完成的质量评估

同一Charge训练样例、首7latent、完整68×120空间、四步rollout。固定无refiner latent L1=0.195546344；未任务微调预训练输出=0.191528618。

| checkpoint | refined latent L1 | 相对无refiner误差降低 |
|---|---:|---:|
| 100 | 0.187991217 | 3.86% |
| 500 | 0.162499309 | 16.90% |
| 1000 | 0.150191352 | 23.19% |
| 1500 | 0.142224520 | 27.27% |
| 2000 | 0.137005627 | 29.94% |
| 2500 | 0.135543749 | 30.68% |

这些是latent空间、训练样例短测，不能当开发集、LPIPS、nativeHR像素或长片指标。step2500较step2000继续略改善，不能从单个训练短测宣布最佳checkpoint。

step1000独立完整67latent回放：refined L1=0.173326999，draft=0.209982291，误差降低17.46%。自身历史、14块、70次前向已完成；所有chunk拼接与完整输出逐元素一致。原VAE和StreamRender预训练因果ViT24均导出226帧。完整视频中的GT列是目标latent经decoder重建，**不是nativeHR RGB**。

本仓库的两个[视频](#7-视频)均为step1000，不是2500。2500完整67latent/RGB、未见样例nativeHR指标、连续视觉审片、真实不同内容长片和本任务B200性能仍未完成。

## 5. 性能与流式范围

step1000 latent任务总耗时3574.874秒，70次前向累计3561.364秒，其他约13.510秒；约59.6分钟并非主要花在启动/保存。该Ascend实现resident=0，50层冻结权重按层从CPU加载，生成历史KV也在CPU并按层搬运。生成历史有缓存，但 `cache_structure=False` 每次重算有限结构条件窗口；窗口填满后约73,994token/前向。首块均值约18.9秒/前向，稳态约61.5秒/前向。无逐算子profiler，不能给搬运与计算各自占比。

在固定上游实现中，超出47latent训练域后的时间位置处理要求重算结构条件窗口；不能简单开启缓存而承诺保持输出。没有全前缀无限重算，但也不能把这套结构处理称为所有条件都只算一次。FP32任务支路未融合，分段算子/拼接/同步也有开销，尚未分别测量。

StreamRender在本次NPU上单路226帧解码约52秒，与latent推理耗时分开。24fps是文件时间轴，不是实时吞吐。没有本任务B200 benchmark。上游Reactor四B200、1344×768案例报告17帧约1.8–2.1秒，且加速路径不逐位等价；不能套用到本任务单卡1080p。参考[上游Reactor说明](https://github.com/Enderfga/SolarRenderer/blob/main/reactor/README.md)，引用日期2026-11-11。

## 6. 负结果、修复与未验证项

- 原整帧方案速度无法满足单日目标，改为裁块/周期整帧；长运动/全局上下文覆盖减少，质量代价尚未量化。
- 完整step1000生成很慢，当前实现未达到实时。B200加速空间不等于已测加速比。
- Ascend第一版causal decoder因mask广播维度不支持而失败；v2只显式展开同一boolean mask，未换权重或放宽门槛，原失败保留。CPU参考精确；NPU FP32 maxabs 3.5346e-5、meanabs 1.4130e-6；BF16对CPU FP32 maxabs0.177594、meanabs0.005896。此局部门槛不等于全226像素或感知验收。
- 预训练decoder小空间8latent测试验证future-pair隔离、缓存最多4latent；完整226导出成功。没有因此声称端到端逐latent零等待或真实长流画质通过。
- 仅两条训练样例，主要短评来自Charge；过拟合与泛化仍未排除。训练误差改善不能替代独立开发集。
- 本路线尚无独立确认的全片LPIPS改善/退化结论。另一条anchored路线的LPIPS/strict decoder负结果不归入本报告。
- 初始任务LoRA零增量等于**预训练refiner**，不是保证等于无refiner draft；训练后零退化、无内容幻觉、无闪烁/漂移均未保证。
- 当前所有视频未完成连续视觉审片。已有抽帧、全解码、帧数与SHA检查分别保留，不伪装成连续观看。

## 7. 视频

以下都是AI生成的step1000训练样例预览，226帧、24fps；文件字节与既有交付SHA一致。

- [draft / refiner / decoded GT：causal ViT24](videos/draft_refiner_gt_causal.mp4)
- [原VAE / causal ViT24 / decoded GT：两decoder比较](videos/step1000_two_decoders.mp4)
- [视频SHA及范围](VIDEO_MANIFEST.json)

## 8. 权重、加载和复现

候选权重见[Release](https://github.com/RedRoman/Tencent-streaming-sr/releases/tag/solar-refiner-tf2500-20261111)，推理SHA `9c5cd061f3f339c5939e9568fb9d4179e27aaec7d584d9ce68be2073c9c76fc3`。加载与依赖准备见[说明](../../experiments/solar_refiner_tf2500_20261111/README.md)。Release上传未完成前不应把链接存在等同资产就绪；最终以Release资产和SHA清单为准。

源码快照仅替换私有环境路径，原文件SHA和公开SHA由PUBLIC_MANIFEST区分。原配置SHA保留用于追溯，不声称脱敏后配置仍与原hash相同；跨机器训练需要重建路径/资产绑定，不能跳过原断言。数据不公开，因此不能承诺下载本repo就逐位重跑同一训练。加载器提供推理参数装配，CPU检查不等于新NPU/CUDA全模型测试。

上游代码遵循Apache-2.0，模型及此派生adapter沿用MiniMax H3 Community License，包含地域、使用与分发限制；不能把模型称为无条件Apache权重。发行附完整LICENSE和NOTICE。训练参数、optimizer/RNG及私有缓存不在Git/Release中。

## 9. 核验入口

- [连续更新、时间和评估汇总](TRAINING_SUMMARY.json)
- [CPU导出及逐张量回读](evidence/publication_tf2500_v1/export/adapter_export_audit.json)
- [公开加载器CPU数值验证](evidence/loader_cpu_validation.json)：真实任务张量在三个代表模块、FP32/BF16基座下均与原训练分支maxabs=0；不是完整模型新设备验收。
- [原始2500步终点收据](evidence/runs/task_tf2500_v2/checkpoint_2500.json)
- [2500步短评](evidence/runs/task_tf2500_v2/eval_2500.json)
- [完整step1000结果](evidence/evaluations/step1000_full67_v2/result.json)
- [step1000前向耗时分解](STEP1000_PERFORMANCE.json)

报告结论：这是完成2500步、已有训练样例正向证据、具备可加载推理delta的候选发布；完整最终画质与部署性能验收仍未完成。
