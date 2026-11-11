# Open H3 真正增量流式 refiner：完整路线、权重与评估报告

日期：2026-11-11。版本：[h3-streaming-refiner-candidates-20261111](https://github.com/RedRoman/Tencent-streaming-sr/releases/tag/h3-streaming-refiner-candidates-20261111)。代码：[独立实验目录](../../experiments/h3_streaming_refiner_20261109/README.md)。本文汇总已完成的收据，取代历史文档中“训练中 / 评估中 / 验证待启动”的当前状态描述；历史文档、失败和冻结源码均保留。

## 1. 结论与交付范围

本路线已经实现并在真实 50 层 33B 模型上验证了**逐 latent 因果、真正增量 KV、无全前缀重算、固定绝对位置噪声与有限缓存**。固定新阶段 1000 的 137-position 工程压力测试通过，首 67 个 latent 精确复现既有输出，真实 KV 落盘后恢复重放也一致。

**尚未得到已验收的端到端严格流式视频 refiner。** 两个 WingIt 样例在两个检查点的全部 226 帧均出现 LPIPS 退化；独立严格 decoder 虽通过工程检查，却未通过原保真门槛；连续视觉审片仍未完成。原六组画质视频使用 official overlap decoder，不能据此宣称整个像素管线严格流式或实时。

本次发布包含两个原始推理检查点、原分辨率视频、轻量播放版、完整指标和失败证据、代码快照、加载说明与 SHA 清单。权重标为候选预发布，不自动把最终 2000 提升为最佳。不发布 optimizer、RNG、KV、训练缓存、私有条件或 33B 基座 payload。

| 已完成事项 | 结果 / 范围 |
| --- | --- |
| 原 TF 阶段 | 累计 2500 更新，完整封存；原 step0 不恒等，完整视频出现明显退化 |
| anchored 新阶段 | 恰好 2000 次真实 optimizer 更新；累计 4500 跨两种配方；自然结束 |
| 新阶段 1000 与 2000 | 各 3 个开发样例，全部 67 latent / 226 帧 / 三路视频完成 |
| 视频交付 | 18 原分辨率 + 18 轻量播放视频；另 2 个 decoder 诊断视频；原片 SHA 保持 |
| refiner 长流 | 固定 stage1000，循环授权 draft 的 137-position 工程验证通过 |
| 严格 decoder | 因果、重置重放与缓存压力检查通过，保真失败 |
| 连续视觉审片 / 模型验收 | 未完成 / 未通过；不能以抽帧或全解码代替 |
| 活动作业 | 所有本任务训练、六个评估和两个 validator 已 exit0、child reaped；不再运行或等待设备 |

## 2. 固定模型、数据与比较协议

基础为 MiniMax-H3 加 Open-H3-Regenerate-2K 发布的合并 refiner LoRA 和 Sharp 分支，保持原 BF16 基座、必要 FP32 运算区、FP32 可训练参数和 Euler 状态，HF32 关闭。额外学习的是 50 层 Q/K/V/out rank-8 adapter、clean-draft 条件投影和新的点式残差头；原 33B 基座冻结。外部原模型版本、合并规则和 14 个 shard SHA 见 [复现说明](../../experiments/h3_streaming_refiner_20261109/release/WEIGHTS_AND_REPRODUCTION.md)。

冻结 upsampler 为 **teacher step2000**，SHA `7b9f43c17462fb2d0554645bc7889dc7abdebc052ee9f18f6780624de912e0b5`。该选择不更换 latest，也不替换为 v5 step1800；冻结与历史用户接受的选择，不等于所有 native-HR 指标均已合格。其既有 guard 失败保留。

几何固定：LR `[1,24,67,44,80]`，refiner draft/output `[1,24,67,68,120]`，解码 1920×1088 后裁 y=4:1084 为 1920×1080。每段 226 帧、24 fps、9.4167 秒。24 fps 是播放率。训练使用两条既有 Charge / Caminandes 配对片段；开发集为 WingIt3586、ElephantsDream13070、WingIt300。这里是有限公开代理开发样例、合成 LR 退化，不是大规模真实生产测试集，未解锁 test set。

实际条件为无文本、已知静音的编码；没有生产 prompt 或未知未来音频因果保证。原版双向对照指固定开源 Open-H3 双向管线，不是闭源官方 API；各组固定 upsampler、输入、噪声、条件、几何和参考来源。stage2000 复用对应已完成 stage1000 的双向参考，并核 source result / gate / manifest / payload SHA。

主比较为「候选 / 原版双向 refiner / 真实 native-HR GT」，消融为「候选 / 无 refiner / GT」。指标在原始 1080p 无损 RGB 上计算；MP4 和轻量预览只用于播放。全部帧的 PTS、帧数、全解码和文件 SHA 已技术审核，但这不是连续视觉观看。

## 3. 推理结构与因果范围

静态 text/audio token 不能读 video token；视频查询只能读静态条件、首帧 sink、前一帧和当前 latent 的空间 token。当前 latent 含完整空间，不等同逐像素因果。历史使用生成完毕的 clean 输出，每层保留对应 K/V；旧历史没有重新前向。

每个 latent 进行三次去噪查询，再额外一次 sigma=0 的 clean KV commit。sigma 固定为 `[0.909375, 6/7, 2/3, 0]`。噪声由 sample_id、固定 seed 和全局 latent 位置确定，因此到达 chunk 的边界不改变噪声。`push()` 可以接收任意数量的 latent，内部始终逐 latent 处理。

anchored 输出头为 FP32 `48→96→SiLU→24`：先计算 `raw_x0=x+sigma*v`，拼接 `(raw_x0-draft, draft)`，得到 `draft + 0.25*tanh(head(...)/0.25)`。零初始化最后投影，使新阶段 step0 输出严格等于冻结 draft，真实模型双 latent 已验证。残差限制作用于 latent 元素；它不构成图像结构、细节、闪烁或零退化保证。

`canonical_rows=True` 和按 latent 分组的输入投影让相同历史下的参考和缓存路径使用一致投影形状。全前缀重算仅出现在明确标记的诊断参考或截断训练上下文，不用于宣称增量流式的推理路径。[核心源码](../../experiments/h3_streaming_refiner_20261109/src/anchored_refiner_v1.py)、[运行时](../../experiments/h3_streaming_refiner_20261109/src/paged_runtime.py)。

## 4. 训练路线、检查点和完整性

原 TF2500 保留全部权重、optimizer 与 RNG。其 step0 并非恒等：即使新增 adapter 为零，高噪声三步去噪仍可改写 draft。原 TF2500 的完整三开发样例 L1 比约 4.737 / 3.986 / 4.735，LPIPS 比约 5.318 / 3.643 / 5.620，均为负结果，不能删除或归为未运行。

独立 anchored 配方从精确 TF2500 adapter 与全部 401 AdamW 状态接续，只新增零输出残差头及 4 组 optimizer slots。原参数 lr=1e-4，新头 lr=1e-3，AdamW betas=(0.9,0.99)，weight_decay=0。当前输入为 draft/noise，历史是 draft 与小扰动；25% 首 latent，25% 其他绝对位置无历史，50% 有界 draft 历史，其中一半加入 std=0.025、裁至 ±0.25 的扰动。HR 只作当前监督，不进入前向历史。目标为 native-HR x0 MSE + 0.1 残差 MSE。这是新配方和截断历史训练，**不是 unchanged TF、完成 self-forcing 或 DMD**。

每 100 更新保存，500/1000/2000 做首 4 latent 开发诊断，随后对指定 1000、2000 完成完整 226 帧评估。真实阶段0恒等、反传梯度、因果、chunk 与恢复 gate 先通过，才进行 optimizer 更新。最终 2000 个连续更新 loss 有限；405 AdamW moments 和步数一致：原 401 组为 4500，新头 4 组为 2000。最终 CPU RNG 恢复精确，NPU RNG 此次终点只验保存字节，不能称再次实际 NPU 恢复。[训练终点收据](anchored_training_complete2000_v1.json)、[CPU 审核](anchored_checkpoint_stage2000_CPU_audit_v1.json)。

| 新阶段 step | 累计更新（跨配方） | WingIt3586 首4 MSE比 | ElephantsDream 首4 MSE比 | WingIt300 首4 MSE比 |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 3000 | 1.201447 | 0.948203 | 1.282472 |
| 1000 | 3500 | 1.096375 | 0.922913 | 1.161290 |
| 2000 | 4500 | 1.102577 | 0.947870 | 1.166400 |

这些短 latent 指标只用于诊断，不能替代视频。完整权重记录：

| 推理文件 | 原始与公开 SHA256 | 状态 |
| --- | --- | --- |
| stage1000 / cumulative3500 | `f16119620e6bcbce0c6da7ce00560dc1733e9d0b3c0d6d4107c38974d43c50a6` | 用户指定的固定中期候选 |
| stage2000 / cumulative4500 | `5e95b0826db51003968597b8ffbc6159f42d0c784ad8c0a10843734fae3dc485` | 固定最终候选；不自动晋升最佳 |

两者各 82,508,131 字节，405 张量 / 20,593,528 参数。完整字节、pickle 元数据多轮转义检查、FP32 有限性、逐张量 SHA 与 CPU 加载通过。没有 optimizer/RNG/KV/cache payload。见 [权重审计](weight_release_CPU_audit_v1.json)、[加载入口审核](release_loader_CPU_validation_v1.json)。原检查点字节未改变。

## 5. 六组完整 226 帧结果

比值定义为候选对 GT 的误差均值除以无 refiner 对 GT 的误差均值，越低越好。时序列衡量相邻帧差分相对 GT 的误差，不等于人工闪烁评分。均值不是逐帧比值的均值。冻结画质限值为 L1、LPIPS 相对误差各不超过 1.02，并要求无结构漂移；未为这次发布放宽。原始全精度结果及每帧计数在 [六组汇总](anchored_six_video_sets_full226_summary_v1.json)，其中旧“验证运行中”字段由本报告完成态取代。

| 样例 | stage | L1比 | MSE比 | LPIPS比 | 时序L1比 | 时序MSE比 | LPIPS改善/退化帧 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WingIt3586 | 1000 | 0.946148 | 0.953731 | 1.076976 | 1.001610 | 0.999900 | 0 / 226 |
| ElephantsDream13070 | 1000 | 0.908708 | 0.898203 | 0.996803 | 0.978680 | 0.935148 | 142 / 84 |
| WingIt300 | 1000 | 0.924426 | 0.917512 | 1.079811 | 1.034530 | 1.070773 | 0 / 226 |
| WingIt3586 | 2000 | 0.971926 | 0.960118 | 1.070931 | 1.000821 | 1.004534 | 0 / 226 |
| ElephantsDream13070 | 2000 | 0.941267 | 0.907540 | 0.991092 | 0.978191 | 0.927836 | 186 / 40 |
| WingIt300 | 2000 | 0.951324 | 0.911391 | 1.067948 | 1.036384 | 1.083188 | 0 / 226 |

stage2000 相对 stage1000，三片 L1 误差分别增加约 2.7246%、3.5830%、2.9097%，LPIPS 误差分别减少约 0.5614%、0.5729%、1.0986%。因此没有全指标优胜检查点。两 WingIt 的感知误差回退不能被平均 L1 改善掩盖；WingIt300 的时序误差也回退。ElephantsDream 的平均改善仍包含局部帧退化，不能推广为全部内容无退化。

用户观察到原版双向引入内容幻觉，而候选与无 refiner 没有同等明显改写，已保留为 [用户视觉反馈](anchored_stage1000_user_visual_feedback_v1.json)。静态抽帧可见双向改变墙面斑点、仪表按钮和机身细节；候选/无 refiner 主体布局更接近 GT，细节仍偏软。**内容幻觉、细节丢失、闪烁与漂移是不同问题。** 无 refiner 本来就较保真，所以相对双向的保真优势不足以证明新增 refiner 的收益。这里没有宣称无幻觉或连续观看通过。

## 6. 真实模型因果与 long137 工程结果

六组评估使用实际完整模型重复四 latent 同历史 cache parity、future-input / future-condition 扰动门槛，观测误差均为 0；短段检查点重载与不同 chunk 重放也为 0。它们不等于全 67/137 offline parity。完整 67 推理为 268 次前向。

long137 固定 stage1000、50 层真实模型、每个 latent 完整 68×120 空间。输入是循环 67 个授权 draft，在 137 个绝对位置处理；不是 137 个不同真实长片输入。结果：

| 检查 | 观测 |
| --- | --- |
| 首67与原始流式产物复现 | maxabs=0 |
| 主要前向 | 548 = 137×(3 query + 1 commit) |
| 每层 K 投影输入 | 每次仅当前 2040 rows；没有全前缀重算 |
| KV 上限 | 6,930,022,400 字节；首 latent 后保持稳定 |
| next66 / next134 保存恢复 | 50 层真实 KV 落盘，读取后恢复绝对计数；同一 worker 重载 adapter/head，3 latent 不同 chunk 重放均 maxabs=0 |
| 额外恢复重放前向 | 24 |
| 中位生成时间 | 57.418 秒 / latent；非实时吞吐 |

结果 SHA `68c817a88697700449b2975e6577a242f11a052fe9b5473c2a49f41aee5971f2`；冻结 stage manifest `6fecf4484198b65d4a95a83de0ddd58f2de8f9b47e62ad84dbae5877bbd95a7d`。[完整分析与退出凭证](REFINER_LONG137_COMPLETE_V1.md)。私人 KV payload 未下载、未发布。

通过范围不包括新进程冷启动整个基座、全137离线一致性、stage2000长流、真实不同长内容的质量、decoder或实时性能。旧进程已自然退出并释放三把锁；本次发布没有重跑验证。

## 7. 严格 decoder：工程通过，保真失败

这是独立 decoder 消融：两路输入均为同一冻结 teacher draft，**没有应用 refiner**。真实完整 36 层预训练 VAE 的 9 个 16×16 latent 与对应增量参考 parity=0；全 68×120 首8个 reset/chunk 重放 RGB=0；扰动第8输入时前7输出不变，第8输出确实变化。

137 循环 draft 输出 464 帧，host KV 上限 63,594,823,680 字节，最多保留7输入 latent，190次slot forward含53次有限历史刷新。reset/chunk重放不是decoder KV落盘恢复，且有限历史刷新与refiner的零历史重算机制应分别描述。

| strict / official overlap，无refiner | 完整226结果 |
| --- | ---: |
| L1 比 | 1.056388590 |
| LPIPS 比 | 1.086851222 |
| L1 退化帧 | 223 / 226 |
| LPIPS 退化帧 | 218 / 226 |
| 时序 L1 / MSE 比 | 1.088515 / 1.119399 |

两项保真比均超过原 1.02 门槛，严格decoder未验收。保持FP32、compensation=false和原源码，不以放宽门槛或替换原六组视频掩盖结果。[完整负结果](STRICT_DECODER_PRETRAINED_RESULT_V1.md)、[独立对比视频索引](RELEASE_VIDEO_INDEX_20261111.md)。

对既有收据的CPU事后分析表明：剔除首帧后L1/LPIPS比为1.048607142/1.086641199；剔除前7latent/22帧后仍为1.046063105/1.084327294。退化不只发生在启动帧。刷新组与普通push组都平均退化，但内容、位置、历史存在混杂；不能据此认定刷新机制或其他唯一根因。该分析没有重新解码RGB、没有NPU前向或optimizer更新。[诊断报告](STRICT_DECODER_EXISTING_METRICS_DIAGNOSTIC_V2.md)。

## 8. 保留的失败、解释与未完成检查

原v3真实4latent parity曾失败：maxabs=0.356204748、relativeRMS=0.010910339。只读trace把离线输入投影改成逐latent分组后差异精确为0，解释了该数值比较失败；不能解释全部画质问题。之后v4五样例的对齐parity和saved-self replay均为0。GT历史与自生成历史、训练截断历史、单步FM和三步推理的诊断保留，仍不构成排他因果结论。[完整历史诊断](TF2500_full_history_analysis_v2.json)。

原step200/TF2500全视频负结果、短pilot延期项、旧控制器身份检查失败、双向worker版本不一致、tmux未启动、CPU编码/收据schema等失败均留在证据目录。发布准备中一次loader测试目录导入失败、一次decoder媒体receipt字段不一致也保留；修复只涉及CPU测试/打包路径，未影响冻结模型。

距离可验收的端到端严格流式方案，具体还需：

1. 解决严格decoder保真退化，并在同一固定协议下重新验证；目前已有因果工程能力，但质量未达标。
2. 证明refiner在无refiner基线上具有稳定细节收益，同时控制WingIt感知与时序回退；不能仅凭较双向少改写而验收。
3. 完成真正连续视觉审片，独立记录结构、细节、闪烁、漂移；现有工具的静态图、接触表、全解码与指标不代替观看。
4. 验证真实不同内容长片、生产条件、完整端到端边界与新进程恢复；137循环draft的工程结论不外推这些范围。
5. 改善推理速度和内存使用；当前约57.4秒/latent以及decoder约63.6GB host KV，不能称实时。

这些是缺口，不是本次已完成项目，也不表示已自动启动新的训练或补偿实验。所有既定计算已结束。

## 9. 发布、复现与审计入口

- [候选 Release：权重、原片、报告、清单](https://github.com/RedRoman/Tencent-streaming-sr/releases/tag/h3-streaming-refiner-candidates-20261111)
- [权重格式、CPU检查、依赖、原运行时安装接口](../../experiments/h3_streaming_refiner_20261109/release/WEIGHTS_AND_REPRODUCTION.md)
- [完整视频目录与逐文件下载](RELEASE_VIDEO_INDEX_20261111.md)，每组候选单路1080p、两种三列对比2880×620；另有独立decoder诊断
- [原片SHA清单](original_video_release_manifest_v1.json)、[Release资产清单](release_assets_manifest_v1.json)
- [source/public SHA映射](source_public_sha_mapping.json)、[原始/公开证据SHA映射](evidence_sha_mapping.json)

公开快照对基础设施路径脱敏，原始SHA与公开SHA分别保存；数值结论与冻结源码版本不被替换。新增loader在真实两份权重上做CPU完整性验证，但未在新环境重跑33B。原始运行依赖兼容的集群diffusers补丁，该补丁不在仓库中，所以发布是可检查的实验路线、权重与运行接口，不应称为任意设备一键复现包。

训练/模型许可、数据权利与外部组件来源分别保留。未来评估需自备授权输入和匹配依赖；这里没有公开私有恢复状态或用户生产数据。以全部负结果和未完成标签作为交付的一部分，而不是以发布成功替代模型验收。
