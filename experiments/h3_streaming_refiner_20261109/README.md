# Open H3 真正增量流式 refiner：候选路线

固定 stage1000 与 stage2000 的代码、推理权重、三个开发样例的完整视频和报告已整理为 [候选 Release](https://github.com/RedRoman/Tencent-streaming-sr/releases/tag/h3-streaming-refiner-candidates-20261111)。**refiner 的增量 KV 工程验证通过；尚无完整端到端严格流式、画质合格或实时推理的验收模型。**

- [完整报告：训练、全部226帧指标、因果与长流、失败和剩余问题](../../results/h3_streaming_refiner_20261109/FULL_ROUTE_REPORT_20261111.md)
- [两个推理检查点：下载、SHA、加载接口与复现限制](release/WEIGHTS_AND_REPRODUCTION.md)
- [18份原分辨率视频及独立decoder诊断](../../results/h3_streaming_refiner_20261109/RELEASE_VIDEO_INDEX_20261111.md)
- [全部原始/公开证据映射](../../results/h3_streaming_refiner_20261109/evidence_sha_mapping.json)

## 当前完成状态

原 TF2500 已封存。新的 anchored 配方恰好完成2000次optimizer更新，累计4500跨两种配方；所有训练、六个评估和两个validator均自然退出并回收。新阶段1000对应累计3500，2000对应累计4500。两个候选都发布，不自动提升最终检查点。历史运行中快照及失败保留在证据和Git历史中，当前状态以本页与完整报告为准。

| 检查 | 已验证 / 限制 |
| --- | --- |
| 真正流式refiner | 逐latent、当前token投影、真实增量KV、无前缀重算；四latent因果/parity误差0 |
| stage1000长流 | 真实50层33B、137绝对位置、循环67授权draft；首67复现0，KV稳定6,930,022,400字节；两次真实KV落盘/同worker恢复重放0 |
| 完整画质评估 | 两检查点×三开发片×226帧完成；18份原片及轻量版交付；两WingIt全部226帧LPIPS退化 |
| 严格decoder | 137循环draft工程门槛通过；无refiner消融L1比1.056389、LPIPS比1.086851，原1.02门槛失败 |
| 连续视觉审片 | 未完成；静态抽帧、全解码及自动指标不替代连续观看 |
| 性能 | 长流中位57.418秒/latent；24fps只是视频播放率 |

## 路线

冻结MiniMax-H3/Open-H3基座与精确teacher step2000 upsampler。原TF2500 adapter接续新增零输出残差头，`draft + 0.25*tanh(head/0.25)`。新阶段0在实际模型上恒等，训练后不保证零退化。用draft/noise当前输入、draft扰动历史、25%首帧/history dropout训练；HR只作监督。这是新的重建配方，不称已完成self-forcing或DMD。

推理保持三次denoise query和一次clean commit；静态条件不能读video，视频看首帧sink、前帧、当前空间token，绝对位置噪声固定。核心：[anchored_refiner_v1.py](src/anchored_refiner_v1.py)、[paged_runtime.py](src/paged_runtime.py)、[causal_attention.py](src/causal_attention.py)。所有冻结stage保持原样；新公开loader只在CPU验证原权重及头加载。

## 画质摘要

比值为候选/无refiner相对GT的完整226帧误差，越低越好。主六组采用official overlap decoder，不能称严格流式像素管线。

| 样例 | stage1000 L1 / LPIPS | stage2000 L1 / LPIPS |
| --- | --- | --- |
| WingIt3586 | 0.946148 / 1.076976 | 0.971926 / 1.070931 |
| ElephantsDream13070 | 0.908708 / 0.996803 | 0.941267 / 0.991092 |
| WingIt300 | 0.924426 / 1.079811 | 0.951324 / 1.067948 |

stage2000三片L1均较1000变差，LPIPS略改善。用户及静态抽帧观察到原版双向对照改写部分细节；候选与无refiner主体更接近GT，但候选细节仍偏软，无refiner本身也较保真。没有据此宣称无幻觉或整体增益已证明。全部负结果、时序指标和逐帧计数见完整报告。

## 使用范围

Release内每份推理权重约82.5MB，包含401组adapter与4组残差头FP32张量；原始字节及SHA保留，不含33B基座、optimizer、RNG、KV或缓存。需要固定外部权重、授权输入、Open-H3模块和兼容的集群diffusers运行时；该补丁未随仓库公开，不能保证任意新环境一键复现。详见加载说明。模型/第三方组件保留各自许可，代码许可不覆盖所有模型。

本次发布不启动新训练、不改decoder补偿、不重跑已完成作业。距离最终验收仍有decoder保真、稳定细节增益、连续视觉审片、真实长片/生产条件与性能缺口。
