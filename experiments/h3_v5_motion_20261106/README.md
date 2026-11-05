# v5 高变化区域监督：等预算对照（2026-11-06）

状态：**正式两臂流程已启动；此快照仅包含预检证据**。尚无两臂训练完成、完整 dev gate 或 test 质量结论；预检小幅数值变化不作为改进证据。

固定 v5 step1800 初始化 SHA：`c7aaaa396d877a84c3fed2a5536189b7ed5e61226a82ef1a9e6be1e831f7f914`。使用原 345,280,216 参数因果 Conv3d upscaler，不改结构/缓存。两臂各额外 2000 次 AdamW 更新、fresh optimizer、seed20261106、lr1e-5→1e-6、warmup100、batch1、128train/16dev。样本、区域、LPIPS 帧与学习率顺序相同。八步预检权重不用于正式训练。

## 损失

设 `D(pred)`、`D(ref)` 为冻结 H3 decoder 的未截断 float RGB；LPIPS 单独裁剪至[0,1]后映射[-1,1]。共同项为 `HR latent MSE + .25 teacher latent MSE + RGB L1 + .2 LPIPS`。

- control：共同项 + `.1 mean|pred_t − warp(pred_prev)|`。
- motion：共同项 + `.1 mean|(pred_t − warp(pred_prev)) − detach(ref_t − warp(ref_prev))| + .5 ROI_L1 + .2 delta_L1`。
- `mask = mean_RGB(abs(detach(D(ref)_t − D(ref)_prev))) >= .05`。`ROI_L1` 在同一 mask 内监督前后两个端点，除以 `2 × 3 × mask像素数`；按 pair 加权，重叠端点会重复计数，空 ROI 返回与预测图相连的零。
- `delta_L1 = mean|((D(pred)_t−D(pred)_prev) − detach(D(ref)_t−D(ref)_prev))|`，覆盖所有有效相邻 pair 像素。

只按实际 RGB 索引差1判断邻接；raw19→23对应真实RGB118→119，仍纳入。GT、flow 与 mask 不输入 student，target/flow 均 detach。eWarp 使用原 backward flow 与 inbounds/valid-pair mask；没有遮挡或置信度 mask，也不声称 flow 一定准确。日志保留旧 raw eWarp，不能把新 residual 指标当旧指标比较。

## 覆盖与选模

训练保留原 H3 crop：90/120 个非 crossfade 帧可被采样，30 个 blend 帧仍未直接像素监督；每4步1次 full-spatial，其他使用原 tile 内域，每区域4个 LPIPS 帧。decoder/LPIPS 冻结，decoder 仍非因果。本轮不是端到端流式模型，也不是 Self Forcing/DMD 训练。

两臂都含 step0，每200步按相同固定16dev区域评分 `L1+.2LPIPS+.5ROI_L1+.2delta_L1` 选模，要求区域 `L1+.2LPIPS` 不恶化超过1%、L1/LPIPS各不超过2%。随后冻结选中 checkpoint，做完整16dev×120帧 gate：motion 的 ROI_L1 和 delta_L1 均须严格改善，且同样全局 guard 通过，才解锁16test+3legacy。gate只接受/拒绝，不重选 checkpoint；control始终作为固定对照。既有 test 已被查看，属于复用测试而非盲测。

完整评价使用组装后的未视频压缩 uint8 decoder RGB→[0,1]，与训练未截断 float RGB 域不同。计划分别报告 full120/prefix102/tail18、30blend/90nonblend、ROI coverage、PSNR/LPIPS、差分与两种 eWarp。

## 已完成的预检

12项CPU测试通过；实际NPU motion预检8更新、2train/1dev、未打开test。像素区域parity共12项通过，decoder/LPIPS/source/precision未变，真实形状因果检查前后各9项通过。输出头梯度 L2：pixel=0.1214202、motion组合=0.056421652、ROI=0.1038539、residual warp=0.13414486。

参数/输入为FP32张量；实际NPU `conv_allow_hf32=true`、`matmul_allow_hf32=false`，不能据此声称内部卷积全精度FP32。原v5的长小空间形状默认HF32 stream/full数值限制仍适用；正式两臂另计划独立HF32off长163latent检查，不能当成默认精度下的保证。

原始启动包装器失败记录也一并保留，未把失败尝试当完成训练。详见[预检证据](../../results/h3_v5_motion_20261106/README.md)。本包不含权重、训练缓存或视频。

## 公开审阅副本与运行合同

`source_fingerprints.json` 逐文件保留原始bytes/SHA与公开bytes/SHA。路径脱敏和依赖解析会改变公开副本SHA，**嵌入JSON的原运行SHA始终不改写**。旧依赖复用仓库已公开版本，通过`env.sh`/`module_source`解析；不覆盖旧源码、旧manifest。

这是可审阅的代码/证据快照，不是可绕过历史合同直接重跑的独立模型包。严格的原v5来源检查在公开依赖SHA不匹配时会正常拒绝运行。复现需要恢复与原合同完全一致的源码/资产，或以公开源码另建具有真实新SHA的完整来源链；不得回填原SHA或跳过检查。提供所需外部资产后，用`source experiments/h3_v5_motion_20261106/env.sh`配置已公开依赖。新脚本保留锁、一次性启动和失败终止规则。
