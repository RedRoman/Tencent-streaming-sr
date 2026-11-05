# v5 高变化区域伪影优化：等预算对照

v5 已正式命名并推送至 Tencent 分支，版本提交96b6c1d；固定step1800 checkpoint SHA c7aaaa396d877a84c3fed2a5536189b7ed5e61226a82ef1a9e6be1e831f7f914。

两臂分别2000个AdamW更新，从同一v5权重开始，fresh optimizer、同seed20261106、相同128train/16dev、样本/空间/时间/LPIPS采样顺序、lr1e-5至1e-6、warmup100、batch1、原345M因果架构和冻结H3 decoder。仅物理NPU5，顺序训练，不占用其他任务。8步预检权重不用于正式训练。

control：继续v5原有HR latent MSE + .25 teacher MSE + .1 raw eWarp + RGB L1 + .2 LPIPS。
motion：将.1 raw eWarp换为.1 GT-residual eWarp，另加.5变化区域RGB L1 + .2相邻RGB差分残差L1。

高变化区域由detached decoded-HR reference的相邻RGB通道平均绝对变化>=.05定义；同一区域的两端帧都监督。差分loss对齐GT真实变化，不把运动变化压成0。只按实际RGB index相邻性判断，native raw19→23对应RGB118→119仍是有效相邻。GT/flow不输入student。无光流遮挡置信度mask，不宣称已经解决遮挡或切镜。

训练像素采样维持原90/120非混合帧区域，每4步1次全空间，其余原tile内域；30个crossfade帧仍未直接监督，作为明确后续方向。本轮不改架构/历史缓存/decoder，不是端到端流式或DMD/SelfForcing训练。

每200步验证（含step0）。两臂同一固定16dev区域评分L1+.2LPIPS+.5ROI_L1+.2delta_L1选模，且该区域global score不得恶化>1%、L1/LPIPS各不得恶化>2%。随后对固定选中模型做完整16dev×120帧评价；若motion全片ROI_L1和delta_L1均改善且全局guard通过，才解锁16test+3legacy。gate只接受/拒绝，不重新选checkpoint，不用test调参；test已复用非盲测。control无论是否过gate都是固定对照。

评价同时报告v5/control/motion，完整/prefix102/tail18、30blend/90nonblend，高变化ROI覆盖、像素差分L1/MSE、PSNR/LPIPS以及旧raw latent eWarp与GT-residual eWarp。每个新选中模型36项真实尺寸默认HF32因果检查；另外独立HF32off长163latent审计。保持原HF32true训练/评估的数值边界说明。

当前状态：12项CPU单测通过；8步实际NPU预检已完成，像素合同、9项前后因果检查、原v5冻结依赖一致性与新增三项head梯度检查通过。完整训练待新评估器审查后一次性启动。
