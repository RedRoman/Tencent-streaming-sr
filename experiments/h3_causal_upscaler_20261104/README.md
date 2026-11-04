# Open-H3 latent upscaler：独立因果化与配对微调

本目录独立改造 Open-H3-Regenerate-2K 使用的社区 latent upscaler，先评估冻结权重的因果改造，再做固定 50 步配对微调并解码。**不加载 H3 diffusion refiner，不运行 Self Forcing/DMD，不反传 VAE 梯度。** 数值结果、诊断记录和视频见 [结果报告](../../results/h3_causal_upscaler_20261104/README.md)；本文件说明实现与复现流程。

## 因果化改了什么

原模型来自 [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)，由 Open-H3 集成。网络输入输出都是 24 通道 H3 latent，保持时间长度、放大空间网格。此轮输入 `[24,13,12,21]`、输出 `[24,13,24,42]`，对应 336×192 → 672×384，统一解码为 43 帧、24 fps。

`CausalUpscalerAdapter` 保留参数名称与形状，同时改变两类算子：

- 所有时间核大于 1 的 `Conv3d` 改成仅左侧零填充，当前输出不读取未来帧。
- 所有 `GroupNorm` 改为逐帧计算：在当前帧的通道分组和空间范围内统计，继续使用原 affine 参数。原 GroupNorm 会统计整个时间维，只改卷积不足以消除未来依赖。

空间卷积、时间长度不变的 trilinear 特征放大和尺度条件保留。因果适配改变了原网络函数，不能当作预训练模型的等价实现。冻结评估比较两种迁移方式：

| 初始化 | 时间核如何处理 | 含义 |
| --- | --- | --- |
| `shifted` | 权重数值不变，只改为左填充 | 原对称核 `[a,b,c]` 对应的时间偏移由 `[-1,0,1]` 变成 `[-2,-1,0]`，原中心核会落到过去帧 |
| `aligned_past` | 3 帧核 `[a,b,c] → [0,a,b]`；5 帧核 `[a,b,c,d,e] → [0,0,a,b,c]` | 保留原过去与当前帧核的时间位置，丢弃原未来核，并将新增更早位置置零；这些零位置随后可以学习 |

两者都不是“保留原模型效果”的保证。`aligned_past` 仅在从发布权重开始构建训练模型时应用一次；本次微调选择该初始化。

## 归一化与流式缓存

输入 `z` 已经按 H3 VAE 统计量归一化，仍严格遵循 Open-H3 发布节点的 **`D(U(N(z)))`**：`N(z)=(z-mean)/std`，`D(y)=y*std+mean`。不能因为输入已归一化就删掉这一层 N/D，也不能把原始 VAE latent 当作此接口输入。网络同时收到 `scale=2` 与目标尺寸 `(T,24,42)`；输出和匹配 HR latent 使用同一 native normalized 表示。

`native_forward` 是可微的整段因果前向，用于训练；`stream_step` 是 eval/inference 接口，对 **62 个时间卷积**分别缓存其输入特征。50 个时间核为 3 的卷积各留 2 帧，12 个时间核为 5 的卷积各留 4 帧。它不重算全部历史，固定分辨率下缓存不随视频时长增长。新视频、权重变更或几何变更前应调用 `reset_stream()`。

完整网络最长卷积路径的感受野为 `1 + 50×2 + 12×4 = 149` 个 latent 帧，这是结构上限。本轮实际测试片段只有 **13 个 latent 帧**，不能据此认为 149 帧及更长时间上下文已经验证。训练后的诊断包含未来输入在 cut `1/5/10` 之后的扰动、单独前缀与整段比较，以及 `[5,5,3]`、13 个单帧块、`[2,3,1,7]` 三种缓存分块与整段比较，阈值为 `1e-4`；失败保留输出证据并非零退出。

这里的因果性只针对 **upscaler 相对于固定输入 latent**。上游 VAE、像素解码 VAE 和其他 pipeline 组件没有因此被因果化。统一 VAE 解码用于评估内容和画质，不能证明端到端像素流式因果。

## 固定微调设置

配对数据沿用相邻 [upscaler 实验](../h3_upscaler_trial_20261104/README.md)：同一前 64 帧 HR RGB，经精确 2×2 区域平均得到 LR RGB，再分别用同一冻结 FP32 VAE 编码，截取前 13 个 latent 帧。数据、发布权重和训练 checkpoint 都放在仓库外部。

| 项目 | 本次设置 |
| --- | --- |
| 训练片段 | `VIDGEN5S_0336361`、`VIDGEN5S_0515402`，固定交替 |
| 开发留出片段 | `VIDGEN5S_0380588`，仅在这三个片段内部互斥，已用于开发诊断 |
| 初始化 | 发布权重 → 一次 `aligned_past` → 安装因果算子 |
| 参数与精度 | 全部 345,280,216 个 upscaler 参数，FP32；eval 关闭 dropout，启用逐块 activation checkpointing |
| 优化器 | AdamW，学习率 `1e-6`，weight decay `0.01` |
| 固定训练长度 | 50 次更新，无 early stopping、无 holdout checkpoint 选择 |
| 目标 | normalized-latent MSE + `0.1 ×` 相邻帧差分 MSE |

此损失是本实验的监督适配目标，不是原作者未公开训练配方的复现。训练程序只对 train manifest 更新参数；完成全部更新后才计算 before/after、重载和语义检查。before 来自单独保存的初始化权重；不会从最终权重反推。程序逐步记录有限梯度、参数连接情况和抽样参数变化。

最终 `checkpoint/model.pt`、`trainer.pt` 和 `checkpoint.json` 均包含 `adapter_config`。**恢复模型时，先加载已经转换并训练过的 state dict，再重新安装 `CausalUpscalerAdapter`；绝不能再次调用 `initialize_aligned_past_`。** 只加载权重、忘记安装因果算子，也会得到错误的执行语义。代码会独立构造新模型核对重载输出。optimizer 状态虽已保存，当前脚本明确不支持 resume，不能把保存状态等同于已有续训接口。

## 环境与执行顺序

环境及固定上游版本沿用相邻 upscaler/refiner 实验。此目录的 `env.sh` 先加载 `h3_upscaler_trial_20261104/env.sh`，复用 `train_upscaler_trial.py` 等训练 helper 和 refiner 目录的 VAE/锁 helper，再把本目录放到 `PYTHONPATH` 首位。可用 `H3_UPSCALER_CODE_DIR`、`H3_REFINER_CODE_DIR` 覆盖 helper 位置，入口自动设置 `EXP_CODE_DIR`。

在本目录执行，先替换所有占位路径：

```bash
export EXP_ROOT=/path/to/h3-work
export H3_BASE_MODEL=/path/to/h3-base
export H3_UPSTREAM_DIR=/path/to/Open-H3-Regenerate-2K
export H3_DIFFUSERS_SRC=/path/to/diffusers-patched/src
export H3_PYTHON_BIN=/path/to/npu-venv/bin
export H3_CANN_ENV=/path/to/Ascend/cann/set_env.sh
export H3_FFMPEG_BIN=/path/to/ffmpeg/bin
export NPU_LOCK_DIR=/path/to/shared-device-locks

# 先评估冻结权重的两种因果改造，并检查解码结果。
bash run_causal_upscaler_probe.sh <idle-physical-npu-id>
bash run_decode_causal_upscaler.sh <idle-physical-npu-id>

# 固定 aligned_past 初始化与50步计划，再评估微调结果。
bash run_train_causal_upscaler.sh <idle-physical-npu-id> aligned_past 50
bash run_decode_causal_upscaler_finetune.sh <idle-physical-npu-id> \
  "$EXP_ROOT/upscaler_causal_finetune/aligned_past_steps50"
```

`H3_UPSTREAM_DIR` 需包含 `open_h3/`；提前准备 `EXP_ROOT/upscaler_trial/pairs/` 的 train/holdout manifest、样本文件，以及 `assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`。设备每次启动前均需重新确认空闲，入口还会持有共享锁并检查 `npu-smi`。

冻结评估写入 `upscaler_causal_trial/`；微调写入 `upscaler_causal_finetune/aligned_past_steps50/`，其中保存初始权重、训练 checkpoint、逐步日志、各片段 `causal_before/causal_after/reference` latent 和语义检查证据。解码分别写入各自的 `decoded/`。已有输出目录不覆盖，重新试验需使用新的外部工作目录或直接调用 Python CLI 指定新的 `--output`。

像素评估的主要目标是匹配 HR latent 经同一 FP32 VAE 得到的重建，指标在视频压缩前计算；源 RGB 为次要目标，其中包含 VAE 重建与时间对齐误差。冻结对照和微调对照各自提供单分支视频与四列视频，具体数值和观察统一记录在 [结果报告](../../results/h3_causal_upscaler_20261104/README.md)。
