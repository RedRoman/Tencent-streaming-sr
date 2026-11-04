# Open-H3 latent upscaler：配对微调实验

此目录只训练 Open-H3-Regenerate-2K 使用的 latent upscaler，暂不接 diffusion refiner，以分开观察 latent 放大和后续生成的影响。**20 次全参数更新和像素验证已完成；留出片段的 latent MSE 下降约 4.0%，PSNR 增加 0.0514 dB，首/中/尾帧目检未见明显画质提升。** 数据、模型权重和生成的训练 checkpoint 均存放在仓库外部。

## 当前模型是什么

Open-H3 使用社区 [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) 的已发布实现和权重，而不是 MiniMax 尚未发布的 Regenerate 内部实现。[Open-H3 项目页](https://nvlabs.github.io/Sana/Open-H3-Regenerate-2K/) 说明了这一来源；[原作者权重仓库](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) 已公开推理模型，但没有完整训练代码及配对数据。因此本实验从现有权重微调，无需从零训练。

该网络含 **345,280,216 个参数**，输入输出均为 24 通道 H3 latent，只改变空间网格，保留时间长度。512 通道的 3D 卷积主干在 resize 前后各有 12 个带尺度条件的残差块，并穿插时间卷积；中间对特征做 trilinear resize。它是单次确定性前向，没有文本条件或扩散采样循环。本次输入 `[24,13,12,21]`，输出 `[24,13,24,42]`，对应 336×192 → 672×384，13 个 latent 帧解码为 43 帧视频。

归一化严格沿用发布的 `open_h3/upscaler.py`：**已经按 H3 VAE 统计量归一化的输入 `z`，仍经过作者节点的 `D(U(N(z)))`**；这里 `N(z)=(z-mean)/std`、`D(y)=y*std+mean`。不能擅自删去 N/D。网络同时收到 `scale=2` 和 `(T,24,42)` 目标尺寸，输出与同样按 VAE 统计量归一化的 HR reference 比较。脚本先检查其可微前向与同设备 FP32 作者节点的一致性；该检查不是与 BF16 运算做 bit-exact 比较。

这仍是**非因果模型**：时间卷积使用双向 padding，GroupNorm 也在整个时间和空间范围上统计。关闭 temporal chunking 保留 Open-H3 的完整片段行为；chunking 是显存处理方式，不能使网络成为严格流式模型。

## 此次训练与验证

| 项目 | 设置 |
| --- | --- |
| 训练 / 留出 | 2 个训练片段、1 个本次实验内互斥的留出片段 |
| 训练 ID | `VIDGEN5S_0336361`、`VIDGEN5S_0515402` |
| 留出 ID | `VIDGEN5S_0380588` |
| 配对方法 | 同一前 64 帧先转为 672×384 RGB，精确 2×2 区域平均并取整得到 LR RGB；分别经同一 FP32 H3 VAE 编码，截取前 13 个 latent 帧 |
| 后验与上下文 | 两路均以 seed 42 采样，使用相同源帧及时间上下文；不使用此前全长缓存的 HR latent 作为训练目标 |
| 训练 | 全部 upscaler 参数，FP32，eval 模式关闭 dropout，逐块 activation checkpointing |
| 优化器 | AdamW，学习率 `1e-6`，weight decay `0.01`，两个训练片段固定交替，20 次更新 |
| 损失 | normalized-latent MSE + `0.1 ×` 相邻帧差分 MSE |
| 对照 | normalized latent 直接 trilinear、发布 upscaler、固定第 20 步 upscaler、匹配 HR reference |
| 保存与检查 | 第 10 步中间保存，最终由第 20 步模型/optimizer 覆盖；有限非零梯度、参数变化、独立模型重载前向一致性 |
| 像素指标 | 统一 FP32 VAE 解码，视频压缩前统计 RGB MSE、PSNR、时间差分 MSE |

**损失和 20 步设置是本实验自定的可行性试验，不是作者未发布训练配方的复现。** 留出片段不参与梯度或 checkpoint 选择。主要像素比较目标是 HR latent 的 VAE 重建；对源 RGB 的指标仅作次要参考，因为其中包含 VAE 重建及时间对齐误差。单个留出片段、由源视频编码得到的 draft，不能代表真实文生视频 draft 的分布或总体泛化效果。

已完成的 latent 对照如下；数值越低表示越接近此次匹配 HR latent，不等价于视觉质量评分。

| 样本 | 直接 trilinear MSE | 发布 upscaler MSE | 第 20 步 MSE |
| --- | ---: | ---: | ---: |
| 训练 `VIDGEN5S_0336361` | 0.46914151 | 0.10866170 | 0.09133935 |
| 训练 `VIDGEN5S_0515402` | 0.47219121 | 0.14484774 | 0.11496492 |
| 留出 `VIDGEN5S_0380588` | 0.38746807 | 0.07114708 | 0.06828142 |

留出 latent 的时间差分 MSE 从 `0.05822694` 降为 `0.05714449`。同设备 FP32 作者前向与本实现的最大绝对误差为 `0`，最终 checkpoint 在新建模型中重载后，留出输出最大绝对误差为 `0`。单张 Ascend 910B3 上，训练循环耗时 21.41 秒，含基线、保存及重载检查的总耗时为 45.73 秒，峰值已分配设备内存 5.27 GiB。这些耗时不含配对 VAE 编码和像素解码。

## 依赖和复现

本目录复用相邻 [refiner 实验](../h3_refiner_self_forcing_20261104/README.md) 的 `data_utils.py`、`prepare_data.py`、`decode_results.py` 和设备锁，避免复制 tensor/VAE 约定。需要原实验的外部 `data/{train,holdout}.jsonl` 与样本文件：其中 `metadata.source_video_path` 必须指向可用源视频；原参考 latent 只用于新旧缓存差异诊断。无需合并或加载 H3 transformer / refiner LoRA。

上游代码固定为 [NVlabs/Sana `781e8341`](https://github.com/NVlabs/Sana/tree/781e8341fb3b29d3aea5d50265772e9619340b66/models/Open-H3-Regenerate-2K)。upscaler 使用 [Open-H3 资产 revision `50c09cf1`](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K/tree/50c09cf1df5f9262b9279868c1c666bd5fe3476b) 中的 `minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`，SHA256 为 `4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6`；发布 BF16 权重经作者加载器转成 FP32。H3 base 与 Ascend runtime 版本沿用相邻实验记录，适配后的 diffusers runtime 仍需另行提供。

在本代码目录执行，替换所有 `/path/to/` 占位路径：

```bash
export EXP_ROOT=/path/to/h3-work
export H3_BASE_MODEL=/path/to/h3-base
export H3_UPSTREAM_DIR=/path/to/Open-H3-Regenerate-2K
export H3_DIFFUSERS_SRC=/path/to/diffusers-patched/src
export H3_PYTHON_BIN=/path/to/npu-venv/bin
export H3_CANN_ENV=/path/to/Ascend/cann/set_env.sh
export H3_FFMPEG_BIN=/path/to/ffmpeg/bin
export NPU_LOCK_DIR=/path/to/shared-device-locks

# 每一步运行前确认空闲；入口还会持有共享 advisory lock 并检查 npu-smi。
bash run_upscaler_pairs.sh <idle-physical-npu-id>
bash run_upscaler_train.sh <idle-physical-npu-id>
bash run_upscaler_evaluate.sh <idle-physical-npu-id>
```

`H3_UPSTREAM_DIR` 应包含 `open_h3/`。入口自动将本目录设为 `EXP_CODE_DIR`；相邻 helper 路径可用 `H3_REFINER_CODE_DIR` 覆盖。工作目录需要以下外部资产，并将生成以下结果：

```text
assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors
data/{train,holdout}.jsonl + data/samples/*.pt
upscaler_trial/pairs/                       # 新建、不可覆盖的匹配 LR/HR 数据
upscaler_trial/finetune20/summary.json       # 前向校验、latent 指标、训练与重载记录
upscaler_trial/finetune20/steps.jsonl
upscaler_trial/finetune20/latents/           # 各样本各对照；不入库
upscaler_trial/finetune20/checkpoint/        # model.pt、trainer.pt、checkpoint.json；不入库
upscaler_trial/decoded/                     # 像素指标、对照图和视频
```

续训使用 `train_upscaler_trial.py --resume /path/to/checkpoint --steps <new-total> --output <fresh-output>`，其余数据、权重和优化设置必须一致；`--steps` 是包含已完成更新的总数。这不是新的 20 步质量对照。现有输出目录不覆盖，重新实验需使用新的外部目录。

## 解码结果

留出样本的 PSNR（相对匹配 HR latent 的 VAE 重建）为：插值 18.1919 dB，发布权重 28.8042 dB，20 步微调后 28.8557 dB。增益仅 0.0514 dB，首/中/尾帧目检未见明显画质提升，不能把 latent MSE 下降当作质量突破。三组指标、对照图和七个视频见 [结果报告](../../results/h3_upscaler_trial_20261104/README.md)。
