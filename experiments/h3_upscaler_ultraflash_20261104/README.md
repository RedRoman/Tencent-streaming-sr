# H3 因果 latent upscaler：扩大数据与 eWarp 对照实验

本实验把 [Ultra Flash v4 §2.2 / Eq. (3)](https://arxiv.org/html/2606.09150v4#S2.SS2) 的「MSE 重建 + 光流对齐时序一致性」训练思路用于现有 H3 upscaler。论文的 upsampler 使用 Causal Memory Network；这里保留约 345M 参数的 H3 Conv3d 网络及已有因果改造，**不是 CMN 架构复现，也不是官方完整训练配方复现**。截至本目录整理时，[官方仓库](https://github.com/xin1u/UltraFlash#-training) 仍将 training code/config 标为待发布。

正式两组均已完成 2000 次更新：eWarp 组训练 43.64 分钟，对照组 43.43 分钟，峰值 allocated memory 约 6.43 GiB。两组均由 16 条 dev 选中第 2000 步；数据与评估范围见下文及结果目录。此前 CPU 10 项、NPU 10 项独立损失/因果测试均通过。

## 模型、数据与时序约定

模型沿用[上一轮因果 upscaler](../h3_causal_upscaler_20261104/README.md)：时间卷积只向左补零，GroupNorm 逐帧计算；流式推理保留每个卷积的输入缓存。模型接收固定的 native normalized H3 latent，保持发布节点的 `D(U(N(z)))` 包装、`scale=2` 和明确的目标尺寸。这里的 `N/D` 是 upscaler 节点的归一化/反归一化，不是 VAE 解码。

两组均从已有 `aligned_past` 50 步 checkpoint 开始，加载转换后的权重并重新安装 adapter，不能再做一次 aligned kernel 移位。训练只更新 upscaler，没有 H3 diffusion refiner、DMD、Self Forcing rollout 或 VAE 梯度；H3 VAE decoder 仍非因果。因果检查范围是 upscaler 相对于固定输入 latent 的依赖关系。

从已有 493 条源视频固定选出 **128 train / 16 dev / 16 test**，旧实验三条另列为 **legacy**。14 条历史异常不进入任何新划分；按源文件 SHA256 去重并排除 legacy 别名。类别配额采用最小一条与最大余数分配，类别内以 seed `20261104` 的哈希顺序选择，先固定 test，再 dev，再 train，不使用模型分数选择。未检测语义近重复；这些是生成视频来源的开发数据，不代表自然视频基准。

每条读前 120 个真实源帧，24 fps、5 秒；先完整检查 `fps=24` 输出，不使用外部补帧、慢放或 `tpad`。HR 为 672×384，LR 为其精确 2×2 area average 后的 uint8 336×192。两者用冻结 FP32 H3 VAE、posterior seed `42` 编码，保留 `[24,37,12,21]` / `[24,37,24,42]` latent 与 uint8 源 RGB。VAE 内部将 120 帧重复末帧补到 136 帧，编码得到 37 latent；解码 124 帧后裁去尾部 4 帧，输出仍为真实时长 120 帧。

H3 第 `j` 个 latent 的时序锚点按 `17*(j//5)+4*(j%5)` 计算，含义是编码器最后的源帧依赖位置，不是 RGB 点采样。37 个 latent 中前 36 个锚点落在真实源帧内（最后为 119），第 37 个落在补齐区（123）。因此重建损失只覆盖前 **36** 个，eWarp 只覆盖 **35** 对有效相邻锚点，跨 17 帧编码块的间隔按实际锚点计算。

## 固定训练对照

| 项目 | eWarp 实验 | 同预算对照 |
| --- | --- | --- |
| 损失 | latent MSE + 0.1 × eWarp L1 | latent MSE + 0.1 × temporal-difference MSE |
| 初始化 | 同一已有 50 步 aligned_past 权重 | 同左 |
| 优化器 | 新建 FP32 AdamW，weight decay 0.01 | 同左 |
| 学习率 | 100 步线性 warmup，峰值 1e-5，余弦趋向 1e-6 | 同左 |
| 更新 | 2000 步，batch 1，gradient clip 1，activation checkpointing | 同左 |
| 调度 | seed 20261105，固定 train 样本洗牌顺序 | 同左 |
| 选模 | step 0 及每 200 步 dev，选择平均 latent MSE 最低者 | 同左 |

eWarp 的本地实现使用固定 GT RGB 的四分之一分辨率灰度图，OpenCV Farneback 估计 **current → previous** backward flow，缩放到 latent 网格时同时缩放向量。将上一帧预测 warp 到当前帧，计算 L1 残差：每对有效像素/通道平均，再对有效帧对平均。只用 in-bounds mask，**没有遮挡检测**。流、GT 和完整源 RGB 都不作为 upscaler 的输入。光流算法、0.1 权重、L1、mask、平均方式及上述超参数均是本实验选择，不能归为论文已公布的完整 recipe。

test/legacy latent 只在 dev 选模结束后打开用于最终评估。最终 evaluator 对全部 16 test + 3 legacy 报告 latent 指标；像素指标和 5 秒视频只覆盖事先固定的四条：test 的 animal/daily/sport 各按 ID 排序取第一条，加旧 legacy holdout。主像素参照为同一 HR latent 解码结果，源 RGB 为次要参照；不将四条像素结果写成完整 test 集指标。对照列为原版非因果权重、prior50、长训练 temporal、长训练 eWarp、decoded reference。

已执行因果检查共 **108 项 = 4 条展示视频 × 3 个因果模型 × 9 项**，每组包含 3 个 future perturbation cut、3 个 prefix 和3种缓存分块对照，阈值 `1e-4`。这不等于对全部 19 条评估视频都做因果检查。独立 checkpoint 重载一致性另覆盖 **2 个长训练模型 × 19 条 test/legacy**，比较重新前向与训练结束时保存的预测。评估器还核验两组数据计划、初始化、损失类别、相同预算及 selected checkpoint 哈希；正式评估已完成：108 项因果检查、38 项独立重载检查均通过，最大绝对差均为 0。

## 可移植运行

Python 文件复用相邻目录的因果 adapter、训练/VAE/device-lock helper，以及 5 秒严格读取和因果检查。无需复制这些 helper。先配置实际资源路径；权重、训练数据和缓存不包含在此目录。

```bash
export EXP_ROOT=/path/to/external-experiment-workspace
export H3_BASE_MODEL=/path/to/h3-base
export H3_UPSTREAM_DIR=/path/to/directory-containing-open_h3
export H3_DIFFUSERS_SRC=/path/to/patched-diffusers/src
export H3_SOURCE_DATASET=/path/to/source-dataset
# 按运行环境配置，可省略已在 PATH 中的项目：
export H3_PYTHON_BIN=/path/to/venv/bin
export H3_CANN_ENV=/path/to/ascend/set_env.sh
export H3_FFMPEG_BIN=/path/to/ffmpeg/bin
# 若未放在 EXP_ROOT 下的默认位置，显式设置：
export H3_UPSCALER_WEIGHTS=/path/to/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors
export H3_UPSCALER_INITIAL_CHECKPOINT=/path/to/aligned_past_steps50/checkpoint/model.pt

cd experiments/h3_upscaler_ultraflash_20261104
source ./env.sh
```

源目录需要已有 `dataset.jsonl`，包含脚本读取的 `id`、`relative_video_path`、`media`（bytes/frames/fps/full_decode_passed/sha256）、`source`（id/category）及 `collection_decision` 审查记录。历史异常依据 `user_restoration` 或 `visual_status=visual_issue` 排除；固定旧三条 ID 由脚本列出。准备脚本写入不可变 plan、源哈希、排除理由、逐条 source proof 和分片 summary。

```bash
# 下列 5/6/7 仅示例物理 NPU ID，应替换为已确认空闲的设备。
bash run_prepare_upscaler_ultraflash_pairs.sh plan 3
# 在三个独立终端运行，全部成功后才 finalize：
bash run_prepare_upscaler_ultraflash_pairs.sh worker 5 0 3
bash run_prepare_upscaler_ultraflash_pairs.sh worker 6 1 3
bash run_prepare_upscaler_ultraflash_pairs.sh worker 7 2 3
bash run_prepare_upscaler_ultraflash_pairs.sh finalize 3
bash run_prepare_ultraflash_flow.sh 0 1  # CPU 光流缓存

# 两组预算、数据和初始化相同；可顺序运行。
bash run_train_upscaler_ultraflash.sh 5 ewarp 2000
bash run_train_upscaler_ultraflash.sh 6 temporal 2000

# 两组完成后，先全部 latent/重载/因果检查，再像素解码。
bash run_evaluate_upscaler_ultraflash.sh 5 latents 2000
bash run_evaluate_upscaler_ultraflash.sh 5 decode 2000 0 1
bash run_evaluate_upscaler_ultraflash.sh cpu finalize 2000 1
```

正式数据只有全分片完成且 SHA、身份、split、所有权校验通过后才生成四份 manifest 和 `READY.json`。每个 worker 的 shard 总数必须与 plan 一致，分片目录独占，失败分片和已有输出不自动覆盖。光流分片也应在完整 pairs finalize 后启动，同一个 index 只启动一次。

解码入口显式传入环境中的 `H3_BASE_MODEL`。可用 decode 的最后两个参数分片，但各分片须使用相同总数，并全部结束后才调用 CPU finalize，传入对应 shard 总数。finalize 校验 latent summary/像素指标哈希、分片身份、四条视频的完整覆盖及全部视频/对照图存在后，才将总状态置为 `completed`；未完成不能作为最终结果发布。

独立损失测试：`python -B test_ultraflash_ewarp.py --device cpu`。`run_upscaler_ultraflash_preflight.sh <物理卡>` 会依次执行 CPU/NPU 测试与 4 步预检；需要事先准备包含 plan、train/dev manifests 和匹配 flow 的 `H3_ULTRAFLASH_PREFLIGHT_PAIRS`，也可将其设为已 finalize 的 `H3_ULTRAFLASH_PAIRS`。预检输出目录必须为空；预检不会打开 test。

恢复示例：在原输出目录继续 `bash run_train_upscaler_ultraflash.sh 5 ewarp 2000 --resume "$EXP_ROOT/upscaler_ultraflash/ewarp_steps2000/latest.pt"`。保存训练 contract、优化器、CPU/NPU RNG、确定性样本顺序种子、不可变 best checkpoint 引用及日志记录；恢复时校验源码/数据/配置哈希，并用 checkpoint 内记录修复超前日志。**尚未验证中断恢复与连续训练逐位一致**。恢复应使用同一份训练/损失源码；checkpoint 加载重新安装 adapter，但不重复 aligned 初始化。

正式结果和视频见[结果目录](../../results/h3_upscaler_ultraflash_20261104/README.md)。16 条 test 上，eWarp 相对同预算 temporal 的 eWarp L1 降低 16.535%，latent MSE 增加 3.261%；该指标取舍不能等同于视觉质量全面提升或已证明泛化。两组最佳检查点均在最后一步，不能据此声称充分收敛。
