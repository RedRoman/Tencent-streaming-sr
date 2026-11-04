# H3 refiner：单步 Self Forcing + DMD smoke（2026-11-04）

此目录保存已经在昇腾端运行的实验代码快照。**训练、checkpoint 独立重载和视频解码均已完成，但画质不合格。** 留出片段中，因果单步分支从第一帧就出现输入里没有的人脸和上半身；6 次生成器更新没有修复这一偏移。此结果只能验证链路，不能证明 Self Forcing 有效或无效。

对应证据、图像和完整训练事件记录位于 [results](../../results/h3_refiner_self_forcing_20261104/README.md)。公开快照对基础设施信息做了脱敏：用户路径、主机名和 SSH 连接标识已替换为占位符，脚本路径改为环境变量；数值算法、权重版本和已记录的数值结果保持不变。此目录不是可在任意机器直接执行的安装包。

## 已完成的设置

| 项目 | 实际运行值 |
| --- | --- |
| 模型 | MiniMax-H3 base + Open-H3-Regenerate-2K 发布的 refiner LoRA、Sharp 分支 |
| 机器 | 2 张 Ascend 910B3，FSDP2；连接信息不公开 |
| 数据 | 2 个训练片段、1 个留出片段；留出仅指本次 3 片段实验内互斥 |
| 训练片段 | `VIDGEN5S_0336361`、`VIDGEN5S_0515402` |
| 留出片段 | `VIDGEN5S_0380588`：手部拆巧克力锡纸 |
| 草稿 | 原视频降至 336×192，经 H3 VAE 编码；没有运行文生视频草稿生成 |
| 输出 | 672×384，43 帧，24 fps，13 个 latent 帧；不是 2K 测试 |
| 更新次数 | 6 次生成器更新，4 次 fake warmup + 6 次 fake 更新，共 10 次 |
| 训练参数 | generator / fake 各自 rank-8 LoRA，学习率 `1e-4`；发布的基础 refiner 参数冻结 |
| 生成器 sigma | `[0.909375, 0]`，每个时间块 1 次去噪 |
| 时间块 | `5 + 5 + 3` latent 帧，因此整段需要 3 次 transformer 调用 |
| 对照 | 冻结 dense 三步、冻结 dense 单步、因果单步训练前、因果单步训练后 |
| 重载验证 | 独立进程加载训练 LoRA 后，留出 latent 最大绝对误差 `0` |

## 如何因果化

`temporal.py` 将 H3 的 packed `[text | audio | video]` 序列划分为静态条件和视频时间块。视频查询仅可读静态条件、当前块及更早块；静态 text/audio 查询仅可读静态条件，以阻止未来视频经过条件 token 间接泄漏。

生成时依次运行 5、10、13 帧前缀。每个前缀由学生自己已经生成的 clean 历史和当前块的 noisy upscaled latent 组成；不送入未来视频块，也不把真实历史作为学生上下文。H3 使用 `t = 1 - sigma`，历史时间为 `1`，当前块时间为 `0.090625`。历史输出和当前查询所读取的历史 K/V 均停止梯度；每次更新只对截断前缀的末块计算生成器损失。当前块内部仍双向注意力，因此是**块因果**。

teacher 和 fake score 保持 dense attention。DMD 对同一学生生成样本加噪，使用 `x0 = xt + sigma * v`，梯度为 `(fake_x0 - real_x0) / clamp(mean(abs(student_x0 - real_x0)), 1e-5)`。fake 通过 flow-matching 更新；generator 通过该 DMD 方向更新，二者参数和优化器独立。

这是前缀重算实现，没有 KV cache 加速。latent upscaler 和 VAE 仍含时间维非因果操作，因此整个输入到像素输出的 pipeline 不具备严格的流式因果性。pure DMD 也没有加入成对重建或 LR 一致性损失。

## 外部来源与权重

此快照不包含上游源代码、模型权重、训练缓存、原视频、认证信息或下载控制器。部署前需要取得下列固定版本，并遵守其各自许可。

| 依赖 | 固定版本 / 来源 |
| --- | --- |
| Open-H3-Regenerate-2K 源码 | [NVlabs/Sana](https://github.com/NVlabs/Sana/tree/781e8341fb3b29d3aea5d50265772e9619340b66/models/Open-H3-Regenerate-2K)，commit `781e8341fb3b29d3aea5d50265772e9619340b66` |
| refiner、Sharp、latent upscaler 权重 | [Efficient-Large-Model/Open-H3-Regenerate-2K](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K/tree/50c09cf1df5f9262b9279868c1c666bd5fe3476b)，revision `50c09cf1df5f9262b9279868c1c666bd5fe3476b` |
| H3 base | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/6818f6c32d12b210915e44ad56a4228c2608f160)，revision `6818f6c32d12b210915e44ad56a4228c2608f160` |
| `adaln_t_table` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/e5eb578a89295337b8ff433a035929ce0279e0b6)，revision `e5eb578a89295337b8ff433a035929ce0279e0b6`，从 `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` 提取同名 F32 tensor `[1025,8]` |

这里使用的 latent upscaler **已有公开实现及权重**。Open-H3-Regenerate-2K 发布仓库采用社区项目 [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) 的 3D convolution 节点；上游 `open_h3/upscaler.py` 明确说明保留作者架构、归一化和尺寸处理，只为脱离 ComfyUI 加载做适配。因此不需要因为“upscaler 未公开”而从零训练。此快照的 upscaler 使用发布权重、CPU FP32 推理，**没有启动 upscaler 微调**。

四个额外权重文件均已做完整 SHA256 验证，见结果目录 `asset_sha256_manifest.json`。base 的 14 个 shard 以已下载缓存的 LFS digest 和当前文件大小匹配官方固定版本；没有重新读取约 66 GB 权重做全文件 SHA256，见 `base_provenance.json` 的限制说明。

## 可配置运行布局和复现顺序

运行时将代码目录与外部工作目录分开。代码保留在本目录，外部工作目录由 `EXP_ROOT` 指定，数据布局如下：

```text
assets/lora/             # refiner LoRA、Sharp、adaln_t_table.bin
assets/latent_upsampler/  # minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors
data/train.jsonl          # 2 个训练样本
data/holdout.jsonl        # 1 个留出样本
data/samples/*.pt         # draft/upscaled/reference/text/audio/metadata
merged_transformer/      # CPU 合并发布 refiner LoRA 后生成
outputs/                 # train.jsonl、adapters.pt、各对照 latent
decoded_results/         # 解码视频、图像、metadata
```

原环境使用 Python 3.12、torch `2.7.1+cpu` 配合 torch_npu `2.7.1.post2`、peft `0.20.0`、在集群适配的 diffusers `0.40.0.dev0` 和 CANN `9.0.1`。此快照没有包含该 runtime 补丁，因此仅凭本目录不能保证在全新环境复现。先配置环境变量，再运行入口：

```bash
export EXP_ROOT=/path/to/h3-refiner-work
export H3_BASE_MODEL=/path/to/h3-base
export H3_UPSTREAM_DIR=/path/to/Open-H3-Regenerate-2K
export H3_DIFFUSERS_SRC=/path/to/diffusers-patched/src
export H3_PYTHON_BIN=/path/to/npu-venv/bin
export H3_CANN_ENV=/path/to/Ascend/cann/set_env.sh  # 如当前 shell 尚未加载 CANN
export H3_FFMPEG_BIN=/path/to/ffmpeg/bin           # 如 ffmpeg 不在 PATH 中
export H3_SOURCE_MANIFEST=/path/to/source_manifest.jsonl
export NPU_LOCK_DIR=/path/to/shared-device-locks  # 与同机其他作业约定同一目录
```

`H3_UPSTREAM_DIR` 需包含 `open_h3/` 包。调用脚本自动以自身位置作为 `EXP_CODE_DIR`，不要求将源代码拷进工作目录。`NPU_LOCK_DIR` 未设置时使用 `/tmp/h3-npu-device-locks`；设备启动前还会检查 `npu-smi`。`H3_RUN_LABEL` 可选，用于日志中不暴露系统主机名的运行标签。

1. 准备上述权重并核对 SHA。`prepare_data.py --help` 说明数据生成接口；以 `H3_SOURCE_MANIFEST` 指向已有数据 manifest。应显式选择 `--draft-mode source --audio-mode silence --upscaler-device cpu --upscaler-precision fp32`，固定 train indices `0,3` 和 holdout index `2`。原视频及 text/audio/HR latent 缓存需另行提供，仓库不包含这些数据。
2. 运行 CPU temporal 语义测试、双卡角色/冻结检查；`test_full_runtime.py` 是 base 兼容性预检，不是 refiner 质量验证。
3. `bash run_merge.sh` 在 CPU 合并发布的 refiner LoRA。Sharp 分支在 runtime 单独加载。合并函数先验证 LoRA 完整 SHA256，输出 `MERGED.json`。
4. 重新确认设备空闲后执行 `bash run_train.sh <idle-device-ids>`。训练记录仅用于复现配置，不代表任意设备当前可用；`locked_run.py` 同时检查 advisory lock 和 `npu-smi`。
5. 训练进程退出后执行 `bash run_eval.sh <idle-device-ids>`，独立加载 `outputs/adapters.pt`。checkpoint 仅保存适配器及配置，不含优化器/RNG 状态，支持推理重载和 warm start，不是精确续训存档。
6. 使用一张重新确认空闲的卡执行 `bash run_decode.sh <one-idle-device-id>`，再运行 `python summarize_results.py --root "$EXP_ROOT"`。

以上命令在本代码目录执行。`wait_and_merge.py` 是原运行等待已验证权重导入的可选工具；本快照不包含下载和跨机桥接控制器。资产已准备齐全时直接运行 `run_merge.sh` 即可。

## 解释范围

当前对照同时改变了注意力、前缀长度和历史的 clean/noisy 分布。额外人物在因果分支**训练前**已出现，所以不能归因于这 6 次 DMD 更新；第一帧已经偏移也说明不能只用后续历史误差累积解释。dense 到块因果的分布变化是待检验假设，尚未由这个原始 smoke 单独隔离。补充的同前缀消融现已完成，见结果目录 `causality_ablation/`：13 帧 dense / static-mask 没有额外人物；5 帧 dense / static-mask 均出现类似人物；5 帧 static-mask 在 sigma 降为 `0.3` 后没有该人物并保留锡纸。此单片段证据削弱了“static mask 是主要原因”的初始猜测，短视频前缀与高噪声更直接相关。全长静音 audio 条件仍固定，尚未隔离其影响；不能据此推出通用修复。代码见 `ablate_causality.py`、`run_causality_ablation.sh` 和 `run_decode_ablation.sh`。
