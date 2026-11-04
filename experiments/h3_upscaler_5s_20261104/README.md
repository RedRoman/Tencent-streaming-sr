# 固定 upscaler 权重的完整 5 秒评估

本轮沿用上一轮的发布权重、`aligned_past` 因果初始化权重和 **50 步微调后的因果权重**，只将评估片段延长为真实源视频的 **120 帧、24 fps、5 秒**。没有新增训练，没有加载 H3 diffusion refiner。仍是相同的三条源视频，不是新数据泛化测试。具体数值、诊断和完整视频见 [结果报告](../../results/h3_upscaler_5s_20261104/README.md)。

## 视频长度与 VAE 的实际时间处理

`prepare_upscaler_5s_pairs.py` 先对每条源视频执行 `fps=24` 的完整帧计数，再严格读取前 120 帧。外部 FFmpeg 路径没有 `tpad`、循环、慢放或末帧保持回退；输出使用 passthrough 同步。程序保存原视频 SHA256、fps 滤镜输入/输出/丢帧/重复帧统计及日志，不足 120 帧或严格 reader 产生重复帧时明确报错，不把短视频补成 5 秒。

三条视频继续使用 `VIDGEN5S_0336361`、`VIDGEN5S_0515402`、`VIDGEN5S_0380588`。所有分支共享同一批 120 帧源 RGB：HR 先做 672×384 的 cover-area resize 和中心裁剪，再经精确 2×2 区域平均、取整得到 336×192 LR。两路分别经同一个冻结 FP32 H3 VAE 编码，使用同一后验采样 seed 42。

**外部不补时长，不表示 VAE 内部没有 padding。** 发布 VAE 的时间路径如下，metadata 会逐项记录：

```text
120 帧真实源 RGB
  → VAE 内部复制最后一帧，补 16 帧至 136（8 × clip_length 17）
  → 保留实际 37 个 latent 帧
  → VAE 原始解码 124 帧
  → 裁掉尾部 4 帧，输出前 120 帧，按 24 fps 保存为 5 秒
```

LR latent 为 `[24,37,12,21]`，HR latent 为 `[24,37,24,42]`，不会裁回旧实验的 13 个 latent 帧。最终视频来自解码裁剪，不通过降低帧率把原来的短输出拉长。

## 既有权重与检查范围

本轮比较原版非因果 upscaler、因果初始化 `causal_before`、50 步后 `causal_after`，以及匹配 HR latent 的重建。代码读取上次试验的 `initial_model.pt`、最终 `checkpoint/model.pt` 和 `adapter_config`，验证最终权重与发布权重 SHA256；不重新初始化已转换的时间核，也不更新参数。因果算子、N/D 约定和逐层缓存实现见相邻 [因果 upscaler 实验](../h3_causal_upscaler_20261104/README.md)。

对两个因果分支、全部三个片段进行 37 帧测试：

- 在 cut `1/13/25` 后扰动未来输入，检查共同的过去输出。
- 比较单独长度 `1/13/25` 的前缀与整段对应输出。
- 比较逐层缓存流式执行与整段执行，分块为 `[5,5,5,5,5,5,5,2]`、37 个单帧块、`[7,13,17]`，并断言缓存记录已处理 37 帧。

一致性判断阈值为 `1e-4`；结果写入报告，失败会非零退出。完整结构最长卷积路径是 149 个 latent 帧，此轮仅覆盖 37 帧上下文，不能称为覆盖了任意时长。VAE decoder 仍非因果，upscaler 的输入 latent 因果检查不等于端到端像素流式因果。

新评估重新编码了 120 帧，编码上下文和后验采样张量均区别于旧的短片段。即使继续使用 seed 42，也不能假定每个对应 latent 的后验噪声逐元素不变。因此不能把新旧指标差异全部归因于“视频更长”。本轮各分支共享新的匹配输入与目标，可在本轮内部比较；旧的 43 帧结果只作为开发背景。

## 环境与运行

此目录 `env.sh` 继承相邻 `h3_causal_upscaler_20261104/env.sh`，继续载入旧训练 helper、VAE 和设备锁，并将当前目录放到 `PYTHONPATH` 首位。可用 `H3_CAUSAL_UPSCALER_CODE_DIR`、`H3_UPSCALER_CODE_DIR`、`H3_REFINER_CODE_DIR` 覆盖对应 helper 目录。

在本目录执行，替换所有占位路径：

```bash
export EXP_ROOT=/path/to/h3-work
export H3_BASE_MODEL=/path/to/h3-base
export H3_UPSTREAM_DIR=/path/to/Open-H3-Regenerate-2K
export H3_DIFFUSERS_SRC=/path/to/diffusers-patched/src
export H3_PYTHON_BIN=/path/to/npu-venv/bin
export H3_CANN_ENV=/path/to/Ascend/cann/set_env.sh
export H3_FFMPEG_BIN=/path/to/ffmpeg/bin
export NPU_LOCK_DIR=/path/to/shared-device-locks

bash run_prepare_upscaler_5s_pairs.sh <idle-physical-npu-id>
bash run_evaluate_upscaler_5s.sh <idle-physical-npu-id>
```

外部工作目录需要原 `data/{train,holdout}.jsonl` 与源视频、发布 upscaler 权重，以及已验证的 `upscaler_causal_finetune/aligned_past_steps50/`（含初始权重、最终 checkpoint 和 summary）。`H3_UPSTREAM_DIR` 需包含 `open_h3/`，固定版本及 Ascend runtime 依赖沿用相邻实验。每次运行前重新确认设备空闲；入口会持有共享锁并检查 `npu-smi`。

新配对数据及源证明保存到 `upscaler_5s/pairs/`，评估与视频保存到 `upscaler_5s/evaluation/`。已有目录不覆盖，旧的配对数据、训练 checkpoint 和结果不修改。需要重试时使用新的工作目录，或直接调用 Python CLI 指定新的 `--output`。

像素指标在压缩前统计，主要目标是同一 FP32 VAE 对匹配 HR latent 的重建；对源 RGB 的指标是次要参考，包含 VAE 重建和时间对齐误差。各分支单独输出视频，并提供原版 → 因果初始化 → 50 步后 → reference 的四列对照；观察结论统一见 [结果报告](../../results/h3_upscaler_5s_20261104/README.md)。
