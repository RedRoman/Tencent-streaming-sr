# v3 方案：overlap 滑窗流式 teacher（零训练）与 causal student 的互补评估

**日期**：2026-11-05 ｜ **状态**：方案定稿，未实施 ｜ 前置实验：method-2 v1（完成，负结果）、v2（训练中）

## 1. 动机

当前两条流式化路线占住质量-延迟曲线的两个端点，本方案补上中间点并给出可测的互补关系：

| 路线 | 延迟 | 训练成本 | 质量上界 |
|---|---|---|---|
| teacher 全序列（非流式基线） | 整段 | 0 | teacher 原生 |
| **v3：overlap 滑窗 teacher（本方案）** | K latent 帧（K=感受野半径，待实测） | **0（无训练）** | 理论上 = teacher 全序列（逐 bit 一致，见 §2） |
| causal student（method-2 在训） | 0 | 3000 步 | 相对 teacher 有 gap（v1 实测见 §4） |

三个点连成 "causalization 代价曲线"：全序列 → K 帧延迟 → 零延迟，每段 gap 可测。

## 2. 理论依据与关键风险（已核实代码）

对纯 3D 卷积网络，滑窗 overlap 不是近似而是**精确等价**：感受野有限，窗口取足 K 帧未来后，
窗口内各帧的计算图与全序列前向完全相同，输出应逐 bit 一致（同精度下）。

对 `causal_upscaler.py` 中 `temporal_structure()` 的分析以及已发布结构（LBH/Open-H3 upscaler）确认：

- **无 attention**：`CausalUpscalerAdapter` 显式检查并拒绝含 attention 的结构（"The published checkpoint has no attention"）。安全。
- **temporal Conv3d 全部 stride-1、zero-pad、对称**：adapter 强制校验（`stride[0]==1`、`padding[0]==dilation*(k//2)`）。这是精确滑窗成立的必要条件，已满足。
- **skip 路径为 identity/1×1**，不扩大时间感受野（`temporal_structure` docstring 已注明）。
- **主要风险：GroupNorm**。原模型的 GroupNorm 统计在**整段 clip 的时间维**上聚合（`original_groupnorm_dependency: "whole input clip"`）。滑窗后 GN 统计只在窗口内聚合，因此输出不再严格等于全序列结果——这是唯一打破逐 bit 一致性的算子。
  - 缓解 A：H3 latent 序列时间上变化平滑、窗口内统计量接近全序列统计量，预期误差小但**必须实测**，不能假设。
  - 缓解 B（如误差超阈值）：对每个 GN 层缓存/外推全序列统计量不可行（流式时无未来），退路是扩大窗口 + 接受近似，或按帧累计 running statistics（因果 GN，会改变函数，需重新蒸馏）。
- **窗口开头的历史帧**：必须用上一窗口的缓存（真实过去）而非 zero-pad，否则开头 K 帧与全序列不一致。因果化 surgery 文件里已有 per-frame GN 与 left-pad 的先例可复用其 pad 逻辑。

**结论**：方案可行性等级 = "Conv3d 精确 + GroupNorm 近似（需实测误差）"。

## 3. 实施步骤（全部无需训练）

1. **精确感受野**：跑 `temporal_structure(teacher.model)` 得 `causal_history_latent_frames`（即 K）。按 k=3/5 的层数直接算出。24fps 下 1 latent 帧 ≈ 1.2 pixel 帧（VAE 时间压缩 4x+1），换算成秒级延迟报告。
2. **滑窗 wrapper**：输入流式 latent 帧序列，维护 K 帧 past 缓存；每步对 [cache + 新帧] 前向，仅吐出"感受野已完整覆盖"的帧（即新吐 1 帧、缓存尾部 K 帧延迟输出）。每窗口前后 frame 精度 fp32 与 eval 中 teacher 一致。
3. **精确性校验**（复用 `eval_student_5s.py` 的因果校验框架）：对 3 个 5s clip，比较滑窗输出与全序列输出逐元素 max_abs。判据：GroupNorm 误差 < 1e-3（latent 归一化空间）视为可接受；同时跑 future_cut / prefix_cut / streaming consistency 三项。
4. **质量-延迟曲线**：对滑窗 teacher 与 causal student 在同一 holdout 上算 vs-reference 像素 PSNR + LPIPS，填入 §1 表格，形成最终对比图。
5. **延迟叠加分析**：检查 regenerator 生成端是否天然有 ≥K 帧缓冲；若有，v3 的等效延迟可被生成端 lookahead 覆盖，v3 可能直接成为部署方案，student 转为低算力备选。

## 4. 现有实测数据（本方案质量曲线的已知点）

来源 `runs/student_5s_eval/summary.json`（v1 student，vs decoded reference）：

| clip | teacher 全序列 | v1 student (causal) |
|---|---|---|
| VIDGEN5S_0380588 (holdout) | 30.11 dB | 22.24 dB |
| VIDGEN5S_0336361 (train) | 23.87 dB | 23.10 dB |
| VIDGEN5S_0515402 (train) | 22.83 dB | 22.33 dB |

注意训练 clip 上 student 与 teacher 差距很小（蒸馏保真好），holdout 上 gap 全部来自泛化——
与 v1 训练结论（2-clip 过拟合）一致。v2 像素合同训练中（step 2160 时 train LPIPS 0.36→0.058），
其 holdout 像素指标是判断"loss vs 数据"瓶颈归因的关键。

causal student 因果性校验（v1，holdout）：future_cut 全 0.0，prefix_cut/streaming ≤4.8e-7，passed。

## 5. 与 v2/下一轮的联动决策

- 若 v3 滑窗 GN 误差小且质量≈全序列 teacher：v3 进入部署候选；student 的定位变为零延迟/低算力替代。
- 若 v2 holdout 像素指标仍差：优先扩训练数据（真实配对 + bilinear 合成对混合），loss 不再是第一嫌疑。
- v3 的滑窗 wrapper 校验代码完成后回填本文件 §3.3 的实测数字并更新 GitHub。

## 6. 复用资产

- `code/causal_upscaler.py`（仓库已有）的 pad/结构分析逻辑、`eval_student_5s.py` 的因果校验框架。
- v3 无新训练，全部在空闲卡或 CPU 可验证（teacher 单窗口前向 ~1s 量级，32193 实测）。
