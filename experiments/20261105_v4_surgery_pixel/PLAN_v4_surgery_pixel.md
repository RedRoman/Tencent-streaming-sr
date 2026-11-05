# v4：surgery 因果化 Open-H3 upscaler + v2 像素合同 + 扩充数据

**日期**：2026-11-05 ｜ **状态**：训练中（step 130/3000 时快照） ｜ 提议人：用户
**前提**：method-1（aligned-past surgery，几乎未训练）、v2（15M causal student + pixel contract，完成）

## 1. 单一问题

预训练初始化（surgery 后的原版 upscaler 权重）+ 因果架构 + 像素级蒸馏合同 + 扩充数据，
能否显著优于 v2 的从零设计 15M student？回答"容量与预训练结构先验在因果化中的价值"。

## 2. 架成

- **可训练模型**：原版 Open-H3 upscaler（193M 结构）经 method-1 的 aligned-past 权重手术
  （k=3 时序卷积核重排为 [0,过去,当前]，丢弃未来系数）+ per-frame GroupNorm + 左侧补零
  （`CausalUpscalerAdapter` 强制校验无 attention、全部 stride-1）。
  可训练参数 345,280,216（fp32）。初始化 = `--init fresh`（原版权重 + 手术）。
- **冻结 teacher**：未改造的双向原版 upscaler（监督目标，与 v2 完全一致）。
- **loss 合同**：逐行复刻 v2（`train_student_pixel_v2.py:184-206`）：
  latent MSE + 0.1 temporal diff + 1.0 pixel L1（raw）+ 1.0 LPIPS(alex) + 0.333 wavelet
  (1-level DWT 全带)；每步随机 5 latent 帧窗口解码（17 pixel 帧），梯度穿过冻结 VAE decoder。
- **数据**：`upscaler_5s_v4exp/pairs` — 训练 10 clip（原 2 + 新增 8 个严格 120 帧 FFmpeg 证明的
  源视频，合同与原 prepare 脚本逐项一致），holdout 0380588 逐字节继承（SHA256 校验），
  与 v1/v2 保持可比。

## 3. 与 v2 的单变量性（诚实声明）

v4 同时改变了三件事：数据（2→10 clip）、架构与初始化（15M 从零 → 345M surgery）、
不变的是 loss 合同与评测协议。因此干净的归因链需要 v2-10clip 对照（loss/架构不动、只换
数据），v4 训完后补跑，成本约一晚。

## 4. 训练前验证门（已通过）

| 门 | 结果 |
|---|---|
| streaming consistency（逐帧缓存 vs 全序列并行） | **0.0（精确）** |
| 数据形状 (24,37,12,21)/(24,37,24,42) | passed（10+1 clip） |
| activation checkpointing | 开启（193M fp32 全序列前向 + 冻结 teacher + 带梯度 VAE decode 挤 61G HBM） |

训练后还会跑 future_cut / prefix_cut / streaming 三项因果校验（与 v2 同框架）。

## 5. 运行环境要点（32193）

- 卡：NPU6（启动时确认无进程）。
- `PYTHONPATH` 需前置 `/home/ma-user/.local/lib/python3.12/site-packages`
  （pytorch_wavelets / lpips 在用户 site，env.sh 设了 PYTHONNOUSERSITE=1）。
- 一次 CANN 编译器残留错误（tbe import/GE init 500001）重跑即恢复，未复现。

## 6. 早期训练信号（快照，非结论）

| step | loss | l1 | lpips | wav |
|---|---|---|---|---|
| 10 | 0.665 | 0.222 | 0.153 | 0.466 |
| 120 | 0.391 | 0.125 | 0.118 | 0.272 |

稳定速率 ~5.45 s/step（v2 为 4.5），3000 步 ETA ~4.4h。

## 7. 预注册判断标准（holdout 像素 PSNR，vs decoded reference）

- 26.5–28 dB：改造路线成立（method-1 的 25.45 被像素合同+数据+训练补回）
- 接近 28.80（teacher 原生）：超预期，因果化损失基本清零
- 仍在 25.5 附近：surgery 丢弃的未来信息不是轻量训练能补的 → 升级方案
  （蒸馏初始化补未来系数 / 加大步数 / aligned50 重训）

## 8. 复用资产

- `experiment/prepare_upscaler_5s_pairs_v4.py`：继承 + 新增 clip 的数据准备（SHA256 继承校验）。
- `experiment/train_v4_surgery_pixel.py`：surgery 模型 + v2 loss 合同 + 预检门；
  支持 `--init aligned50` 直接从 method-1 checkpoint 续训（reload 规则：永不重复手术）。
