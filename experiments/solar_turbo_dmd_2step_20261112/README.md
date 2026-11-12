# Solar Turbo 两步蒸馏训练方式

本说明对应本项目实际执行的 `turbo_training_v4`：冻结已经完成 task2500 训练的 Solar 因果四步 refiner，用**条件分布匹配蒸馏（DMD）和学生自生成历史**训练两步学生。

目标是减少去噪次数，同时尽量保留四步模型的恢复质量。本次 pilot 完成 **200 次学生更新、1020 次 critic 更新**，原训练进程正常结束。训练完成不等于画质验收；本文不把候选提升为已验收模型。

这是本项目的条件、块因果 DMD 配方，不能称为对上游 SolarRenderer 官方 Turbo 训练配方的原样复现，也不是 Open H3 anchored refiner 路线。

## 1. 三个角色与 LoRA 的作用

| 部分 | 教师 | 学生 | critic |
|---|---|---|---|
| H3 基座及已合并的 Solar 因果控制、四步适配 | 冻结 | 同一冻结基础 | 同一冻结基础 |
| 已完成 task2500 的任务 LoRA | 冻结 | 冻结 | 冻结 |
| 新增学生 LoRA | 不使用 | 训练 | 不使用 |
| 新增 critic LoRA | 不使用 | 不使用 | 训练 |

学生和 critic 分别新增独立的 rank16 分支，作用于 50 层 attention 的 Q/K/V/out，分别有 400 个可训练矩阵、40,140,800 个 FP32 参数。A 以小随机值初始化，B 初始化为零；因此新增分支最初不改变已有模型的函数。

各角色共享冻结基座存储，但不共用可变分支或历史 KV。教师与 critic 必须在相同的学生历史内容上，按各自权重重新构建 KV；参数版本变化后不能复用旧版本缓存。

**task LoRA 先学习恢复任务；Turbo 学生 LoRA 再学习用两步完成恢复。** critic 是训练时估计学生生成分布的去噪模型，不是真假二分类 GAN 判别器，部署时不需要它。

Upsampler、decoder、原 task LoRA 和基座均未更新。此训练没有让 decoder 参与反向传播，因此不能称为联合训练 refiner 与 decoder 的适配。

教师背景与原 task LoRA 见[此前的 Solar task2500 发布说明](https://github.com/RedRoman/Tencent-streaming-sr/blob/dc90e4dee96da8751279e38a733e3e5eeaf358f4/experiments/solar_refiner_tf2500_20261111/README.md)。

## 2. 两步学生实际如何生成

使用 H3 的速度参数化：

```text
x_sigma = (1 - sigma) * x_clean + sigma * epsilon
v = x_clean - epsilon
x0_hat = x_sigma + sigma * v
```

教师推理日程为 `1 → 36/37 → 12/13 → 0.8 → 0`，学生为 `1 → 12/13 → 0`。

学生先从噪声预测一次干净 latent，再按第二个噪声强度重新加入一份独立噪声，进行第二次干净 latent 预测。这里保留随机重新加噪采样，没有把教师偷偷改为确定性 Euler 求解。

每个块完成后，还需要一次 clean KV 提交前向。因此完整 67 latent 的 14 个块中，教师是 70 次前向，学生是 42 次前向；“两步”指两次去噪，不包括缓存提交，不代表整个管线必然快两倍。

## 3. 一轮训练做什么

1. **用当前学生实际跑两步生成。** 之前块的历史来自学生完整两步后的最终输出，并提交到历史 KV。GT 或教师生成的历史不会替换这些学生历史。
2. **随机选择当前块的第一或第二次干净预测作为训练对象。** 保存对应噪声、条件、位置、参数版本及历史，使反向前的重放精确匹配。
3. **更新 critic 五次。** 给选中的学生输出重新加噪，训练 critic 预测其去噪速度。目标是学生输出减去所加噪声，采用速度 MSE。
4. **计算教师和 critic 的差异。** 两者在同一个加噪学生样本、相同条件与学生历史下预测干净 latent；用差异构造 DMD 更新方向。
5. **精确重放选中的学生前向并更新学生一次。** 历史输出和历史 KV 均停止梯度，不对整段视频做贯穿全部历史的反向传播。

即使本轮选择第一步预测来做 DMD，历史仍使用完整两步生成的最终结果。教师在训练中作为冻结的去噪/score 参考，并不是每轮先生成一段四步视频，再让学生逐像素复制视频。

当前实现每块最多 5 个 latent，历史窗口为 6 个块：**块间因果、块内双向**。它不是逐 latent 严格因果的定义。

## 4. 损失函数

令 `y` 为选中的学生干净预测，`r` 为冻结教师对加噪 y 的干净预测，`f` 为 critic 对同一输入的干净预测。停止梯度的 DMD 方向为：

```text
g = (f - r) / max(mean(abs(y - r)), 1e-4)
g = per-sample RMS cap(g, 10)
L_DMD = 0.5 * mean((y - stop_gradient(y - g))^2)
```

这里的 surrogate loss 用于施加 DMD 梯度，并非直接计算学生与教师四步终点的配对 MSE。

只有选中第二步最终输出时，额外加入真实 HR 对应 latent 的保真约束：

```text
L_student = L_DMD
          + 0.05 * L1(y, HR_latent)
          + 0.01 * L1(delta_time(y), delta_time(HR_latent))
```

时序项比较相邻 latent 的变化误差，并包含可用的前一块末尾边界；它不是惩罚运动本身。GT 仅用于损失目标，不输入模型，也不充当生成历史。

这些约束发生在 **latent 空间**。训练没有穿过视频 decoder 计算 RGB 或 LPIPS 损失，所以它不能直接保证像素感知质量、无闪烁或 decoder 适配。

## 5. 实际优化与数据配方

| 项目 | 实际配置 |
|---|---|
| critic 预热 | 20 次更新 |
| 后续更新比例 | 每 1 次学生更新，5 次 critic 更新 |
| pilot 终点 | 学生 200 次，critic 1020 次 |
| 学生学习率 | 5e-6，前 20 次学生更新线性 warmup |
| critic 学习率 | 1e-5 |
| 优化器 | 两份独立 AdamW，betas=(0.9, 0.99)，weight_decay=0 |
| 梯度裁剪 | 全局梯度范数上限 1 |
| 新 LoRA 初始化 | A 标准差 0.01，B 为零 |
| 训练片段 | Charge_train_3856_4081、Caminandes3_train_168_393 |
| 常规训练窗口 | 7 latent，原生 latent 网格裁剪 32×64 |
| 较长自生成历史 | 每 20 次学生更新使用 37 latent 窗口 |
| 全空间诊断 | 第 10 次及之后每 50 次学生更新，12 latent、68×120 |
| 开发集 | WingIt3586、ElephantsDream13070、WingIt300，与训练片段分离 |

预检另外做过 1 次学生和 1 次 critic 的可丢弃诊断更新，单独记录；正式 pilot 前新分支恢复初始化，并使用新的优化器。因此诊断更新不计入 200/1020，也没有重置原 task2500 的训练状态。

v3 因静音音频布局索引不匹配而失败，正式训练更新为零；v4 修正了静音布局及 CPU 索引检查。该修正不改变 DMD 损失、采样日程或优化器配方，历史失败保留。

## 6. 部署形式与结论边界

部署保留：

```text
冻结 H3 + Solar 因果控制/四步基础
    + 冻结 task2500 LoRA
    + Turbo 学生 LoRA
    → 两步生成
    → 冻结 decoder
```

不加载 critic 或其优化器。

本项目已证明这套实现能执行真实更新、保存训练状态并产出两步结果；不能据此推断训练配方已达到质量验收。仍需分别检查同条件四步教师、直接减为两步而不蒸馏、无 refiner 三种对照，以及完整视频的感知质量、时序误差和连续视觉审片。

本次提交是训练方法说明，不宣称新设备一键复现、实时吞吐、逐 latent 严格因果或完整生产条件验收。它不发布训练 checkpoint、optimizer、RNG、KV、训练数据或基座 payload。

## 7. 对应的执行与源码标识

以下标识用于区分实际执行版本与早期设计草稿。源码 SHA 在发布本说明前已与 v4 冻结 manifest 逐项匹配。

| 标识 | SHA-256 |
|---|---|
| v4 源码 manifest | `c5618bf5a1cfe84d9862abc344132305cd028ba99e8828ecf4e6ec77ca58769a` |
| 教师 task2500 原 checkpoint | `aec47799fbdf3425b8d11828653eaccc0349faa011919a8bd2fbe9fd9a9de422` |
| Turbo pilot 最终原 checkpoint | `b7cb43a8025297532cad2aa82d6505b389ac6048748b66be7a4e9908ea752ce6` |
| train.py | `d3f21768446f7f72df411ea8787d1f79310547cd803c18f64388858e5ab46145` |
| objectives.py | `72ab6db9cb1d2b4b75c5906c2d94a87aa844f1a7d2cd0c2298788e4c3b4937e4` |
| stream_session.py | `5248c97f72a5a01bcee27634de6c349712d912133193879ea8b0e126b56b7a53` |
| runtime_roles.py | `f5be2c186abf3330530c66efb04612f3ace206f797a43cdcf9b6c70d6d368152` |
| role_adapter.py | `47f85cadb2b55668e0a7e6dea085d6cedba0f116c718caf89e34f0d6c30b1ffa` |
| 原 recipe.json | `4f1398d94e4959364430f0827f8d2bec29264558ccfe0bfa2002021486bb0917` |

原 recipe 含准备阶段的历史状态字段；实际更新数和完成态取自训练完成凭证及评估 checkpoint 绑定，不能把旧状态字段当成最新进度。最终 checkpoint 的文件回读校验与终点 live reload 是不同检查，不能相互替代。

原 checkpoint SHA 与此前 Release 导出的纯推理 task LoRA SHA 不同，因为文件封装和所含状态不同。本文只列原始标识，不将其冒称为已发布的 Turbo 推理权重。
