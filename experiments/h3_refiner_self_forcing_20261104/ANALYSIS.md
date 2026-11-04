# 因果化与新增人物：实现审查和实测消融

2026-11-04。对象是 **Open-H3-Regenerate-2K 的 H3 refiner**，与本仓库原有的 Sol-H3 像素域 refiner 实验分开。结论来自一个 held-out 巧克力拆包装片段；不是通用性能结论。

## 结论先行

这次失败不能简单归因为“Self Forcing 生成了人物”或“因果 mask 有错”。人物在训练前第一块就出现。新增消融发现：**把视频上下文从 13 缩到 5 个 latent frame，即使用 dense attention，也会出现同样的人物；保留 13 帧而只禁止 static token 读取视频，则没有出现人物。**

在同一 5 帧因果首块中，将初始噪声 σ 从 0.909375 降至 0.3，人物消失，银箔主体得到保留。证据更支持：当前 checkpoint 在**短视频前缀、固定全长文本/静音音频条件、高噪声重生成**的组合下发生内容漂移。这里尚未把视频前缀长度与音频时长匹配等因素进一步拆开。

## 上次具体如何因果化

原 H3 transformer 对打包的 `[text | audio | video]` 做 dense attention。此次只替换 transformer blocks 的 attention processor；文本 token refiner 保留原实现。

令 S 为文本及静音音频 token，V0/V1/V2 为 5/5/3 个 latent frame 的视频块。query 可以读取的 key 如下：

| Query / Key | S | V0 | V1 | V2 |
|---|---|---|---|---|
| S | 是 | 否 | 否 | 否 |
| V0 | 是 | 是 | 否 | 否 |
| V1 | 是 | 是 | 是 | 否 |
| V2 | 是 | 是 | 是 | 是 |

S 的隐藏状态仍逐层更新，并非冻结 embedding。禁止 S 读取视频，是为了避免未来块经 S 再进入过去视频块。它同时移除了原 dense 模型的多模态视觉反馈，所以并非无代价的推理开关。

每次重算一个前缀：5 → 10 → 13 帧。输入为 `[模型自己生成的干净历史 | 当前加噪块]`；历史使用 H3 时间 t=1，当前使用 t=1−σ。过去输出及历史 K/V 截断梯度，只对终止块保留计算图。没有 KV cache 加速。

每块的单步 Euler 更新为：

```text
x_sigma = (1 - sigma) * upscaled_LR + sigma * noise
x0 = x_sigma + sigma * velocity(x_sigma, t=1-sigma)
```

因此“单步”是每块一步，整段有 3 次 transformer 调用。块内仍双向，审计字段 `future_frames_exposed=0` 表示没有输入未来块，**不是逐帧因果**。原 upscaler 和 VAE 也未改成因果版本，不能称为完整流式 pipeline。

## 为什么六步 DMD 没有修复

生成器从公开的双向 refiner 权重直接加 rank-8 LoRA 并使用上述 mask；未先完成因果适配预训练或单步蒸馏初始化。训练只有两个片段、六次 G 更新，属于运行链路验证。

fake score 用 detached 的生成样本做 flow matching；G 接受 `(fake_x0-real_x0)/normalizer` 的 DMD 更新。真实运行证据包括 G/F 非零梯度、6G/10F 更新、重放误差 0、独立 checkpoint 重载误差 0。冻结参数只做了 100 个张量各 256 个值的采样校验，不是全量 hash。

teacher 和 fake score 接收的是生成结果再加噪及文本/音频，**没有独立的原始 LR 保真条件，也没有配对重建或 LR 一致性损失**。G 虽通过初始化接收到 upscaled LR，分布匹配目标仍没有显式惩罚“新增了输入中没有的人物”。这解释了漂移为何未必被 DMD 纠正；它不是训练前幻觉的原因。

σ=0.909375 时，初始化中原 upscaled latent 的系数只有 0.090625。该系数不是“剩余图像信息百分比”，也不能据此量化文本条件强度。

## 新增固定权重消融

全部使用原始冻结 refiner、相同文本/静音音频、相同随机种子与对应噪声前缀，不做更新。两种长度均将全部视频放在一个 block，因此 dense 与 static-mask 的差别只有 **S→video** 是否开放；video→video 都是全连接。13 帧的 static-mask 项并不是 5/5/3 分块流式模型。

各输出都裁成前 5 个 latent frame，再使用相同 VAE 上下文解码成 17 帧，避免解码长度差异影响比较。

| 视频前缀长度 | S 可以读视频 | σ | 已检查帧 0/8/16 中出现额外人物 |
|---|---|---|---|
| 13 | 是（dense） | 0.909375 | 否；包装纹理仍被改变 |
| 13 | 否 | 0.909375 | 否；包装纹理仍被改变 |
| 5 | 是（dense） | 0.909375 | 是 |
| 5 | 否（原 causal 首块） | 0.909375 | 是 |
| 5 | 否 | 0.3 | 否；银箔主体保留 |

13 帧 dense 精确复现旧单步基线，5 帧 static-mask 精确复现旧 causal 首块，二者 maxabs 均为 0。前 5 帧 latent 的 RMS 变化：固定 5 帧改变 static mask 为 0.132559；dense 下从 13 缩到 5 帧为 0.432208。这些是数值变化诊断，不是感知质量分数。

**本例中 static mask 不是出现人物的必要条件，也不能单独解释该现象。** 高噪声短前缀足以复现；降低 σ 的单例对照改善了内容保真，但未验证完整多块视频、细节增强或其他场景，不能宣布问题已解决。改变 σ 同时改变原输入系数、噪声幅度、模型时间和 Euler 步长，不能把该对照解释成只改变了噪声张量的幅度。

- [消融数值](../../results/h3_refiner_self_forcing_20261104/causality_ablation/summary.json)
- [消融画面对照](../../results/h3_refiner_self_forcing_20261104/causality_ablation/comparison.png)
- [消融视频](../../results/h3_refiner_self_forcing_20261104/causality_ablation/comparison.h264.mp4)

## 已排除与尚未隔离

已核对源视频 SHA、caption、文本缓存 SHA、prepared embeddings/tags，未发现错配。caption 只描述手部剥巧克力包装，原视频抽帧无人像。人物在第一块、训练前已经出现，所以六步更新与后续历史累积都不是初始原因。5/10/13 帧的共同前缀 position IDs 一致。

代码审查未发现确定的 QKV/RoPE、DMD 符号或角色切换错误；但未额外完成原生 attention processor 与替换 processor 的全模型逐值 parity，不能声称排除了所有实现问题。

尚未隔离：固定全长静音音频对短视频前缀的影响、不同 clip/seed、不同 block 长度、lower-σ 的完整 rollout、因果适配初始化、显式 LR 保真条件。下一步优先验证这些项，尤其先建立不漂移的短块学生，再扩大量级做 DMD；直接从零重训 upscaler 没有被本例证据支持。

## Upscaler 状态更正

此前“官方 latent upscaler”的称呼不准确：本实验用的是 Open-H3 发布包集成的 **LBH 社区 3D-conv upscaler**。代码与权重公开，代码 MIT、权重模型卡 Apache-2.0；当前未发现作者公开完整训练脚本/数据。MiniMax 官方 H3-Regenerate-2K 整体仍未公开，不能推断其内部结构与此社区模型相同。

- [MiniMax 官方模型卡](https://huggingface.co/MiniMaxAI/MiniMax-H3#h3-regenerate-2k)
- [Open-H3 发布包](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K)
- [LBH 代码](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) / [权重与训练数据概述](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)

Upscaler 微调尚未开始，本次没有把它写成已完成结果。
