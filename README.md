# Sol-H3 — 流式视频超分

2026-11-04 新增独立研究快照：[Open-H3 refiner 的 Self Forcing + DMD 单步实验](experiments/h3_refiner_self_forcing_20261104/README.md)、[因果化与人物幻觉消融分析](experiments/h3_refiner_self_forcing_20261104/ANALYSIS.md)、[结果与视频](results/h3_refiner_self_forcing_20261104/README.md)。该实验使用 H3 自身作 refiner，训练链路已跑通但画质未达标；与下方原 Sol-H3 像素域 refiner 结果分开记录。

低分辨率流式 H3 + 高清 refiner。H3 在 448×256 生成，一个 10.24M 的像素域 refiner
放大 3 倍到 1344×768；H3 每隔 30 帧额外产出一个 HR latent 作为细节参考，
refiner 直接消费该 latent（不解码）。

| | PSNR | LPIPS↓ | 比 bicubic 好 | flicker |
|---|---|---|---|---|
| bicubic | 26.005 | 0.2852 | — | 0.0123 |
| v1 | 26.577 | 0.2052 | 28.1% | 0.0122 |
| **v2（交付版）** | **27.320** | **0.1714** | **39.9%** | 0.0114 |

端到端流式：块周期 710 ms（预算 708 ms），漂移 −0.01 s，**零欠载**，首帧 1.30 s。

## 从哪开始

- **接手的人先读 [`HANDOFF.md`](HANDOFF.md)** —— 环境、跑法、踩过的坑、没做完的事
- 完整数据与推理过程见 [`REPORT.md`](REPORT.md)
- 权重与对比视频在 [Releases](../../releases)

## 目录

```
code/      全部源码（127 个 .py + 启动脚本）
logs/      训练与诊断日志
results/   实验元数据（config / metrics / summary，196 个 run）
```

## 注意

代码跑在 ModelArts 30778（昇腾 910B3），**每个 shell 必须先 `source solh3_env.sh`**，
否则所有算子报 CANN 561103。脚本末尾的 `free(): invalid pointer` / SIGABRT
是 CPython 退出噪声，不是失败。详见 HANDOFF 第 3 节。
