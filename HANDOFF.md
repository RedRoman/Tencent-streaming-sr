# Sol-H3 流式视频超分 — 接手说明

**交接日期**：2026-11-04
**机器**：ModelArts 30778（昇腾 910B3 ×8），代码在 `~/modelarts/user-job-dir/sol_h3/`
**当前状态**：v2 已交付并验证，latent 超分路线已封（证据完整）

> 先读这一篇，再按需翻 `REPORT.md`（完整技术报告）。
> **如果你只有 10 分钟**：读第 1、2、7 节。

---

## 1. 这个项目在做什么

H3 是一个 33.1B 的视频生成模型。直接让它生成高清视频太慢，跟不上播放。

做法是分工：**H3 只在低分辨率（448×256）生成，一个 1024 万参数的小网络把每帧放大 3 倍到
1344×768**。低清像素量只有高清的 1/9，所以贵的部分都便宜一个数量级。

v2 在此基础上加了一条：**H3 每隔 30 帧额外产出一个 HR latent，refiner 直接吃这个 latent
（不解码）当细节参考**。

```
                    ┌──────────────────────────────────┐
   prompt ─────────→│  H3  33.1B  (冻结, 多卡)          │
                    └────────┬─────────────┬───────────┘
                             │             │
               每帧 LR latent │             │ 每30帧 1个 HR latent
               [24,T,16,28]   ↓             │ [24,48,84]
                    ┌─────────────────┐     │
          NPU-A     │ H3 VAE 解码     │     │ 不解码
                    │ 427 ms/块       │     │
                    └────────┬────────┘     │
                             │ 448×256      │
                             ↓              ↓
                    ┌──────────────────────────────────┐
          NPU-B     │ Refiner PixelSR 10.24M           │
                    │ 302 ms/块                        │
                    └────────────────┬─────────────────┘
                                     ↓
                               1344×768 @ 24fps
```

三个阶段分卡并行，稳态块周期 = max(各阶段) 而非相加。

---

## 2. 当前成绩

**画质**（6 clip，全分辨率，统一口径）：

| 版本 | PSNR | LPIPS↓ | 比 bicubic 好 | flicker |
|---|---|---|---|---|
| bicubic | 26.005 | 0.2852 | — | 0.0123 |
| v1 | 26.577 | 0.2052 | 28.1% | 0.0122 |
| **v2（交付版，kf=30）** | **27.320** | **0.1714** | **39.9%** | 0.0114 |
| v2 kf=8 | 27.705 | 0.1576 | 44.7% | 0.0113 |
| （GT flicker） | | | | 0.0123 |

**流式**（双卡，真实播放时钟）：

| | |
|---|---|
| 块周期 | 710 ms（预算 708 ms） |
| 漂移 | −0.01 s |
| 欠载 | **0** |
| 首帧 TTFF | 1.30 s |

---

## 3. 环境（第一件事）

```bash
cd ~/modelarts/user-job-dir/sol_h3
source ./solh3_env.sh      # 必须，每个新 shell 都要
```

**为什么必须**：系统 CANN 缺 910B 算子包，不 source 会在所有算子上报 `561103`。
`solh3_env.sh` 指向私有 CANN，同时**保留系统 CANN 的 python 路径给 `tbe`**——
两边都要，少一个就跑不起来。

**两个会误导人的现象**：

1. `free(): invalid pointer` / SIGABRT 出现在脚本末尾 —— 这是 CPython 退出噪声，
   **不是失败**。本项目所有脚本结尾用 `os._exit(0)` 规避。
   （前人就是据此误判放弃了整条路线。）
2. `[ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!`
   同样是退出噪声，忽略。

---

## 4. 跑起来

```bash
source ./solh3_env.sh

# 训练 v2 配方（单变量对照就用这个，改一个参数）
$SOLH3_PY pixel_train.py --window 1 --width 160 --blocks 20 --lpips 1.0 \
  --ref-max 30 --ref-latent 1 --clips-per-step 48 \
  --steps 40000 --eval-every 2000 --out results/<name>

# 全分辨率评测（这是唯一算数的口径，见第 6 节）
$SOLH3_PY pixel_eval.py --ckpt results/<name>/best.pt --clips 6 --save-video 2 \
  --kf-interval 30 --out results/ev_<name>

# 端到端流式（双卡）
$SOLH3_PY stream_kf.py --ckpt results/<name>/best.pt --kf-interval 30 \
  --dec-dev npu:0 --sr-dev npu:1 --latents 35 --out results/stream_<name>

# 对比视频
$SOLH3_PY mkfull.py --new results/ev_<new> --old results/ev_<old> \
  --clip <clip> --out results/demo_full --crop 224 --zoom 2

# 流式过程可视化（时间轴 = 真实墙上时钟）
$SOLH3_PY mkstream_viz.py --run results/stream_<name> --out results/demo_stream
```

单卡指定用 `ASCEND_RT_VISIBLE_DEVICES=<n>`，脚本内部固定用 `npu:0`。

---

## 5. 数据与权重

| 位置 | 内容 |
|---|---|
| `/temp/yunfeng/h3_hrx0kd_transfer_20261008/` | 原始 800 对 LR/HR latent + text embedding |
| `data/pixels/{lr,hr}/*.npy` | 解码后的像素对（338 GB，mmap 读） |
| `data/hrlat/*.npy` | HR latent 本地 fp16 镜像（5.4 GB，**必须用这个**，见下） |
| `ckpts/h3-base/vae` | H3 VAE，**必须 fp32 加载**（有 fp32 保留模块） |
| `ckpts/h3-base/transformer` | H3 transformer 33.12B bf16，61.7 GiB，**单卡装不下** |

**`data/hrlat` 为什么必须**：直接从 `/temp` 读 `.pt.part` 会让训练从 0.22 s/step
掉到 0.61 s/step —— 每步 48 个 clip 就是 48 次 NFS 往返。本地 fp16 mmap 后回到 0.03 s/step。
重建用 `make_hrlat_cache.py <shard> <nshard>`（8 路并行约 1 分钟）。

**权重**（见本仓 Release）：

| 文件 | 说明 |
|---|---|
| `v2_refiner_lat30.pt` | **交付版** PixelSR 10.24M |
| `hrdecoder_s768.pt` | 蒸馏解码器 32.4M，266 ms/块，22× 加速 |
| `hrdecoder_s384.pt` | 蒸馏解码器 6.6M，83 ms/块，71× 加速 |

远端原址：`results/latref_2200/lat30/`、`results/hrdec_1200/{s768,s384}/`

---

## 6. 评测口径（这一节最容易踩坑，请务必读）

这个项目在口径上翻过好几次车，排序被翻转过。规则：

**只认全分辨率评测**（`pixel_eval.py`，固定 6 clip，固定 30 帧关键帧调度）。
训练日志里的 `val_lpips` / `d0..d30` 是 **192×192 crop** 上测的，只能用于看训练趋势，
**不能用来和别的 arm 比，更不能和交付基线比**。

实测过的翻转案例：

| 改动 | crop 口径 | 全帧口径 |
|---|---|---|
| 逐区域门控 | −0.0009（更好） | **+0.0030（更差，6/6 clip 全差）** |

原因：crop 是 64×64 的 LR 窗口，很多空间性质在这个尺度上被抹平。

**其他三条硬规则**：

1. **比 arm 之前先量噪声底。** 随机评测的噪声曾达 0.44 dB 标准差，淹没 1 dB 的真实差距。
   现在评测探针是固定 seed 的，重复评测 sd = 0.000，但你**新加的评测要自己验这一点**。
2. **测时延必须预热。** NPU 首次调用含算子编译，可达稳态的百倍。本项目栽过两次：
   关键帧编码器 1338 ms → 实为 4 ms；解码器评测 786 ms → 实为 83 ms。
   两处数字差一个数量级时，**先怀疑测量，不要先怀疑模型**。
3. **PSNR 和 LPIPS 会给出相反的排序。** 曾有 arm PSNR 最高（26.905）但感知比 bicubic 还差
   （LPIPS 0.3008 vs 0.2852）。交付指标是 LPIPS；`best.pt` 的选择器也必须用 LPIPS。

---

## 7. 已经封掉的路线（**别重做**）

每条都有完整数据，详见 `REPORT.md` 第 5、7 节。

| 路线 | 结果 | 机理 |
|---|---|---|
| **容量放大** 9.3M→28.4M | 全部略差 | 缺的高频是被 3× 降采样物理销毁的，需要*生成*，加宽卷积学不会 |
| **文本条件**（H3 prompt_embeds → FiLM） | 和不用打平 | 池化 caption 只提供低分辨率先验，像素本来就有 |
| **时序判别器** | LPIPS 随权重单调变差 | 问题不是"纹理帧间不一致"，是**纹理本身就不对** |
| **VAE attention sparsity** | 上限 1.75×，需求 8.3× | VAE 解码器注意力**已经是窗口化的**（见下） |
| **latent 域确定性回归** | 结构 28.5 vs 像素 34.06 | manifold deviation |
| **latent 域 flow matching** | 收敛到恒等映射 | 第二级没有第一级不知道的信息，最优解是"什么都不做" |
| **latent 关键帧加密** | 密度 ×3 只买 0.95 dB | 需要的是 4.9 dB |
| **逐区域门控** | 全帧 6/6 变差 | 门控信号本身不可靠（见下） |

### 7.1 两个值得单独记住的发现

**H3 VAE 解码器是 36 层 2.42B 的 ViT，不是 CNN。而且它的注意力已经窗口化了**：

| | 注意力调用次数 | 每次输入形状 |
|---|---|---|
| LR 448×256 | 72 | (1, 1797, 2048) |
| HR 1344×768 | **1008** | **完全相同** |

HR 贵 14 倍纯粹因为 tile 多了 14 倍。改大 tile **反而更慢**（768 tile → 7674 ms vs
默认 5885 ms）。时间分布 attn 42.8% / FFN 47.2%，所以**注意力优化上限 1.75×**。

**latent 超分失败的最终原因是采样率，不是目标函数**：

| 起点（未过任何网络） | 结构 | 细节 |
|---|---|---|
| LR latent 三线性 3× 再解码 | **17.10** | 18.85 |
| LR 像素 bicubic 3× | **33.18** | 27.19 |

差 16 dB。网格算术解释了一切：

```
LR latent 一个 cell = 16 LR 像素 = 48 HR 像素   ← 每 48 px 才有一个位置锚点
LR 像素一个点       =              3 HR 像素   ← 每 3 px 就有一个
```

物体移动 20 个 HR 像素，在像素域是采样点挪位、插值自然跟上；在 latent 域**完全发生在
一个 cell 内部**，位移信息只存在于 24 个通道的数值里。空间插值等于把两个不同内容的
描述符加权平均，解码出来就是畸变。

这也解释了为什么**静止区域 latent 反而更好**（cell 内容不变，插值无损，且继承关键帧真高频）：

| 区域运动量 | latent vs 当前帧 | 像素 vs 当前帧 | 差 |
|---|---|---|---|
| 最静 | **56.88** | 55.60 | **−1.28（latent 赢）** |
| 最动 | 27.50 | 32.27 | +4.77 |

---

## 8. 还没做完的事（按优先级）

### 8.1 接上 H3 侧的关键帧产出 ← **最优先**

**这是 v2 变成真正可交付系统的唯一缺口。**

现在流式验证里的 HR latent 是**从磁盘读的**，不是 H3 现场产的。成本已经实测：

| | 50 层耗时 |
|---|---|
| LR 5 帧（正常一步） | 89.2 ms |
| LR 5 帧 + HR 关键帧 | 201.8 ms |
| **增量** | **112.6 ms** → 摊到 30 帧 = 3.75 ms/帧 |

（这是在**实际部署的 VDN 混合注意力**上测的，不是 base 稠密注意力。注意：这是**下界**，
用单个 attention 模块 ×50 层推算，忽略了不随 token 增长的每步开销。）

要做的事：在 H3 serving 里加一个"每 N 帧多产一个 HR latent"的开关。

### 8.2 冷启动

流的开头没有关键帧，模型在无参考时退化到 LPIPS 0.235（比 v1 的 0.187 还差）。
两个选项：接受开头 1.25 s 画质低，或开头切到无参考分支 / v1 权重过渡。

### 8.3 关键帧加密到 8 帧

**目前唯一已验证能继续提升画质的旋钮**：LPIPS 0.1714 → 0.1576（好 8%），
代价是 H3 侧 3.75 → 14.1 ms/帧。权重已训好（`results/latref_2200/lat8/`）。

### 8.4 再往上走：接入 H3 内部状态

第 7 节八条失败路线集中在三个方向：latent 域、目标函数、像素域启发式。都走不通了。

剩下的方向是**接入 H3 的内部状态**（hidden states / 去噪轨迹 / 运动信息）。
依据是第 7.1 节的结论：像素方案的剩余缺口核心是**不知道哪里该信关键帧**，
而这个问题 H3 内部是有答案的——它生成时就知道物体怎么动，但运动信息全部丢在 H3 里面。

可测的第一步：拿 H3 的注意力图或 hidden states 替换 `|LR_t − LR_kf|` 作为门控输入，
和第 7 节测出的 oracle 上限 3.5% 比较。

---

## 9. 差异化的边界（汇报时请准确表述）

| 版本 | 与普通流式 SR 的区别 |
|---|---|
| v1 | **没有区别**。接口只有像素，机理上就是普通流式 SR 挂在 H3 后面 |
| **v2** | 开始用上外部 SR 拿不到的信息：**H3 自产的 HR latent**（不解码，外部 SR 接不住） |
| 尚未 | H3 的内部状态（hidden / 轨迹 / 运动）**完全没用上** |

v2 真正成立的那条性质是：**refiner 可以无记忆**。W=1 与 W=5 实测完全打平
（+1.096 vs +1.072 dB），输出 flicker 0.0114 还低于 GT 的 0.0123——
时间一致性是从 H3 的流里**继承**的，不是 refiner 算出来的。

---

## 10. 代码索引

### 主线（v2）

| 文件 | 作用 |
|---|---|
| `pixel_model.py` | PixelSR 主干 + KeyframeRefLatent（含 local_gate，已证无效，默认关） |
| `pixel_train.py` | 训练，含按关键帧距离分桶的评测 |
| `pixel_eval.py` | 全分辨率评测 + 三联视频 ← **唯一算数的口径** |
| `stream_kf.py` | 三阶段流式驱动（含墙上时钟时间戳） |
| `stream_2card.py` | 双卡流水线 |
| `make_hrlat_cache.py` | HR latent 本地 fp16 镜像（训练前必须先跑） |
| `solh3_env.sh` | **环境，每个 shell 都要 source** |

### 解码器蒸馏（独立可复用成果）

| 文件 | 作用 |
|---|---|
| `hrdec_model.py` / `hrdec_train.py` | 学生网络 + 蒸馏训练 |
| `hrdec_align.py` | **帧契约验证，别跳过** |
| `hrdec_bench.py` / `hrdec_eval.py` | 容量-速度扫描 / 全帧评测 |

帧契约（实测，不是假设）：`latents[5k:5k+5] → frames[17k:17k+17]`，偏移全部命中，39–42 dB。
17 = 5×4 − 3，正是 VAE 自己的账（`token_drop=3`）。**约 40 dB 是任何分块学生的天花板。**

### 已封路线的代码（保留作证据，不要在上面继续）

`latsr_model.py` `latsr_train.py` `latsr_eval.py` `latflow_model.py` `latflow_train.py`
`h2w_*.py` `taeh3_*.py`

### 诊断工具（复用价值高）

| 文件 | 回答什么问题 |
|---|---|
| `latsr_gap.py` | 误差在结构还是细节（频带分解） |
| `motion_err.py` | 误差随区域运动量怎么变 |
| `pos_info.py` | 两条路线的**起点**各有多少位置信息 |
| `latent_motion.py` | latent 空间能否表达运动（直线插值测试） |
| `blend_probe.py` | 门控混合的 oracle 上限 |
| `vae_profile.py` / `vae_block_prof.py` | VAE 解码器时间都花在哪 |
| `where_err.py` | 误差的空间分布 |
| `hf_arms.py` | 各 arm 的高频能量 vs GT |

### 可视化

| 文件 | 产出 |
|---|---|
| `mkfull.py` | 全帧对比 + inset 放大镜（主力） |
| `mkstream_viz.py` | **流式过程**（时间轴 = 真实墙上时钟） |
| `mkstream_vs.py` | 与「直接高清」的同轴对照 |
| `mk4way.py` | 四栏（bicubic / 像素 / latent SR / GT） |

---

## 11. 关键常数（抄错会导致结论错）

```python
FRAMES_PER_LATENT = 3.4     # 不是 4！token_drop=3 对每个 clip 计费
                            # 按 4 算会把块预算从 708ms 误报成 833ms，
                            # 并制造一个虚假的 -0.125 s/块漂移
N_LAT, N_FRAMES   = 5, 17   # VAE 最小解码单位：5 latent → 17 帧，解不出更少
LAT_PER_LRPX      = 3 / 16  # 1 HR latent cell = 16 HR px = 16/3 LR px
块预算             = 708 ms  # 17 / 24fps
HR VAE 解码        = 5847 ms # 5-latent 最小块，是块预算的 8.3 倍
LR VAE 解码        = 427 ms
```

训练裁剪位置必须落在 16 的倍数上，HR latent 的裁剪才是精确整数格。

---

## 12. 本仓目录

```
├── HANDOFF.md          ← 本文件
├── REPORT.md           ← 完整技术报告（558 行，12 节）
├── code/               ← 全部源码（127 个 .py + 启动脚本）
├── logs/               ← 训练与诊断日志
└── results/            ← 实验元数据（config / metrics / summary，196 个 run）
```

二进制产物在本仓 **Releases**：权重 3 个 + 对比视频。

远端完整产物（含 338 GB 数据与 112 GB 中间结果）仍在
`30778:~/modelarts/user-job-dir/sol_h3/`，**机器是共享的，开跑前先查卡**。
