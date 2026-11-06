# H3 分辨率诊断草稿只读复核（2026-11-06）

审查对象：`${SCRATCH}/diagnose_h3_v5_resolution_chain.py`，SHA256 `5d1515f44a8f0b30c34fc2f06b54f524a142715a3d8f78540b57a481fcc16ba1`。

范围：逐段阅读草稿 604 行，交叉阅读现有 `data_utils.py` 的 encode/decode、`prepare_upscaler_5s_pairs.py` 的原始 reader、`decode_results.py` 的媒体布局；没有修改草稿或冻结运行源码，没有上传、导入/运行此脚本、启动 NPU 或额外训练。本文件是静态审查记录，不是实际数据、显存、模型数值或媒体检查通过的证据。正在运行的 zero-aligned 动画任务未触碰。

## 结论和阶段边界

未发现当前分支公式、归一化、latent 尺度或固定模型身份的明显错误。可以继续作者修正及 CPU `check/prepare/self-test`；正式发布分辨率因果解释前，应完成下面两项小修正并复核。它们无需扩大四例 latent、两例 pixel 的样本范围，也不需要训练。当前尚不能声称脚本已执行成功、768 输入已改善，或低分辨率就是伪影根因。

## 最小修正清单

1. **补齐新旧 reader 的共同帧验证（prepare，行 233–277）。** 当前新 reader 严格验证原片前 120 帧、1344×768、PTS `i/24`、字节数和 source SHA；旧 pair 的 source SHA 也绑定。但 `projection_vs_cached_source384` 只记录一个误差，没有明确验证新旧提取链所指的 120 个源帧对应。建议只增加 CPU 重放原来的 `read_first_120_strict` 路径，要求重放的 384 RGB 与旧 `pair.source_pixels` 精确一致，并保存原 reader 的无重复帧证据。新原生 reader 仍检查零起始连续 PTS。**不要要求原生 RGB 后做 avg2 与旧 FFmpeg resize RGB 精确一致**：旧过滤器先 `fps=24,scale=...:flags=area,crop` 再输出 RGB，颜色变换/缩放和舍入顺序可能不同。现有 projected/raw 差值应继续保留，用来量化这项处理差异。此项是共同时间参考的审计补强，不是已发现存在时间错位。

2. **完整验证可恢复产物的上游链（行 280–328、369–428、459–461、532–555）。** 当前新 latent 的 `done.json` 验证 contract、sample、文件 SHA、冻结状态和精度，但读取 prediction 时没有核对所记录 `checkpoint_sha256/model/input_origin` 是否等于本合同及当前输入 receipt；`latent_summary` 也遗漏了两份 `information_matched384` 编码 receipt。新鲜运行的写入路径是正确的，此问题主要影响分阶段恢复和最终独立复核。最小修正：核对 teacher/motion 对应权重 SHA、native 分支的原 pair record、information-matched 分支的编码 receipt record；汇总收齐 6 编码 + 10 预测 receipt；pixels/finalize 验证这些 receipt 的当前 SHA 与 latent summary 相同。`check_prepared` 要求 reader key 恰好是四个 IDS；finalize 要求两类媒体及必需文件存在，实际统计为四个完整解码的 MP4，再写最终数量。无需重算已有指标或另加因果测试。

这两项完成后可冻结新的脚本 SHA，使用新合同目录；不要在已经生成该草稿 SHA 合同的目录中静默替换源码。

## 已核对的设计

- **样本和选模**：行 44–48 固定四个已查看的问题 dev，像素只奔马 `0421364` 与弹琴 `0665341`。行 108–121 绑定原 dev manifest/plan/source/pair；行 131–132 固定既有 motion step 2000。没有训练、优化器、test 数据参与或重新选 checkpoint 的逻辑。这是事后诊断，不是盲测。
- **分辨率分支**：旧 `draft=z192` 与 `reference=z384` 原样复用；新原生768编码得到 `z768` 作为同尺寸目标。teacher/motion 的 384→768 输入均是旧 z384；另两例用 `S384 -> avg2 S192 -> bicubic S384 -> encode -> teacher ->768`，保留原 S192 的信息来源，控制相同 upscaler 输入网格。没有把 z192 直接插值冒充真实 z384（行 193–214、332–356、369–400）。
- **归一化和采样**：新编码调用冻结 `encode_pixels`，ImageNet 像素归一化，posterior sample，CPU seed 42，然后 `(raw-mean)/std`；decode 采用反归一化与原 decoder。teacher 使用原 `wrapper_forward`，motion 使用既有 wrapper，无新 weight surgery。37 latent 解码至原协议的 124 帧再保留前120；没有误用简单 4× 时间下标（冻结 helper `data_utils.py:149–161,187–202`）。
- **latents 指标**：目标384和目标768分别聚合；首36/37有效 latent 的 MSE、MAE、时间差分 MSE 各自只在同网格内比较（行 359–365、403–428）。不会把不同尺寸的 latent MSE 降低直接当作画质改善，也没有缩放旧 flow 计算高分辨率 eWarp。
- **像素公共域**：每条路径都以同一旧 pair 的原始 `source_pixels` 672×384、uint8/255 为参考；所有768输出先确定性 avg2 回公共384。全120帧与119真实邻接对、原 frame regions、固定 GT motion masks 不随模型变化。原生768辅助指标/裁片另列；不拿 MP4 作为数值输入（行 217–230、471–515）。
- **节点可见性**：有普通 LR→384 bicubic、LR VAE roundtrip、已有384 VAE roundtrip、新768 roundtrip、两尺度 teacher/motion，以及同信息高网格 teacher。实际 native 裁片采用原像素，不放大缩小（行 439–447）。两例各两个有限列宽的5秒视频，共四个；information-control 还有静态图。
- **冻结和精度**：当前代码绑定脚本、历史合同、模型 checkpoint、原始 upscaler 权重/实现、VAE 权重/实现、数据、LPIPS、旧缓存；比较历史 execution precision，不修改 HF32 flags。模型 FP32/eval/no-grad，参数与 buffer 版本在每个阶段前后核对；末尾重算外部文件合同（行 66–79、86–181、332–356、369–400、464–526）。这是静态确认存在这些检查，尚无本次执行通过证据。

## 解释必须保留的限制

1. 已观察到 LR VAE roundtrip 形变，只能定位到 **编码与解码组成的链**，不能仅凭该结果认定 encoder 单独有问题。768 roundtrip 可进一步检验工作尺寸，但不单独分离 encoder 和 decoder 的贡献。
2. 192→384 与真实384→768同时改变输入细节、网格与 VAE 工作尺寸；同信息 bicubic 分支有助于区分来源，但插值、posterior 网格、感受野/tiling 仍改变，不能称为完美单因素消融。两个同384网格的分支使用同 seed 和噪声形状；不同分辨率共享 seed 不代表共享同一噪声场。
3. 原 pair 的 z192/z384 为历史编码，草稿只保证其文件身份和已知 recipe；没有新执行一次旧384编码来做数值 parity。若当前依赖身份未能从原准备证据完整追溯，或结果只出现微小差异，可加**单例原384重新编码 anchor**，并如实比较误差；这不是本轮必需的六编码规模扩展，也不应先以它阻塞最小诊断。
4. 384输入仍没有达到上游显式576/768输入档位，且原片仅有真实1344×768。不能生成假1152 GT，不能由该实验外推“已达到官方输入域”。
5. 原 VAE 非因果，整个结果是固定120帧离线诊断；不证明端到端流式质量。所有四例已经看过，不适合泛化率、显著性或选模主张。

## 计算量与下一步

草稿现有计划：CPU四例 prepare；latent 阶段 4次原生768编码 + 2次同信息384编码、4次 teacher + 4次 motion + 2次同信息 teacher forward；pixel 阶段仅两例，每例7次新 decode，另复用1份原384 lossless RGB reference，总14次新 decode、4个5秒比较视频。此前 LR probe 已执行，但若它未保存 lossless RGB 张量，不可用其有损MP4替代本草稿的数值 decode。先完成 CPU 合同/reader 小修与检查，再按阶段运行；是否推进像素应由 latent 和现有 LR probe 的实际证据决定。没有理由重新训练或重跑旧全量评估。
