# H3 refiner：块因果与逐 latent 帧因果探针

这里将“单向 refiner”具体化为**时间因果注意力**：输出只能读取当前及过去的视频 latent。实验使用已发布 Open-H3 refiner 权重，比较每块 5 个 latent 帧与每块 1 个 latent 帧的推理；**本次没有 optimizer、梯度更新或新的 DMD 训练**。首轮未来内容扰动检查通过，但改变执行前缀长度后输出不一致，因此完整 probe 仍判定失败。随后固定长度、未来槽补零的三个对照检查均通过（最大绝对差为 0）；原始变长前缀失败记录保留。详见[结果与视频](../../results/h3_causal_refiner_20261104/README.md)。

## 两种因果粒度

| 方式 | 可读取的视频范围 | 13 帧整段 rollout |
| --- | --- | --- |
| `block5` | 当前块内部双向，块间仅访问过去 | `5 + 5 + 3`，共 3 次单步 transformer 前向 |
| `frame1` | 当前 latent 帧内部空间双向，只访问过去 latent 帧 | 13 个单帧块，共 13 次单步 transformer 前向 |

“单步”指**每块 1 次去噪**，不代表整段只调用模型一次。`frame1` 是 latent 时间维逐帧因果；一个 H3 latent 帧并不等于一个像素视频帧，因此不能称为逐像素帧因果。模型位置编码保留真实时间位置，已生成的 clean 历史由自身输出提供，当前块以 noisy upscaled latent 初始化，每个前缀重新计算，无 KV cache。

注意力实现复用相邻 [Self Forcing 实验](../h3_refiner_self_forcing_20261104/README.md) 的 `temporal.py`：视频 query 可读静态 text/audio 和不晚于自身块的视频；静态 text/audio query 仅能读取静态条件，避免未来视频先写入条件 token 再间接泄漏。H3 时间约定为 `t=1-sigma`，clean 历史的时间为 `1`。

## 本次实际配置与检查

默认 `real` 角色使用 H3 base 合并发布 refiner LoRA，并保留已有 runtime 的发布 Sharp 分支；绕过先前训练的 generator/fake 适配器。CLI 可显式给出 `--generator-checkpoint` 做其他诊断，但这不是默认冻结权重试验。仅使用先前留出片段 `VIDGEN5S_0380588`，固定 upscaled 输入 `[24,13,24,42]`、文本以及全长 43 帧静音编码 `[2,32,72]`。

- `block5`：在 sigma `0.909375` 时替换第 5 个 latent 帧之后的输入，比较前 5 帧输出；dense 分支作为应响应未来扰动的负对照。
- 比较相同输入前缀的 `causal5` 与 `causal13` 前 5 帧，核对不同前缀中的位置值和第一块 rollout 重放。
- 完整 `block5` rollout 分别运行 sigma `0.909375` 和 `0.3`。
- `frame1`：在 sigma `0.3` 时改变第 1 帧之后的输入，检查第 1 帧输出；随后完成 13 帧 rollout 并重放校验首帧。
- 采用固定 seed `20261104`、noise seed `20262095`，数值判断 tolerance `1e-4`。若旧输出存在，同时检查同设置输出是否匹配。

完整入口共需 26 次 transformer 前向。解码入口比较 upscaled、`block5 sigma=0.3`、`frame1 sigma=0.3` 与 reference，输出 672×384、43 帧、24 fps 的各分支及对照视频。reference 仅用于比较，不作为模型条件。

## 已发现的差异与固定长度对照

首轮 probe 中，`block5` 与 `frame1` 的未来内容扰动均没有改变被检查的过去输出，最大绝对差为 `0`。但是长度 5 的前缀与长度 13 的完整执行在共同前 5 帧上的最大绝对差约为 `0.164967`；逐帧 rollout 的首帧与长度 13 的对应输出也有约 `0.086939` 的差异。两者均超过 `1e-4` tolerance。原始 `summary.json` 保留 `passed=false`，脚本在写完结果后以断言失败退出；**按当前设置复现原命令，非零退出是已记录的检查失败，不能通过删除检查或放宽阈值视为成功。**

当前证据区分了“改变未来输入内容”和“改变执行张量长度”这两件事。采样中首次观察到差异的位置为第一层注意力输出、输出投影之前（`transformer_blocks.0.attn.to_out.0.input`），最大绝对差 `0.001953125`；此前采样的输入、Q/K/V 与归一化输出一致。具体根因尚未确定，不能据此断言是某个 NPU 算子的数值问题。`refiner_trace.py` 仅采样并保存共享 token 行在各模块的真实数值，帮助定位首次观察到的分歧；采样行相同不代表所有隐藏值都相同。

新增 `FixedHorizonRefiner` 在每次调用时都创建 **13 个 latent 帧**的张量，填入实际当前前缀，其余位置保持零；注意力掩码禁止早期视频 query 和静态条件 query 读取未来槽。真实未来内容从未填入这些槽。`probe_refiner_fixed_shape.py` 比较完整输入与零未来槽执行的共同输出，并完成 `block5` / `frame1` 的 sigma `0.3` rollout。此对照控制张量长度，仍属于离线固定窗口原型，没有 KV cache，也没有证明无限长度流式执行。完整输入与零未来槽执行的首块检查（block5、frame1）以及 frame1 rollout 首帧重放均通过，最大绝对差为 0。固定窗口 sigma `0.3` rollout 的 block5 / frame1 分别耗时约 `10.247` / `46.090` 秒，不含 VAE 解码；这不是端到端速度或 2K 基准。原始变长前缀失败记录未被覆盖。

已解码的变长前缀与固定窗口 sigma `0.3` 输出，在已检查的首、中、末帧中保留了手部与锡纸，没有发现此前多出来的人物。这是单片段抽帧观察，不能代替完整视频审阅或消除上述数值一致性失败。

所有结论都限定为**固定 upscaled 输入及固定完整 text/audio 条件下，refiner 对视频时间的条件因果性**。上游 VAE、latent upscaler 仍未因果化；没有扰动未来音频，也没有测试任意输入与所有块边界。因此本探针不证明端到端流式因果。降低 sigma 同时改变输入保真比例、噪声幅度、模型时间和 Euler 步长，不能将效果归结于一个因素。是否保留内容、是否产生额外人物仍需完整视频检查。

## 复现入口

依赖、固定权重版本和 Ascend runtime 沿用相邻实验。此目录只包含新增 probe 与入口，复用 `model_runtime.py`、`temporal.py`、`locked_run.py` 和解码 helper。模型权重、原数据、latent 输出和基础设施路径均不入库。

在本目录执行，先配置外部环境：

```bash
export EXP_ROOT=/path/to/h3-work
export H3_BASE_MODEL=/path/to/h3-base
export H3_UPSTREAM_DIR=/path/to/Open-H3-Regenerate-2K
export H3_DIFFUSERS_SRC=/path/to/diffusers-patched/src
export H3_PYTHON_BIN=/path/to/npu-venv/bin
export H3_CANN_ENV=/path/to/Ascend/cann/set_env.sh
export H3_FFMPEG_BIN=/path/to/ffmpeg/bin
export NPU_LOCK_DIR=/path/to/shared-device-locks

bash run_refiner_causality_probe.sh <idle-physical-npu-id-a,id-b>
bash run_decode_refiner_causality.sh <idle-physical-npu-id>
# 独立的固定13帧执行对照；原始 probe 的失败记录仍保留。
bash run_refiner_fixed_shape.sh <idle-physical-npu-id-a,id-b>
bash run_decode_refiner_fixed_shape.sh <idle-physical-npu-id>
```

`H3_UPSTREAM_DIR` 包含 `open_h3/`；外部 `EXP_ROOT` 需要已经准备好的 `merged_transformer/`、`assets/` 和 `data/holdout.jsonl`。原始 probe 默认结果写入新的 `EXP_ROOT/refiner_causality_probe/`；目录已存在时拒绝覆盖。其 `--output` 可指定另一个新目录。固定长度对照单独写入 `EXP_ROOT/refiner_fixed_shape/`，同样拒绝覆盖。两组命令各自运行，原始 probe 的非零退出后仍可单独检查已保存的报告并执行对照。入口自动设置 `EXP_CODE_DIR`，可用 `H3_REFINER_CODE_DIR` 覆盖相邻 helper 路径。设备运行前仍需确认空闲，入口还会持有共享锁并检查 `npu-smi`。
