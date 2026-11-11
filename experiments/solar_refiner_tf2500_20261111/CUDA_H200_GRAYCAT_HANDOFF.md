# Solar2500 在 CUDA H200 上更换灰猫数据的复现交接

更新日期：2026-11-11。对象是 `solar-refiner-tf2500` 的 `task_tf2500_v2` 四步任务微调，不是另一条 anchored refiner，也不是后续 Turbo DMD 两步训练。

**复现本配方应从 Ref2VA＋上游 causal-control＋causal-4step 开始，新增输出增量为零的任务 LoRA，再训练到累计2500步；不应默认加载本项目2500步任务权重继续训练。** 换灰猫后属于相同配方的新数据实验，不能沿用原样例 SHA 或声称逐位复现原模型。本文提供数据约定、缺失辅助源码、原合并证据和 CUDA 迁移边界；尚未执行 H200 训练或灰猫质量验收。

三个容易混淆的对象：

| 对象 | 本任务中的用途 |
|---|---|
| 已验收 upsampler `teacher step2000` | 冻结，把 LR latent 转成 refiner 的 draft 条件；不是 refiner 初始化权重 |
| SolarRenderer `causal-4step` | 合入 Ref2VA 的上游预训练权重，是任务微调起点的一部分 |
| 本项目 `task_tf2500_v2` | 后加 rank16 任务 LoRA 的训练结果；可做灰猫迁移的另一条对照，但不属于从头复现起点 |

## 1 数据准备依赖与精确定义

### 三个文件的职责与获取位置

| 文件 | 职责与交付 |
|---|---|
| `h3_production67_cohort_router_v1.py` | 历史数据注册、分组、split、防重叠与 SHA 校验器，不是编码器。[固定提交中的归档源码](https://github.com/RedRoman/Tencent-streaming-sr/blob/a103fa16b25fcc34df25383d3759eb274ee41625/experiments/h3_production67_limiteddev_20261108/archived_sources/h3_production67_cohort_router_v1.py) |
| `check_h3_production67_explicit_hf32off_npu_v1.py` | 旧 NPU 诊断入口；数据准备借用其 `build()` 装配 causal upsampler，不执行其诊断主程序。[归档源码](https://github.com/RedRoman/Tencent-streaming-sr/blob/a103fa16b25fcc34df25383d3759eb274ee41625/experiments/h3_production67_limiteddev_20261108/archived_sources/check_h3_production67_explicit_hf32off_npu_v1.py) |
| `data_utils.py`，含 `encode_silence` | 本次补交[完整原文件](handoff_cuda_h200_v1/data_utils.py)，字节未修改；SHA `25678af75463515f13a86b2a71aae8066cb661417e2056a3500180d370a2c488`。它支持 device 分派，但 H200 上的数值一致性仍需实测。 |

前两个位于另一实验目录，原 Solar 快照没有带入。归档文件已做路径脱敏，其内嵌 SHA 仍指原文件，且依赖 binder、loader、policy、原始数据凭证。**它们不是可直接在新灰猫目录执行的数据生成命令，不能删除断言或伪造旧 SHA 使新数据通过。** 灰猫应建立自己的清单、配对、split 与生成收据，按下面的数学和几何约定生成缓存。旧注册器的 `formal_training_allowed=false` 是旧注册阶段的状态，不是禁止本次另行授权的数据实验。

机器可读定义见 [data_preparation_contract.json](handoff_cuda_h200_v1/data_preparation_contract.json)。来源为原[数据协议](https://github.com/RedRoman/Tencent-streaming-sr/blob/a103fa16b25fcc34df25383d3759eb274ee41625/experiments/h3_production67_limiteddev_20261108/frozen_protocols/h3_production67_mixed_cohort_policy_v1.json)和 [Solar 实际输入来源](../../results/solar_refiner_tf2500_20261111/evidence/source_pairs.json)。协议中的旧 bidirectional teacher 字段不是 Solar 的实际 upsampler；后者以独立 step2000 SHA 为准。

### RGB 几何：原流程没有直接编码720p

1. 取226个真实、连续、对齐的24fps帧。先得到1920×1080的真实 HR：原 Charge 从3840×1608中心裁出1080p，Caminandes3原片即1080p。灰猫需记录自己的原始尺寸、裁剪框、帧索引和 PTS，不能把上采样图当 native HR。
2. **HR：上下各补4像素 reflect padding**，得到1920×1088；不是缩放1080→1088，也不是复制边缘像素。
3. **LR：对同一份1088×1920 HR compute RGB，逐帧 OpenCV `INTER_AREA` 缩至704×1280**，uint8输出。这是明确的非等比合成退化：高乘11/17，宽乘2/3，没有额外 blur/catblur。
4. LR/HR分别用同一 H3 VAE 编码，得到 `[24,67,44,80]` / `[24,67,68,120]`。所谓“720p”不能替代此704p定义；没有执行“先720再裁16行”的隐含步骤。
5. RGB解码输出1920×1088后，显示及对原始HR计分均裁 `y=4:1084, x=0:1920`。

```python
# hr_native: 已按清单选择的226帧，uint8 RGB，不做自动补帧或拉伸
assert hr_native.shape == (226, 1080, 1920, 3)
hr_compute = np.pad(hr_native, ((0, 0), (4, 4), (0, 0), (0, 0)), mode="reflect")
lr_compute = np.stack([
    cv2.resize(frame, (1280, 704), interpolation=cv2.INTER_AREA)
    for frame in hr_compute
])
```

`data_utils.read_video()` 是通用 cover-resize/crop/final-frame-hold 工具，**不能直接把它的默认处理当成上述配对生成协议**。先固定真实帧和上述几何，再调用 `encode_pixels`。本协议226帧经VAE内部补12帧至238，最终保留67 latent、解码回226帧；没有外部复制末帧或减速。

### VAE、精度、posterior 和归一化

- 视频编码器是 Diffusers `AutoencoderKLMiniMaxH3`，H3 base 的 `vae`，不是 Ref2VA DiT、FL2VA DiT 或 StreamRender causal decoder。配置 `clip_length=17, token_drop=3`；空间压缩16。
- 参数与输入 FP32，关闭 autocast；原NPU配对关闭 conv/matmul HF32。CUDA 的基准迁移应关闭 matmul TF32、cuDNN TF32，`torch.set_float32_matmul_precision("highest")`，明确记录 attention backend。相同设定不保证 CUDA/NPU bit-exact。
- 空间 tiling：最小256×256、重叠64×64，native attention。不要无记录切换 fast-VAE、BF16、FP16、不同 tiling 或 posterior mode。
- 视频 posterior 使用 **sample**：每次LR、HR编码各自重置CPU generator seed42，FP32 CPU `randn`再移到设备，`raw = posterior.mean.float() + posterior.std.float() * noise`。不是 `posterior.mode()`，也不是GPU generator seed42。
- RGB先除255，再按3通道 `(RGB-[.485,.456,.406])/[.229,.224,.225]` 处理。
- 视频 latent 使用 **24通道分别归一化**：`z=(raw-vae.config.latents_mean)/vae.config.latents_std`，广播形状 `[1,24,1,1,1]`。不使用全局均值方差、不额外乘 `scaling_factor`。解码前仅做一次逆变换 `raw=z*std+mean`。

完整24个 mean/std与RGB常量见 [normalization.json](handoff_cuda_h200_v1/normalization.json)。`normalization_binding_sha256` 是**整个归一化定义对象**的 canonical JSON SHA，包含helper SHA、VAE config SHA、公式及数值；不是只对mean/std数组求hash，也不是文本文件字节hash：

```python
obj = json.load(open("handoff_cuda_h200_v1/normalization.json"))
digest = hashlib.sha256(json.dumps(
    obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
).encode("utf-8")).hexdigest()
assert digest == "7295286b2fc1cdd4f57f62034b42aaf57ec79078181141b81d25a300faa22fe1"
```

VAE config SHA为 `78f67deec3d63aae807f2bfe7154bc1e26f6372cb20b63265fcbae1b62bb5745`；三份VAE权重各自的 SHA/大小已在机器可读定义中列全。原两样例的执行模式也分别保留：Charge的原determinism模式未绑定，Caminandes3为strictdet；不能声称两者从生成阶段起全部严格确定性。

### LR latent 到 refiner draft，以及静音和文本

原配对记录中的 `draft` 指低分辨率latent；Solar任务文件中的 `upscaled` 才是refiner结构条件。用冻结**causal upsampler step2000**（SHA `7b9f43c17462fb2d0554645bc7889dc7abdebc052ee9f18f6780624de912e0b5`）执行 `forward_full(lr[None], target_size=(67,68,120))`。保留该wrapper自己的normalize→model→denormalize路径，不额外归一化或用原始bidirectional upscaler/v5 step1800替换；目标高度比例17/11、宽度3/2，不能用单个1.5比例代替显式目标尺寸。

最终训练文件接口为：

```python
{
    "sample_id": "graycat_train_clip_unique_id",
    "upscaled": draft_cpu_fp32,  # [24,67,68,120]，冻结upsampler输出
    "reference": hr_cpu_fp32,    # [24,67,68,120]，真实HR编码，不是draft
}
```

两个latent都在上述normalized H3域，连续、有限。LR及raw HR RGB另存作来源与验收证据，不能只留下这两个训练张量。没有HR目标时不能按本监督配方训练；未知编码/归一化来源的灰猫latent也不能仅凭shape相同直接混入。

`encode_silence` 的独立定义：32kHz，长度 `round(frames/fps*32000)`，输入 `[2,1,samples]` 的双声道零波形；FP32 `AutoencoderKLMiniMaxH3Audio`，取posterior **mode**，按音频VAE自身32通道mean/std归一化，再保存BF16 `[2,32,T_audio]`。它不是视频24通道归一化。

但**实际 `task_tf2500_v2` 训练并未使用前一准备阶段的226帧 conditioning.pt**：训练读取固定上游 `assets/h3_silence_153_158_170.safetensors` 的 `silence_158f`，文件SHA `e3be621209aee7a7833bdc17316c15522cf3bbd222f9185c44fced9c756ae331`。应优先保留这个输入来复现。已有[完整67latent入口](source_snapshot/eval_step1000/infer_v2.py)对超出47latent的静音张量仅补未被注意力读取的尾部，并保持原双声道有效前缀；不要把补齐尾部变成新的可见条件。

文本条件沿用 [restoration prompt](../../results/solar_refiner_tf2500_20261111/evidence/prompt.json)：

> Restore this video with sharp, natural details. Preserve the original subjects, identity, colors, lighting, composition, camera motion, and timing.

Qwen3-VL第50层、28 token，prompt缓存SHA `253b1ebea968a41fee598952b162dac8a5900c3dc18c1a7cdc2fa51d60da440a`。保留embeddings和token tags；旧source_pairs中的“no_text”是上游数据准备字段，不代表Solar实际训练无文本。

## 2 NPU 包装与 CUDA 迁移

| 文件/行为 | 是否NPU专属 | CUDA/H200处理 |
|---|---|---|
| `ascend_port.fused_attention` 的 `torch_npu.npu_fusion_attention` | 是 | 不安装NPU补丁，使用固定SolarRenderer的CUDA/native attention；保留segment可见性，不换成整段双向注意力 |
| `HostChunkKVCache`、segment内CPU→device KV搬运 | 否，是为显存选择的执行策略 | 可以改为原生device KV；仍需相同C5/W6、提交/淘汰语义及前缀/分块检查。不要误当成训练方法 |
| `PagedInference` | 设备通用，但明确 **inference-only** | 不能拿来直接训练；推理是否offload按实测内存决定 |
| `runtime.TaskLoRALinear` | 不是 | 必须保留rank16、alpha16、FP32 A/B与输出相加语义；不要用全局autocast或直接merge入BF16声称等价 |
| `runtime.RecomputeBlock`、冻结层分页、CPU激活保存、resident=24 | 不是 | 可按H200内存迁移为常驻或正确的activation checkpointing。先验证梯度；不能直接删除重计算或假定完整上下文必然装得下 |
| `runtime.set_resident` 末尾 `torch.npu.empty_cache()` | 是 | 改为CUDA设备对应操作，或取消不需要的清缓存 |
| `patch_core.PatchCore`、`batch_for` | 不是，属于本配方 | 保留原生裁块、全68×120画布RoPE位置、完整首帧外观条件、对齐draft、音频空间位置；不能整份换成默认尺寸的原生core |
| trainer中的设备、HF32、RNG、同步、峰值内存与启动器 | 是/平台绑定 | 显式迁移为CUDA、TF32配置、CUDA RNG get/set、同步与内存记录；NPU三锁和NPU launcher不是CUDA入口 |

**可以使用 SolarRenderer 原生CUDA算子和因果core，但不能只运行默认CLI就说复现了本项目训练。** 原生包不自动提供本次任务LoRA、裁块RoPE、样本协议、训练日程和存取恢复。源 [runtime.py](source_snapshot/stages/task_tf2500_v2/runtime.py)、[patch_core.py](source_snapshot/stages/task_tf2500_v2/patch_core.py)、[train.py](source_snapshot/stages/task_tf2500_v2/train.py) 是迁移参照；本文没有把未执行的CUDA trainer称为可用成品。

FP32任务分支精确形式为 `base(x) + ((x.float() @ A.T) @ B.T).to(base_output.dtype)`。底座主要BF16，但 `H3_FP32_PARAMETER_NAMES` 指定的参数保持FP32。原训练没有全局autocast。冻结底座不等于切断其对输入的梯度，LoRA训练仍需穿过冻结层传播梯度。

### 是否存在未发布的改动

此次将实际冻结的 `omni_causal.py/model.py/stage0p5.py/layout.py/flow.py/geometry.py` 六文件与 `solarwm@0ca4f5e45b062cc9e9aee506de46aabdc22e4c3d` 上游原文件做SHA比对，全部一致，见[逐文件结果](handoff_cuda_h200_v1/upstream_source_comparison.json)。在这六文件上没有额外私有修改；自定义行为在已公布的wrapper中。

原环境另有一个 `diffusers-patched` H3 runtime，不能据此推断任意版本的pip Diffusers等价。旧快照遗漏了两个按SHA绑定的依赖，本次补交参考源码和许可证：

- [H3视频VAE实现](handoff_cuda_h200_v1/runtime_reference/autoencoder_kl_minimax_h3.py)，SHA `72cf83c0b8c055b67d7d26abd62769edea752c86adb4a94254a3ed7266feb5d2`。
- [Ref2VA转换器](handoff_cuda_h200_v1/runtime_reference/convert_minimax_h3_to_diffusers.py)，SHA `86f61f62934d1eccf2a76ec57b6d072e2c23767c6e49d96aa1cf24d40422b42c`。

这些是实际文件快照，不是完整Diffusers安装包；VAE依赖同包的attention/config/output模块，音频VAE等也仍来自匹配runtime。该安装目录没有可用Git提交身份，因此不编造revision、不宣称整个依赖树无修改或与pip最新版一致。H200需锁定实际依赖版本并通过加载、encode/decode和梯度检查。源码许可见[LICENSE](handoff_cuda_h200_v1/runtime_reference/LICENSE.txt)；模型权重继续遵守本发布的H3模型许可。

## 3 合并证据与 SHA 复核

已有 [causal_model.json](../../results/solar_refiner_tf2500_20261111/evidence/causal_model.json)、[base_conversion.json](../../results/solar_refiner_tf2500_20261111/evidence/base_conversion.json) 与[实际CPU合并脚本](source_snapshot/prepare_causal_model_cpu.py)。本次补交[完整合并日志](handoff_cuda_h200_v1/merge_causal.log)、[日志提取收据](handoff_cuda_h200_v1/merge_execution_receipt.json)和[原合并index字节](handoff_cuda_h200_v1/diffusion_pytorch_model.safetensors.index.json)。日志仅脱敏环境路径，原/公开SHA均保留。

| 绑定对象 | SHA256 |
|---|---|
| 原Ref2VA source index | `fb457a26ffa6294660e249b0ddd03a337f2e5393f770b5c34c8b8f90a29a7efb` |
| 转换后、合并前index | `51de67f9e29b9479af4090659150909b907b3ee5a2838268eaa0ce51d0540af9` |
| causal-control adapter | `49704c2948a111f958f3a0fc2cb9802081f9b5c129d91f30da6c8758787a8888` |
| causal-4step adapter | `10d275abf164f2115e682fd62bd1676e9b7db3c2fddaf8a66275651304d19d83` |
| 两次合并后index | `5ba637cf6e4cbab0fcfd223646cecddcaf831a46b16256109a5c92353d7e018e` |
| 原 `causal_model.json` 文件 | `f21ed5843b424b606eb816bf3332710ab0f6f724e4c88c87bb2a7f9ce852ba72` |
| 原合并日志 | `4614257a02861a7019fdd2c9ce7c290837c7436397bc669374e8f311faffdcff` |

执行顺序固定：转换Ref2VA并strict-load → 合并control → 合并4step → 保存。每个上游adapter rank384/alpha384/scale1，312个目标模块；每次FP32计算delta后写回底座dtype，再进行下一次合并。先把两份delta相加再只舍入一次不是相同运算顺序。不能把后加的rank16任务LoRA混入这一 `5ba637cf…` 身份。

日志记录两份adapter各26→312模块的全部进度与最终ready；输出14份safetensors shard。转换前后的QKV重排、FFN半区交换由精确转换器处理；不要仅凭source index一致跳过转换器差异。

**Index SHA只证明索引内容，不证明权重payload相同。** 合并前13份转换权重的完整文件SHA已在base_conversion中；原合并收据没有记录合并后14份权重的完整SHA清单。本次不伪造该清单。若需要跨机器完整权重等价，还需双方对合并后张量或shard做独立比对；不同序列化/分片可能导致文件hash不同，不能把这种差异直接当数学差异，也不能以index匹配取代payload检查。脱敏后的receipt SHA不等于原文件SHA，查[补充清单](handoff_cuda_h200_v1/MANIFEST.json)。

## 4 灰猫训练配方确认

### 起点和步数

| 目的 | 初始化和计步 |
|---|---|
| 在灰猫上复现原配方 | 已合并上游control＋4step底座冻结；新建任务LoRA `A~N(0,0.01), B=0`，任务增量为零，但底座本身不是无refiner恒等。新AdamW，灰猫更新从1计到2500 |
| 从已发布2500权重迁移灰猫 | 加载已学任务LoRA再训练，属于另一个迁移实验；明确base2500＋灰猫新增N。Release是纯推理delta，不含原AdamW/RNG，不能声称无缝恢复原optimizer |

原历史执行是：v1先完成12次完整12-latent、68×120更新；v2从精确step12的任务参数、400组AdamW和CPU/NPU RNG恢复，做step13–2500。若严格保持历史日程，灰猫新训练也保留这前12次完整更新；若从step1就用v2裁块，应标为配方变体。不能把 `v2` 理解为“从2500继续”，也不能照抄旧 `resume` 路径和数据SHA。

### 原配方参数

| 项目 | 保留值 |
|---|---|
| 更新组件 | 50层DiT的Q/K/V/out，rank16/alpha16任务LoRA，共200模块、400张量、40,140,800参数；底座、上游LoRA、upsampler、decoder冻结 |
| 训练目标 | 上游 teacher-forced flow matching，未加权velocity MSE；`v_target=x0-noise`，每块shifted-uniform sigma、shift12。不是DMD、自生成历史训练或RGB感知损失 |
| 因果结构 | C5/W6，块内双向、块间因果；训练mode=teacher、corrupt=0；推理用自身生成历史 |
| 普通更新 | 7 latent、32×64原生空间裁块；不是缩放。draft/GT同裁剪；完整68×120的窗口首帧draft作为外观条件；空间RoPE来自完整画布对应坐标 |
| 完整更新 | 每100个累计更新一次12 latent、68×120完整画布，随后保存 |
| 优化器 | AdamW，lr `1e-5`，betas `(0.9,0.99)`，eps `1e-8`，weight_decay `0`，foreach=False，grad clip `1` |
| warmup | 前50累计更新线性warmup：`lr=1e-5*min(update/50,1)`；不是从v2 step13重新预热 |
| 目标 | 累计2500次真实optimizer更新；每100保存，100/500/1000/1500/2000/2500原为首7latent训练样例短评 |
| seed与采样 | 原seed20261110，精确时间/裁块/noise日程见[common.py](source_snapshot/stages/task_tf2500_v2/common.py)。原代码硬编码两个样例取模；扩大灰猫数据时必须重建采样器和清单，不能悄悄只使用头两个文件 |
| LR类型 | 上述从HR合成的 `INTER_AREA` 704p；如称其clean，需要写明这个定义。没有已记录的catblur变体；真实H3生成LR也不是原训练分布 |
| 文本 | 沿用固定restoration prompt作为第一轮对照，保留embeddings/tags。改成猫caption是另一个条件消融 |

训练GT在teacher-forcing历史和监督目标中使用；推理 `batch_for(..., inference=True)` 把target内容置零，外观与结构条件取draft，不能把真实HR首帧或历史偷偷带入推理。原32×64训练只覆盖较短历史，这也是灰猫漂移需要另做完整序列评估的原因，不能用训练loss下降证明解决。

CUDA移植后至少先检查：400个任务张量可训练且FP32、底座梯度隔离、直接/重计算梯度一致性、关闭任务LoRA与零增量初始化一致、完整保存/读回optimizer及CUDA RNG、同条件短rollout、块边界/窗口淘汰/超过47latent后的行为。用明确的CUDA容差报告最大误差，不能复用NPU“通过”标签，也不要求未经证明的跨平台逐位一致。

## 5 验收口径与灰猫异常定位

**本任务主目标是保真复原：对真实、对齐的HR验收，同时比较简单插值和无refiner管线。** Flow模型有生成能力，但现有2500步结果没有授权把内容改写解释为成功增强。catblur和真实H3 720p属于需要单独定义的数据域，不应混进同一均值掩盖退化。

| 数据域 | 如何训练和评价 |
|---|---|
| 配对合成LR，即原配方的灰猫版本 | 从同一HR生成704p；对真实native HR报告完整226帧的L1/PSNR/SSIM/LPIPS、时序误差，分样例及前/中/后段统计 |
| catblur | 先定义退化发生在RGB还是latent、模糊核/强度/时域处理以及配对GT。若保留对齐GT，仍可作保真复原评估；这是新退化协议，不能冒称原clean复现 |
| 真实H3生成720p或未知来源latent | 先固定来源模型、VAE、归一化、几何/时间、prompt；若没有逐帧对齐HR，就不能用某个不对应的HR算有意义的PSNR/LPIPS。单列内容/身份/时间一致性与感知审片，不能仅以“更锐”验收 |

配对测试至少保留：①LR RGB Lanczos到1920×1088后裁回1080p；②同一冻结upsampler＋同一decoder、无refiner；③上游预训练四步、关闭我们的任务LoRA；④灰猫训练后四步任务模型。原task2500可作为额外迁移对照。Lanczos以明确的704p RGB为输入，记录实现、版本、颜色域和对齐，不把latent插值称为RGB Lanczos。将HR latent的decoder重建单列为codec误差参考，**真实GT列仍使用原始HR RGB**。

所有refiner臂用相同条件、几何、decoder和明确噪声来源；比较4步与Turbo两步时，不能只复用一个seed而忽略不同调用次数消耗的随机数。完整67latent/226帧评估与连续视觉审片分别记录。原发布的step1000视频GT列是decoded GT，原2500仅有训练样例首7latent短测；均不能替代新的灰猫nativeHR验收。

针对已反馈的闪烁、漂移、后段恶化：先看无refiner是否已有问题；再看关闭任务LoRA是否恢复；记录首个异常latent、每5latent块边界、首次缓存淘汰和47latent之后的位置变化。固定同一refined latent对两decoder做独立消融，才能区分refiner与decoder。没有拿到灰猫latent时，数据不足、teacher-forcing历史差异、条件/位置实现错误都只是待检假设。

先按视频/场景拆分灰猫train/dev/test，再做裁块；保留当前失败片作为holdout，避免同一猫视频相邻帧跨split。原训练只有两条素材，扩数据应增加独立场景、运动、遮挡和实际draft误差类型，而非仅重复采样同片。新灰猫实验的接受阈值须在训练前固定；原发布未完成未见数据验收，不能从训练短测临时挑一个阈值宣布通过。

## 交接方需回传的最小信息

无需先传大latent文件即可核对：所用checkpoint/adapter SHA、底座及上游LoRA组合、CUDA/PyTorch/attention后端、VAE配置与实现SHA、各张量shape/dtype/通道均值范围、归一化次数、draft来源与目标尺寸、clip帧数/起止PTS、prompt身份、四步schedule、chunk/window/cache_structure、decoder版本、首个异常帧与对应latent。之后再做一条固定样例的分支对照。

本补充没有启动训练、重算原权重、替换原视频、修改冻结快照或改动当前Turbo运行。文件来源、原/公开SHA见[MANIFEST](handoff_cuda_h200_v1/MANIFEST.json)；总报告仍见[2500步完整报告](../../results/solar_refiner_tf2500_20261111/REPORT.md)。
