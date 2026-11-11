# SolarRenderer refiner — cumulative step2500 candidate

本目录交付固定2500步任务LoRA的推理加载器与训练源码快照。训练已正常结束，权重导出通过CPU逐张量回读；**完整最终视频、未见样例nativeHR指标与实时部署尚未验收**。

[完整中文报告](../../results/solar_refiner_tf2500_20261111/REPORT.md) · [Release权重](https://github.com/RedRoman/Tencent-streaming-sr/releases/tag/solar-refiner-tf2500-20261111) · [逐张量导出审计](../../results/solar_refiner_tf2500_20261111/evidence/publication_tf2500_v1/export/adapter_export_audit.json) · [step1000视频](../../results/solar_refiner_tf2500_20261111/REPORT.md#7-视频)

[CUDA/H200 灰猫复现交接](CUDA_H200_GRAYCAT_HANDOFF.md)：补充数据准备源码与归一化定义、NPU/CUDA迁移边界、上游权重合并日志及逐SHA复核、初始化/训练配方和验收口径。此文档与依赖补交不表示已完成H200训练或灰猫质量验收。

## 下载和检查

Release资产为 `solar_refiner_tf2500_lora_rank16_fp32.safetensors`：160,606,248字节、400个FP32张量、40,140,800个参数。SHA-256：

```text
9c5cd061f3f339c5939e9568fb9d4179e27aaec7d584d9ce68be2073c9c76fc3
```

```bash
gh release download solar-refiner-tf2500-20261111 \
  --repo RedRoman/Tencent-streaming-sr --dir release
cd release
sha256sum -c SHA256SUMS
python load_task_adapter.py solar_refiner_tf2500_lora_rank16_fp32.safetensors
```

最后一条只做CPU权重检查，不启动大模型推理。未完成Release上传时需等待资产出现；不要下载旧训练checkpoint作为替代。权重不包含基座、文本编码器、decoder、optimizer或RNG。

## 依赖和组合顺序

1. 使用被许可的 [MiniMaxAI/MiniMax-H3 Ref2VA](https://huggingface.co/MiniMaxAI/MiniMax-H3)，按配套转换器准备Diffusers transformer。原source index、转换器与转换结果逐文件SHA见[base conversion收据](../../results/solar_refiner_tf2500_20261111/evidence/base_conversion.json)。原缓存的HF revision没有记录，不能用虚构revision替代这些hash。重新序列化造成index变化时，应查清文件内容与转换差异，不跳过加载器断言。
2. 安装SolarRenderer源码固定revision `0ca4f5e45b062cc9e9aee506de46aabdc22e4c3d`。环境采用兼容的torch、safetensors和配套Diffusers；原Ascend环境为torch 2.7.1、torch_npu、CANN9.0.1和修订版Diffusers0.40.0.dev0。CUDA/B200环境尚未用本任务权重验收。
3. 取得以下两个上游文件，revision固定 `c5512be825f297ed8db390cbeda0289fb0c24779`，核对[原SHA](../../results/solar_refiner_tf2500_20261111/evidence/public_assets.json)，先合并causal control，再合并causal 4step，各rank384/alpha384。采用上游 `merge_lora_delta` 的逐模块FP32计算与逐次BF16舍入顺序。参考保存的[组合源码](source_snapshot/prepare_causal_model_cpu.py)；其中环境路径已脱敏，需映射自己的工作目录。

```bash
git clone https://github.com/Enderfga/SolarRenderer.git
git -C SolarRenderer checkout 0ca4f5e45b062cc9e9aee506de46aabdc22e4c3d
python -m pip install -e ./SolarRenderer
hf download Enderfga/SolarRenderer \
  solarrenderer_causal_control_lora.safetensors solarrenderer_causal_4step_lora.safetensors \
  --revision c5512be825f297ed8db390cbeda0289fb0c24779 --local-dir upstream_weights
```

两者都不是本次新训练任务adapter。组合后的模型index SHA应为 `5ba637cf6e4cbab0fcfd223646cecddcaf831a46b16256109a5c92353d7e018e`，文件名为 `diffusion_pytorch_model.safetensors.index.json`。

4. 最后加载本次rank16任务分支。它保留训练时的独立FP32运算，**不能把task LoRA直接并入BF16底座并声称与本次评估等价**。

```python
from pathlib import Path
import torch
from solarwm.backends.minimax_h3.model import SolarMiniMaxH3Transformer3DModel
from solarwm.backends.minimax_h3.optional import H3_FP32_PARAMETER_NAMES
from load_task_adapter import attach_task_adapter

model_dir = Path("prepared_causal_4step/transformer")
model = SolarMiniMaxH3Transformer3DModel.strict_from_pretrained(
    str(model_dir), torch_dtype=torch.bfloat16,
    local_files_only=True, low_cpu_mem_usage=True,
).eval().requires_grad_(False)
for name, parameter in model.named_parameters():
    assert parameter.dtype == (
        torch.float32 if name in H3_FP32_PARAMETER_NAMES else torch.bfloat16
    )
# 按机器容量安排基础模型的位置；单卡容量不足时不要盲目把整个pipeline搬入显存。
# 例如CUDA机器已核实容量后：model = model.to("cuda:0")
model = attach_task_adapter(
    model,
    "release/solar_refiner_tf2500_lora_rank16_fp32.safetensors",
    model_dir,
)
```

以上是参数装配示例，不冒称已在新机器完成推理。移动设备可用 `.to(device)`，不要在装配后 `.half()` 或 `.bfloat16()`，它们会把任务buffer也降精度。原推理没有全局autocast；任务分支保持FP32，底座保留其混合精度规则。

## 实际推理调用与输入

输入是自己的授权draft latent、匹配的prompt embeddings/标签及静音条件，不能用训练GT作为推理内容。`source_snapshot/stages/task_tf2500_v2/patch_core.py` 的 `batch_for(..., inference=True)` 把目标内容置零，并把draft作为对齐结构条件与首帧外观参考。

与已经执行的step1000回放保持相同调用语义：

```python
from solarwm.backends.minimax_h3.flow import make_shifted_schedule
# core是针对当前模型/设备构建的PatchCore；batch按上述输入契约构建。
# device上的CUDA/NPU attention适配应与平台匹配；不要在CUDA上安装Ascend补丁。
schedule = make_shifted_schedule(5, shift=12.0)
with torch.no_grad():
    refined, *_ = core.stream(
        batch,
        generator=torch.Generator(device=device).manual_seed(20261109),
        schedule=schedule, exit_index=3, window=6,
        cache_structure=False,
    )
```

每块四次去噪和一次clean-cache提交。67latent超过上游47latent训练域，原实现的长片时间位置处理要求重算结构条件窗口；不应直接改变此开关并宣称结果不变。GT只用于事后计算指标。

原[step1000完整回放源码](source_snapshot/eval_step1000/infer_v2.py)保留了输入、model index、源码和checkpoint的固定SHA。它是**历史执行快照**，仍绑定step1000训练checkpoint；不能直接启动它来代表2500。公开加载器加载2500推理delta，但本次发布未执行新的2500完整67latent回放。远端原训练/评估源文件未修改。

可选StreamRender causal ViT24仅用于像素解码，权重来自 `Efficient-Large-Model/StreamRender-H3` revision `9f81b42fb6bac3509ccc6ebf4cb6c8aec7469c62`，SHA `a1455dd6fedb61b1b835c3e1d2e5bad4a132c02a45585f992372ca2d25e22889`。不包含在本次adapter内，decoder与refiner的速度、因果范围和画质需要分别验证。

## 复现、许可和证据边界

`source_snapshot/`来自原冻结实现，仅替换私有路径；`PUBLIC_MANIFEST.json`同时记录原SHA和公开SHA。冻结配置内原hash用于追溯，不等于脱敏文件的新hash。跨机器训练需要重新绑定路径与资产、通过对应门槛；本目录不提供或重新启动训练队列。

代码及模型许可分开：上游代码见 `LICENSE.upstream-code.txt`；任务权重沿用 `LICENSE.weights.txt` 的MiniMax H3 Community License，包括地域和使用限制，分发附 `NOTICE`。这不是不限用途的Apache模型权重。

原完整checkpoint SHA `aec47799fbdf3425b8d11828653eaccc0349faa011919a8bd2fbe9fd9a9de422`仅作来源证明。400组AdamW步数、有限矩与CPU RNG恢复在导出前核验；发布仅保留参数和小型JSON审计。模型不是已经验收的最佳版本，也不保证训练后零劣化或无幻觉。
