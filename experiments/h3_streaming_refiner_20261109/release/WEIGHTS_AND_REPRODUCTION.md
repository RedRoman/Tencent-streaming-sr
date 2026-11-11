# 候选权重与复现说明

发布版本：[h3-streaming-refiner-candidates-20261111](https://github.com/RedRoman/Tencent-streaming-sr/releases/tag/h3-streaming-refiner-candidates-20261111)。这是研究候选预发布，两个检查点均未通过完整画质验收，不指定最佳模型。

## 下载与身份

```bash
gh release download h3-streaming-refiner-candidates-20261111 \
  --repo RedRoman/Tencent-streaming-sr --pattern '*.pt' --dir weights
gh release download h3-streaming-refiner-candidates-20261111 \
  --repo RedRoman/Tencent-streaming-sr --pattern SHA256SUMS --dir weights
cd weights
shasum -a 256 h3_anchored_refiner_*.pt
```

`SHA256SUMS` 还列出视频、报告和 JSON 附件；若只下载了权重，只核对下面两个条目，不把其他未下载附件当作损坏。

| 文件 | 字节数 | SHA256（与原训练推理检查点完全相同） |
| --- | ---: | --- |
| `h3_anchored_refiner_stage1000_cumulative3500.pt` | 82,508,131 | `f16119620e6bcbce0c6da7ce00560dc1733e9d0b3c0d6d4107c38974d43c50a6` |
| `h3_anchored_refiner_stage2000_cumulative4500.pt` | 82,508,131 | `5e95b0826db51003968597b8ffbc6159f42d0c784ad8c0a10843734fae3dc485` |

每个文件含 401 个 adapter 张量和 4 个残差头张量，共 20,593,528 个 FP32 参数。401 个 adapter 张量包括 50 层 Q/K/V/out 的 rank-8 A/B 参数与 `condition.weight`；4 个头张量对应 48→96→24 的线性层。`base_step=2500`，`stage_step` 分别为 1000/2000，`step` 分别为 3500/4500。累积步数跨两种训练配方。

发布文件保留原始字节，没有移除或重新序列化张量；原文件本来就是推理检查点，不含 optimizer、RNG、KV、训练缓存或 33B 基座。恢复精确训练需要未公开的私有恢复文件；本次发布不提供该能力。详细形状与逐张量 SHA 见 [CPU 权重审计](../../../results/h3_streaming_refiner_20261109/weight_release_CPU_audit_v1.json)。

## 先做 CPU 检查

在仓库根目录、已安装 PyTorch 的 Python 环境中执行：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 ASCEND_VISIBLE_DEVICES='' ASCEND_RT_VISIBLE_DEVICES='' \
python experiments/h3_streaming_refiner_20261109/release/load_candidate.py \
  weights/h3_anchored_refiner_stage1000_cumulative3500.pt --stage 1000
```

该入口先核完整文件 SHA，再用 `weights_only=True` 在 CPU 读取，检查显式 checkpoint 身份、405 张量、有限值和参数总量，并用原始 `ResidualHead` 严格加载四个头张量。真实两个文件均已通过此检查；错配 stage 在反序列化前被拒绝。它不读取 private resume，不加载 33B，不执行 NPU 前向，不能把成功输出当成新环境的模型推理验证。[CPU 验证收据](../../../results/h3_streaming_refiner_20261109/release_loader_CPU_validation_v1.json)。

## 必需依赖与固定版本

本发布提供增量 adapter 和残差头，仍需要原模型和同一冻结 upsampler。不要将它直接当作完整 MiniMax-H3 权重或通用 LoRA 文件导入其他工作流。

| 依赖 | 固定来源 |
| --- | --- |
| H3 base | [MiniMaxAI/MiniMax-H3 @ 6818f6c32d12b210915e44ad56a4228c2608f160](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/6818f6c32d12b210915e44ad56a4228c2608f160) |
| Open-H3 源码，含 `open_h3` | [NVlabs/Sana @ 781e8341fb3b29d3aea5d50265772e9619340b66](https://github.com/NVlabs/Sana/tree/781e8341fb3b29d3aea5d50265772e9619340b66/models/Open-H3-Regenerate-2K) |
| 合并 refiner LoRA / Sharp 分支 | [Open-H3-Regenerate-2K @ 50c09cf1df5f9262b9279868c1c666bd5fe3476b](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K/tree/50c09cf1df5f9262b9279868c1c666bd5fe3476b) |
| timestep curve table | [Comfy-Org/MiniMax-H3 @ e5eb578a89295337b8ff433a035929ce0279e0b6](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/e5eb578a89295337b8ff433a035929ce0279e0b6)，由上游下载器提取 `[1025,8]` F32 表 |
| 冻结 teacher step2000 upsampler | [既有 Release 权重](https://github.com/RedRoman/Tencent-streaming-sr/releases/download/h3-limiteddev-226f-20261108/h3_limiteddev_teacher_step2000_fp32.pt)，1,381,220,283 字节，SHA `7b9f43c17462fb2d0554645bc7889dc7abdebc052ee9f18f6780624de912e0b5` |

上述 teacher step2000 与 v5 step1800 不同。它是本路线固定的比较基线；其历史 native-HR guard 失败也保留，不用“冻结基线”暗示全面保真合格。

运行布局：

```text
asset_root/
  merged_transformer/              # 原 base 加官方发布 refiner LoRA 后的 14 个 shard
  assets/lora/
    open_h3_regenerate_2k_lora.safetensors
    sharp_curve_branches.safetensors
    adaln_t_table.bin
```

CPU 合并代码已保存在 [merge_refiner.py](../../h3_refiner_self_forcing_20261104/merge_refiner.py)：对 312 个目标按 `cast(W_fp32 + B_fp32 @ A_fp32, original_dtype)`、scale=1 合并。Sharp 50 层分支由运行时另行安装，strength=1。不得用新发布 IC 版或不同合并版本替代。本次完整重读审计的 14 个 merged shard、config、index 和附加资产 SHA 在 [依赖清单](DEPENDENCIES_SHA256.json)；`MERGED.json` 的机器路径不作为公开部署配置。

实际验证环境为 Python 3.12、torch 2.7.1、torch_npu 2.7.1.post2、CANN 9.0.1、兼容集群适配的 diffusers 0.40.0.dev0、safetensors 0.8.0、transformers 5.15.0、numpy 1.26.4。**该 diffusers 运行时补丁未随仓库发布，仅安装同名版本号不能保证全新环境可运行。** 需要能导入 `MiniMaxH3Transformer3DModel` 及 `diffusers.modular_pipelines.minimax_h3.before_denoise` 的兼容实现。公开源码/权重与实验复现证据不等于已验证的一键安装包；本次新增 loader 仅做 CPU 验证。

模型和第三方组件沿用各自来源及许可，不能用仓库代码许可覆盖模型许可；参见 [上游模型卡](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K#license)、既有 [SolarWM notice](../vendor/NOTICE.SolarWM) 和随源码保留的许可。

## 将权重安装到原始流式运行时

下面是集成接口示例；调用者需先完成设备空闲、内存与共享锁检查，并在整个 worker 生命周期持有同机约定的三把锁。该段不是自动调度脚本，也没有在本次发布中重新执行 NPU 推理。

```python
import sys
from pathlib import Path
import torch
import torch_npu

code = Path('experiments/h3_streaming_refiner_20261109')
sys.path[:0] = [str(code / 'src'), str(code / 'release')]
# 另将上述固定版本中包含 open_h3 的目录、兼容 diffusers src 加入 PYTHONPATH。
from paged_runtime import Runtime
from anchored_refiner_v1 import AnchoredStreamingRefiner, align_input_projection
from load_candidate import load_candidate, load_head

torch.npu.conv.allow_hf32 = False
torch.npu.matmul.allow_hf32 = False
torch.set_float32_matmul_precision('highest')
weights = load_candidate('weights/h3_anchored_refiner_stage1000_cumulative3500.pt', stage=1000)
runtime = Runtime('asset_root', device='npu:0', rank=8, resident=24, canonical_rows=True)
align_input_projection(runtime)  # 每个新 runtime 只调用一次
runtime.load_adapter(weights['adapter'])
head = load_head(weights, device=runtime.device)
stream = AnchoredStreamingRefiner(runtime, head, condition,
                                 sample_id='WingIt_dev_3586_3811', seed=20261109)
# condition 与 draft_chunk 必须由调用者按下述合同提供。
# 每次可喂一个或多个 latent；内部始终逐 latent 因果处理。
refined_chunk = stream.push(draft_chunk)   # [1,24,T,68,120] -> 同形状 CPU FP32
print(stream.audit())
```

`condition` 是原评估准备的静态条件字典。实际用无文本条件及已知全长静音的真实 audio-VAE 编码；不可用零张量冒充静音编码，也未验证未知未来音频或生产文本 prompt。字段和生成过程见 [prepare_pairs.py](../src/prepare_pairs.py) 与冻结 [stage1000 evaluator](../stages/anchored_step1000_evaluation_v1/src/run_anchored_step1000_evaluation_v1.py)。每个新 clip 建立新 runtime stream；sample_id 与 seed 参与绝对位置噪声，改变它们会改变输出。

`draft_chunk` 来自上述固定 teacher，LR `[1,24,67,44,80]` → draft `[1,24,67,68,120]`。原评估所用配对数据、GT、静音条件及缓存均不作为 Release 权重附件提供，用户需要自行准备有授权的输入。主画质对比通过 official overlap decoder 输出 1920×1088，再裁 y=4:1084 到 1080p；这段 decoder 不是严格流式。严格 decoder 的独立保真失败见完整报告。

原冻结 evaluator 额外绑定私有训练完整性收据，不能直接以公开 adapter 替代所有输入文件后宣称原实验复现成功。上述 loader 专门提供无需 optimizer/RNG 的推理权重加载路径；全部历史 gate、source SHA 和 original/public 映射仍保留。未来新环境应重新做实际模型因果、chunk、cache 和像素验证；本版本没有启动任何新训练或 NPU 重跑。
