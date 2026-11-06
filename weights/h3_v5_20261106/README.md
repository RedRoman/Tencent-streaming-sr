# H3 causal upscaler：五版 FP32 推理权重

[GitHub Release](https://github.com/RedRoman/Tencent-streaming-sr/releases/tag/h3-v5-weights-20261106) 保存五个 `safetensors`；Git 仓库只保存加载器、配置、许可证和核验记录。每版权重约 1.38 GB（十进制），共约 6.91 GB。六格视频的第六格 `reference` 是 decoded HR 视频，没有对应第六个训练权重。

这些版本用于复现实验与定位因果改造损伤。**motion 的部分数值小幅改善，但没有解决主要视觉伪影；大运动、面部和细节仍可能明显退化，不应视作已解决版本。** v5 是已经冻结的基线，也不是无损因果转换或质量保证。参见[现有进度与六格对比](../../results/h3_v5_progress_20261106/README.md)、[v5 版本定义](../../versions/v5/README.md)和[16 条测试补评](../../results/h3_conv3d_test16_20261105/README.md)。复用过的测试集不构成新盲测。

## 五个版本

| 版本 | 本轮选中 step | 从因果转换起累计更新 | 权重下载 | 含义 |
|---|---:|---:|---|---|
| `zero_aligned` | 0 | 0 | [h3_causal_zero_aligned_fp32.safetensors](https://github.com/RedRoman/Tencent-streaming-sr/releases/download/h3-v5-weights-20261106/h3_causal_zero_aligned_fp32.safetensors) | 因果算子与 aligned-past 初始化，未微调 |
| `aligned50` | 50 | 50 | [h3_causal_aligned50_fp32.safetensors](https://github.com/RedRoman/Tencent-streaming-sr/releases/download/h3-v5-weights-20261106/h3_causal_aligned50_fp32.safetensors) | 早期 50 步微调 |
| `old_ewarp` | 2000 | 2050 | [h3_causal_old_ewarp_fp32.safetensors](https://github.com/RedRoman/Tencent-streaming-sr/releases/download/h3-v5-weights-20261106/h3_causal_old_ewarp_fp32.safetensors) | 原 causal eWarp 训练选中版本 |
| `v5` | 1800 | 3850 | [h3_causal_v5_fp32.safetensors](https://github.com/RedRoman/Tencent-streaming-sr/releases/download/h3-v5-weights-20261106/h3_causal_v5_fp32.safetensors) | Conv3d hybrid 正式 v5 基线 |
| `motion` | 2000 | 5850 | [h3_causal_motion_fp32.safetensors](https://github.com/RedRoman/Tencent-streaming-sr/releases/download/h3-v5-weights-20261106/h3_causal_motion_fp32.safetensors) | 从 v5 继续的 motion 目标实验；主要视觉伪影仍未解决 |

`v5` 从 2,000 步训练中按 dev 选择 step 1,800；累计更新为 `50 + 2000 + 1800 = 3850`。`motion` 继续 2,000 步，累计 5,850。累计列只计所选模型的真实训练链，不把中断并丢弃的 attempt 更新加进去。`zero_aligned` 的 0 步表示没有微调，不表示参数与原双向 teacher 相同：初始化已丢弃未来卷积系数并重新排列 past/current taps。

每个文件仅含 322 个原名张量、345,280,216 个参数，全部保留 FP32 位模式，无 FP16/BF16 转换或量化，无 optimizer/RNG/训练缓存。`state_dict_format=original_upscaler`，空间倍率固定 ×2，24 通道 native H3 latent，使用左侧时间 padding 和逐帧 GroupNorm。

## 原始 checkpoint 与公开文件映射

容器从内部 `.pt` 变为不含 pickle 的 `safetensors`，文件 SHA 必然不同；下表保留两者身份，而不是把导出文件冒充原文件。完整 bytes、配置 SHA、每张量 SHA 见 [weights_manifest.json](weights_manifest.json) 和各版本 `*_tensor_validation.json`。

| 版本 | 原 `.pt` SHA-256 | 公开 `.safetensors` SHA-256 |
|---|---|---|
| `zero_aligned` | `e989d93ec87e751d295b9ec13a0af1c41401f1d75b69c4e25b2a3596c9dbc31e` | `30ff18236a79b718e15bd86dbbff5c44d506f28bd9dfa22a6ad416528d433fa5` |
| `aligned50` | `76e9afc97bfc31029fecb64c249ae23d26fcdfc6745f3a2f10ccc57ce82b222c` | `29e3555fb220d647b7b745aa035eca5425dd361deba62b6ba971714500a77eec` |
| `old_ewarp` | `b09291baab8265557430794b9b44154f5e05b37cc0c603228fb2c2922f38c2a6` | `fcfc25d7545cf2470836330101c9f0eb049824c8475ab09716a31be3be3f32c5` |
| `v5` | `c7aaaa396d877a84c3fed2a5536189b7ed5e61226a82ef1a9e6be1e831f7f914` | `a9331b3e16ada52a504b71ebf5a60e9149a1e0dc92d9bdda36247a4e714a3e31` |
| `motion` | `b4311c9e91365dfc0a5a6014d437e8d34f1c7d71bc31e12879b45b0560652d37` | `a7f10b0a7eb7cd14b643d7dc110653c854ddb1b0a65354d477f43b62621a0b44` |

`weights_manifest.json` 是**导出时的原始快照**，其中 `publication_status: not_uploaded` 有意保留原字节；当前上传状态及 GitHub 实际 asset ID/size/digest 以 [github_assets.json](github_assets.json) 和 Release 页面为准。所有配置、导出核验与 CPU 加载核验同样保留原始字节。

## 加载与校验

依赖 Python、PyTorch、NumPy、safetensors、einops。CPU 加载检查不需要 H3 base/decoder/refiner 权重，也不需要启动 NPU。先 checkout 包含本目录的公开分支，在仓库根目录运行：

```bash
repo_root="$(pwd)"
export PYTHONPATH="$repo_root/weights/h3_v5_20261106:$repo_root/experiments/h3_conv3d_hybrid_20261105:$repo_root/experiments/h3_causal_upscaler_20261104:$repo_root/experiments/h3_upscaler_trial_20261104:$repo_root/experiments/h3_cmn_hybrid_20261105:$repo_root/experiments/h3_refiner_self_forcing_20261104${PYTHONPATH:+:$PYTHONPATH}"

# 将选中权重放在对应 config 的同一目录；示例为 v5。
curl --fail --location \
  'https://github.com/RedRoman/Tencent-streaming-sr/releases/download/h3-v5-weights-20261106/h3_causal_v5_fp32.safetensors' \
  --output "$repo_root/weights/h3_v5_20261106/h3_causal_v5_fp32.safetensors"

# 仅下载精确版本的两个外部源码文件，保留 open_h3 目录层级。
upstream_dir="$repo_root/external/Open-H3-Regenerate-2K-781e8341"
mkdir -p "$upstream_dir/open_h3"
for name in upscaler.py upscaler_node.py; do
  curl --fail --location \
    "https://raw.githubusercontent.com/NVlabs/Sana/781e8341fb3b29d3aea5d50265772e9619340b66/models/Open-H3-Regenerate-2K/open_h3/$name" \
    --output "$upstream_dir/open_h3/$name"
done

TORCH_DEVICE_BACKEND_AUTOLOAD=0 ASCEND_RT_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
python weights/h3_v5_20261106/load_h3_causal_release.py \
  --config weights/h3_v5_20261106/h3_causal_v5_config.json \
  --upstream "$upstream_dir"
```

CLI 构造原架构、加载参数、安装因果 adapter 后逐张量校验，不执行 forward。源码 SHA、权重 SHA、配置、张量数或模型结构不匹配会拒绝加载。仅替换 variant 的文件名可检查其他版本。下载前请确认目标不存在，避免覆盖已有本地文件；`.gitignore` 防止把大权重加入 Git。

Python 入口（沿用上面的 `PYTHONPATH`）：

```python
from load_h3_causal_release import load_release

model = load_release(
    "weights/h3_v5_20261106/h3_causal_v5_config.json",
    upstream="external/Open-H3-Regenerate-2K-781e8341",
    device="cpu",
)
# 后续输入必须是规范的 native-normalized H3 latent [B, 24, T, H, W]。
# model 内部保留 D(U(N(z)))，不要在调用前重复应用 N/D。
```

保存的权重已经执行过 aligned-past 初始化；加载器只设置标记，**绝不能再次调用 `initialize_aligned_past_`**。把这些参数直接塞进未改造的双向节点也会改变行为。公开依赖与精确 SHA 见 [public_runtime_dependencies.json](public_runtime_dependencies.json)。

## 已验证范围与限制

五个导出均通过逐张量 `torch.equal` 和原始字节 SHA 比对，原 checkpoint 在导出前/后/最终汇总 SHA 一致。五个实际 CPU 架构和 adapter 加载均通过 322 张量 bitwise 核验，见 [cpu_load_validation.json](cpu_load_validation.json)。公开模块链另经隔离 CPU import 验证，见 [public_import_validation.json](public_import_validation.json)。这些检查没有执行模型 forward 或 NPU，不是新画质、速度或端到端复现测量。

因果范围仅是相对于固定 latent 输入的 upscaler；H3 VAE/decoder 及后续 refiner 没有因此变成因果模型。FP32 权重存储不保证 NPU 内部 HF32 卷积关闭；既有特定小尺寸长序列在默认 HF32 下曾不满足严格分块数值等价，HF32-off 的独立检查与实际 37-latent 生产尺寸检查应分别解读，不能推断任意尺寸/数学模式都等价。stream cache 由单实例持有，不支持同一实例并发或交错视频。

## 来源、许可证和修改声明

社区权重模型卡声明 [Apache-2.0](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/raw/main/README.md)，本目录 [LICENSE](LICENSE) 保存该标准文本。社区节点代码保留其 [MIT LICENSE](LICENSE.upstream-node-MIT)，版权归属与原文保持不变。这里只分别记录两种上游声明，不把整个 H3 或整个仓库统一解释为 Apache-2.0。许可证原文 URL/抓取 SHA 见 [license_provenance.json](license_provenance.json)，修改说明见 [NOTICE](NOTICE)。

本包不含原版双向 teacher 权重、H3 base、VAE/decoder 或 diffusion refiner 权重。使用这些依赖时仍需分别遵守其来源和适用许可。
