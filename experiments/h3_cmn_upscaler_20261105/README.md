# H3 CMN-inspired latent upscaler

本轮只训练新的 latent upscaler，继续使用原有冻结 H3 decoder。[Ultra Flash v4 §2.2](https://arxiv.org/html/2606.09150v4#S2.SS2) 提供 CMN/eWarp 思路；这里是 H3 适配原型，未声称复现未公开的完整训练 recipe，也未训练 CMN decoder。

模型有 **1,061,336 参数**，宽度 64，3 stages、每 stage 2 blocks，空间因子 `[1,2,1]`、时间因子 `[1,1,1]`。每个 block 将当前与上一输入 feature concat，经 conv1→ReLU→conv2→ReLU→conv3→skip→ReLU；初始 memory 使用零，是区别官方默认 first-feature replication 的本地选择。前向直接接收/输出 native normalized 24-channel H3 latent，不额外套原版 upscaler 的 N/D。

随机初始化网络，zero residual head 加在逐帧 2× bilinear base 上，初始输出严格为插值。第一步内部梯度为零但连接完整，head 更新后内部获得梯度。每层保存上一输入 feature，训练 forward/chunk 向量化为 B×T Conv2d；独立 forward_step 用于真正逐 latent 推理，state 跨任意 chunk 和第5 token边界保持。`detach_state` 只用于显式 TBPTT，不隐式切断历史。这里不使用 decoder timing helper，也不修改 H3 decoder。

沿用前轮同一 SHA 的 128 train / 16 dev / 16 test + 3 legacy；这些 retained test 已用于前轮比较，不能称为新盲测。每条120真实RGB帧/24fps，37latent中前36个真实锚点进入MSE、35对进入eWarp。配对 VAE 内部 padding 与既有非因果 decoder 保持一致。光流为 GT quarter-resolution Farneback backward flow，仅 in-bounds mask、无遮挡检测；完整RGB和flow不作为模型输入。

本轮单个 eWarp 主 run：新增 10000 步，MSE+0.1×eWarp L1，FP32 AdamW，lr=0.0002、min_lr=2e-05、warmup=200，batch1、clip1，seed=20261105；每 500 步以 dev latent MSE 选模，包含step0。当前没有本轮同架构同预算 MSE-only 对照，旧Conv3d实验同时改变架构、参数量、初始化和训练预算，不能把二者差异单独归因为CMN或eWarp。

## 运行

复用[前轮环境与数据准备](../h3_upscaler_ultraflash_20261104/README.md)。配置 `EXP_ROOT`、`H3_BASE_MODEL`、`H3_UPSTREAM_DIR`、`H3_DIFFUSERS_SRC` 及运行时 PATH；`H3_ULTRAFLASH_PAIRS` 指向已 finalize 且含匹配flow的旧pairs。`env.sh` 继承前轮helper链并将本目录置于PYTHONPATH首位。

本目录额外原样包含 `ultraflash_ewarp.py` 和 `train_upscaler_ultraflash.py`：冻结trainer按同目录查找它们并校验源码哈希，因此只设置PYTHONPATH不足以满足该contract。其余VAE/像素helper从邻目录继承；`plot_h3_cmn.py` 生成真实训练曲线。未使用的decoder timing两文件不随本轮发布。

```bash
source ./env.sh
python -B test_h3_cmn.py
# 以下5仅示例已确认空闲的物理NPU。
bash run_train_h3_cmn.sh 5 10000
bash run_evaluate_h3_cmn.sh 5 latents
bash run_evaluate_h3_cmn.sh 5 decode 0 1
bash run_evaluate_h3_cmn.sh cpu finalize 1
python "$EXP_CODE_DIR/plot_h3_cmn.py" --run "$EXP_ROOT/upscaler_cmn/ewarp_steps10000" --output "$EXP_ROOT/upscaler_cmn/training_curves.png"
```

所有decode显式传 `H3_BASE_MODEL`，评估保留全部37latent供原有decoder解码124帧再裁到120；upscaler相对固定latent的因果性不等于端到端源像素零前视。恢复接口保留训练contract/optimizer/RNG/log，但不声称bitwise resume已验证。源码哈希与训练contract对应关系见[结果证据](../../results/h3_cmn_upscaler_20261105/source_fingerprints.json)，结果与视频见[结果README](../../results/h3_cmn_upscaler_20261105/README.md)。不包含checkpoint、cache或原始源视频。
