# H3 CMN hybrid：teacher KD + decoder-aware pixel supervision

**状态：2026-11-05 正式训练进行中，尚无本轮最终质量结论。** 本目录保存唯一一个 combined warm-start 2000-step run 的实现与预检证据；最终选模、完整评估和视频结果完成后再补充。预检通过不代表质量提高。

从[前轮 CMN](../h3_cmn_upscaler_20261105/README.md) 的 dev-selected step9500 继续训练，仍是 width64、每 stage 2 blocks、1,061,336 参数的同一个因果 latent upscaler。新增冻结原版 H3 upscaler 的 latent teacher KD，以及经过冻结 H3 decoder 的 RGB L1/LPIPS。teacher 不出现在部署推理链中，decoder 没有训练。本轮没有 Self Forcing、DMD、refiner 更新，也没有同预算单因素消融，不能把最终差异单独归因于某一项损失。

## 数据与目标

沿用同一 SHA 的 128 train / 16 dev / 16 test + 3 legacy。120 帧/24fps 的 HR 源 RGB 经严格 2× area 下采样得到 LR，再分别编码为 H3 latent；这次输入不是实际低清生成模型的输出。teacher 对已保存的 LR latent 运行原版双向 H3 upscaler，按原实现 N→U→D（latent 归一化→上采样→反归一化）规范计算 native normalized latent 标签，只缓存 train/dev 144 条。teacher 标签本身不需要 HR 真值，而本轮其他损失继续使用成对 HR。

训练目标为：

```text
L = HR latent MSE + 0.25 × teacher latent MSE
    + 0.1 × eWarp L1 + 1.0 × RGB L1 + 0.2 × LPIPS(AlexNet, v0.1)
```

HR MSE 和 KD 使用前 36 个真实锚点 latent；eWarp 使用 35 对有效时间位置及既有 backward-flow/in-bounds mask。flow 和 HR RGB 不作为 CMN 输入。RGB target 是冻结 decoder 解码的 **HR reference latent**，并非 teacher 输出，也不是原始 RGB。第 37 个 latent 保留为原 decoder 的 context，虽然不进入 latent MSE/KD，它可经像素分支接收梯度。LPIPS backbone 与线性权重均冻结；预测 RGB 的解码保留 autograd，目标解码不反传。

每 4 次更新有 1 次使用完整空间区域（25%），其余 3 次使用原 decoder 原生空间 tile 的内部区域（75%）。分辨率为 384×672，tile256、overlap64；内部区域覆盖四角，完整空间分支补足中心。每次只采一个原生 7-latent、stride5 的时间窗口。排除跨窗口 blend 的 5 帧及合成尾帧后，训练像素损失可采到 **90/120 个真实时间位置**，含 frame119；这不表示每次更新覆盖 90 帧，也不是全视频像素损失的无偏估计。RGB L1 使用区域内全部有效帧与未截断浮点 RGB；LPIPS 每区域选 4 帧，将 RGB clamp 到 [0,1] 后映射到 [-1,1]。

FP32、batch1、fresh AdamW、weight decay0.01、grad clip1；新增 2000 updates，lr3e-5、min_lr3e-6、warmup100、cosine schedule、seed20261105。每 200 步按 16 个 dev 视频的固定完整空间时间区域均值 `RGB L1 + 0.2 LPIPS` 选模，包含 step0；这是区域代理选模，不是全120帧分数。test/legacy 只在选模后读取，且这些样本此前已被检查，不能称为新盲测。

CMN 对固定输入 latent 保持因果；原 H3 decoder 的窗口内 attention 仍双向，因此不宣称端到端零前视。完整解码保留全部37latent，解码124帧再裁到120。最终评估计划同时报告全120帧以及 prefix102/tail18，后者标示可能受 padding context 影响的尾段。

## 预检证据

首次预检在将完整空间多 tile 解码整体 checkpoint 时 OOM。改为**逐个原生空间 tile checkpoint，再使用原始 `_stitch_tiles`** 后，第二次 8-update 预检通过：12 项原生区域与完整解码对照的 max-abs 均为 0，RGB L1/LPIPS 的输入梯度有限且非零、区域外梯度为 0，冻结模块未改变；峰值 allocated **20.41286 GiB**（reserved 20.58594 GiB）。详见 [preflight_evidence.json](preflight_evidence.json)。这不是所有设备/尺寸的显存保证；当前正式训练完成后才能报告最终峰值与耗时。

## 环境与运行

复用[前轮 CMN 环境](../h3_cmn_upscaler_20261105/README.md)及其 helper 链；不在脚本内安装依赖。原运行使用 Python3.12、torch2.7.1、torch_npu2.7.1.post2、torchvision0.22.1、lpips0.1.4、CANN9.0.1 和兼容 H3 的 patched diffusers0.40.0.dev0。还需 NumPy、Pillow、Matplotlib 与 FFmpeg。原环境缺 libx264，helper 回退输出 MPEG-4；发布时另行用本地 avconvert 转 H.264 并完整解码审计，转换不参与未压缩 RGB 指标。LPIPS AlexNet backbone 与包内 `weights/v0.1/alex.pth` 必须预先缓存，权重 SHA 见证据文件；训练/评估不应现场下载模型。

先按本机位置配置：

```bash
export EXP_ROOT=/your/external/experiment-data
export H3_BASE_MODEL=/your/models/h3-base
export H3_UPSTREAM_DIR=/your/Sana/models/Open-H3-Regenerate-2K  # 包含 open_h3
export H3_DIFFUSERS_SRC=/your/patched-diffusers/src
export H3_CANN_ENV=/your/cann/set_env.sh     # 已配置 CANN 时可省略
export H3_PYTHON_BIN=/your/python-env/bin    # 可省略，含 python 可执行文件的目录
export H3_FFMPEG_BIN=/your/ffmpeg/bin        # 可省略
source ./env.sh
```

`H3_ULTRAFLASH_PAIRS` 默认 `$EXP_ROOT/upscaler_ultraflash/pairs`，须含 finalize 后的 plan/READY、4 个 manifests、原始 paired tensors 和匹配 flow。`H3_CMN_INITIAL_CHECKPOINT` 默认前轮 `best_step_009500.pt`；`H3_CMN_EVALUATION` 须指向与该 checkpoint 相符的完整原始评估目录，包括19条记录、4例缓存 latent、不可变 latent_summary 与各项 SHA。仓库的精简 results JSON/视频不能代替这些外部 tensor/checkpoint。teacher 权重由 `H3_UPSCALER_WEIGHTS` 指定；其余目录可用本目录 env.sh 中的 `H3_HYBRID_*` 变量覆盖。

```bash
python -B "$EXP_CODE_DIR/test_h3_cmn.py"
python -B "$EXP_CODE_DIR/test_ultraflash_ewarp.py" --device cpu
python -B "$EXP_CODE_DIR/test_h3_hybrid_pixel.py"
# 5 仅示例；必须换成已确认空闲的物理 NPU。
bash "$EXP_CODE_DIR/run_prepare_h3_hybrid_teacher.sh" 5
bash "$EXP_CODE_DIR/run_train_h3_hybrid.sh" 5 preflight
bash "$EXP_CODE_DIR/run_h3_hybrid_pipeline.sh" 5
```

每个计算阶段通过共享 advisory lock 和 `npu-smi` 再查空闲状态；没有默认占用的卡号。pipeline 顺序执行正式训练、19条独立重载与36项因果检查、4例同 VAE 重解码/LPIPS、完整媒体解码审计与 CPU 绘图。仅评估可分别调用：

```bash
bash "$EXP_CODE_DIR/run_evaluate_h3_hybrid.sh" 5 latents
bash "$EXP_CODE_DIR/run_evaluate_h3_hybrid.sh" 5 decode 0 1
bash "$EXP_CODE_DIR/run_evaluate_h3_hybrid.sh" cpu finalize 1
# 中断后恢复同一训练目录；保留同一源码/参数/数据合同。
bash "$EXP_CODE_DIR/run_train_h3_hybrid.sh" 5 train --resume "$H3_HYBRID_RUN_DIR/latest.pt"
```

fresh run、preflight、latent evaluation 拒绝覆盖既有目录；不要用 pipeline 重启已经完成的训练。恢复保留 optimizer、采样 RNG、最佳 checkpoint 和已保存步骤日志，未声称 bitwise resume 已验证。最终像素对比预定五列为 pretrained、旧 causal Conv3d eWarp、CMN、hybrid、reference；旧 Conv3d 不是 Method2。四例全120帧 LPIPS 基于新解码的未压缩 RGB，不能以有损 MP4 代算；历史基线来自不同架构/预算，不能当等算力消融。

## 源码与证据边界

[source_fingerprints.json](source_fingerprints.json) 同时记录实际运行源码 SHA、公开文件 SHA、变化说明、继承 helper 的位置与 SHA。模型、像素 helper、tests、evaluator 和 plot 原样复制；公开 trainer 将私有 base-model 默认改为 `H3_BASE_MODEL`，teacher prep 接受 `H3_UPSTREAM_DIR`，启动脚本改为显式环境变量/物理卡参数。**原始与公开 SHA 不相同的文件不冒充同一源码。**

trainer 按同目录校验 `h3_cmn.py`、`ultraflash_ewarp.py`、`data_utils.py`，所以三者必须随本目录提供。`train_h3_cmn`、`evaluate_h3_cmn`、`train_upscaler_ultraflash` 及更早的 helper 通过继承 PYTHONPATH 定位；evaluator 的 module_source 校验实际导入源文件。

公开版本适用于形成新的、对应公开 SHA 的运行合同；原始 checkpoint/cache 严格绑定原始源码，公开改路径后的源码不能直接通过原 run 的 resume/evaluate 合同。若复核原 run，应使用原执行快照与原合同；若复跑公开版本，应使用独立 teacher cache/output 重新建立合同。这里不放宽 SHA 校验、不改历史 checkpoint，也不包含模型权重、原视频、访问凭据或训练缓存。
