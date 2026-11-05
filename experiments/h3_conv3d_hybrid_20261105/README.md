# 原版 H3 causal Conv3d：teacher KD 与 decoder 像素监督

**状态：第一次正式尝试中断，第二次尝试已启动、正在初始化。** Attempt1已确认写入83条更新后退出，shell退出码143支持SIGTERM判断，发起者未知；未见Python traceback，未找到OOM证据。该次尚无包含optimizer/RNG的`latest.pt`，只有step0权重，无法恢复第83步的训练状态。原run目录与日志已按原字节归档，原合同和冻结模型/训练源码未改。Attempt2从同一初始权重、相同配置重新开始；本快照尚未确认新更新。**丢弃的83步不计入新run的2000步，也不拼接两次日志；没有新的最终画质结论。** 这次按用户要求将小 CMN 学生替换为此前已因果化并微调的原版 H3 upscaler，沿用 CMN hybrid 的互补损失与数据。不会把准备完成、审计脚本存在或 FP32 tensor dtype 当作训练通过的证据。

初始长序列检查使用真实345M checkpoint、163个latent、2×2 latent空间尺寸。默认NPU卷积HF32设置下，未来扰动、前缀及未来梯度检查为零，但流式与整段输出的最大绝对差为 **0.00112772**，未达到原先固定的一致性阈值，原失败JSON/log保留。独立进程关闭卷积HF32后，扩展的 **16项检查全部通过、最大差为0**，阈值未改变。这支持数值执行路径随形状变化的解释，不能将它当作默认HF32模式的通过证据。训练和decoder保持先前默认数学设置以便比较recipe；真实生产尺寸12×21、37latent上的初始/预检末尾因果及分块检查均通过，最大差为0。正式训练后的相同真实尺寸检查和独立长序列HF32off检查仍须重新通过。

中断发生于首次100步可恢复checkpoint之前，第二次尝试不能冒充resume。参见[attempt_history.json](evidence/attempt_history.json)与[interruption.json](evidence/failures/attempt1_terminated_step83/interruption.json)。记录保留失败run的steps、summary、contract及train.log原始SHA；这些SHA仍指归档原文件，不能用新尝试或公开副本的SHA替换。SIGTERM原因目前未知，现有证据不足以归因为OOM或某个任务。

## 模型与初始化

学生为原版 **345,280,216 参数** H3 Conv3d upscaler。沿用既有 aligned-past 改造：时间卷积只在左侧补零，GroupNorm 逐帧计算统计量，空间插值保持时间长度，scale condition 仅依赖空间大小；保留原模型空间通路、参数及 latent 归一化/反归一化。绕过原版双向 overlap 分块路径，逐层缓存历史输入 feature。原模型 dropout 保持关闭，参数梯度仍开启。

初始化是旧 causal eWarp 的 dev-selected checkpoint：先前50步适配后再训练2000步的版本。原始 checkpoint SHA256 为 `b09291baab8265557430794b9b44154f5e05b37cc0c603228fb2c2922f38c2a6`。已经对齐和微调的卷积核原样加载，**不会再次执行 aligned-past 权重平移**。新 checkpoint 保存原模型参数名，另附 adapter 配置；不将 wrapper 的 `model.` 前缀混入原始权重格式。

`h3_conv3d_hybrid.py` 实现模型装载与流式状态接口。状态只属于当前 adapter 实例，须顺序推进，不能当作可随意分叉的便携快照。结构感受野为149个latent，即当前位置与148个历史位置。架构具有因果依赖结构，不代表所有设备精度设置下整段/分块运算的数值差异自动达标。

## 数据、监督与选模

沿用相同源 plan SHA 的 **128 train /16 dev /16 test +3 legacy**。每段120真实RGB帧、24fps、5秒；HR 672×384 经严格2× area下采样得到LR 336×192，再分别由原H3 VAE编码。LR来自成对降采样，**不是实际低清生成模型的输出**。源文件SHA去重不等于语义近重复去重；test与legacy已在前轮查看，不能称为新的盲测。本轮不是2K训练。

Teacher 为原版双向 H3 latent upscaler，只给train/dev的144条LR latent提供缓存标签；teacher不进入学生部署链。像素目标来自冻结decoder解码的HR reference latent，不是teacher像素，也不是源RGB。

```text
L = HR latent MSE +0.25 × teacher latent MSE
    +0.1 × eWarp L1 +1.0 × RGB L1 +0.2 × LPIPS(Alex, v0.1)
```

37个latent中前36个真实锚点进入HR MSE/KD，35对有效时间位置进入eWarp。光流只用于loss，不作为学生输入。最后padding-only latent保留为原decoder上下文，允许经像素分支接收梯度。

像素监督复用原H3 decoder的7-latent、stride5时间窗与tile256/overlap64空间逻辑。每4步中1步使用完整空间时间区域，3步使用单个原生tile内部区域；所有时间窗口可采到的真实位置合计为90/120帧，排除跨窗混合位置与合成尾帧，含frame119。这不是每步覆盖90帧，也不是完整视频loss的无偏估计。L1使用所选区域全部有效帧和未截断浮点RGB；LPIPS每区域4帧，RGB clamp至[0,1]后映射到[-1,1]。预测解码保留autograd，目标解码no_grad；VAE和LPIPS参数冻结。

计划追加2000 updates，batch1、fresh AdamW、weight decay0.01、clip1、lr3e-5→3e-6、warmup100、cosine、seed20261105。参数和输入存储为float32；执行合同另外记录NPU卷积/矩阵HF32标志、torch/torch_npu版本与matmul precision，不能仅凭dtype宣称内部运算使用完整FP32尾数。实际预检沿用conv HF32=true、matmul HF32=false，正式训练仍须保持并核验其合同设置。

每200步在16个dev视频的固定完整空间时间区域计算 `RGB L1 +0.2 LPIPS` 均值，含step0，取首次最小值。该选模是区域代理，不是完整120帧分数。选定后才读取test/legacy；没有使用test选权重。

这是把相同互补recipe迁移到已预训练的大模型，没有新做等预算KD-only/pixel-only消融。与1.06M CMN不同的参数量、初始化及先前训练历史，使它不是等算力或仅改变架构的严格对照。**没有DMD、Self Forcing、diffusion refiner或VAE参数训练。**

## 检查与评估范围

真实345M权重已做163-latent小空间输入上的未来扰动、前缀、混合分块、cache reset、未来/历史输入梯度检查，结果按上述精度模式分别报告；8-update像素预检已完成。小空间检查验证运算与状态语义，不验证长视频主观质量。预检检查原生区域/完整解码一致性、decoder到学生的非零梯度、全部学生参数梯度连接以及冻结模块未变。任何失败保留原输出和源码SHA；修正后使用新的输出目录与对应合同。8-step预检训练含验证80.98秒，峰值allocated为25.7429GiB（reserved25.9883GiB），decoder到学生输出head梯度L2为0.3737575，冻结模块与执行精度设置未变；详细检查和原始/公开证据SHA见[preflight_evidence.json](preflight_evidence.json)与`evidence/`。

正式训练后计划独立重载19条样本，四条固定视频进行36项因果/分块检查，并逐例解码全部120帧。六列顺序是 **原版teacher → 旧causal eWarp → 原CMN → CMN hybrid → 新Conv3d hybrid → decoded HR reference**。旧causal eWarp不是Method2的15M学生；本实验没有直接重训Method2。

PSNR、LPIPS及时间差分指标在压缩前RGB上计算，主目标是相同冻结decoder解码的HR reference，源RGB仅作次要对照。像素指标只有3条选定test+1条legacy，不能写成16-test完整像素均值；latent指标覆盖16test+3legacy。完整视频还拆分prefix102/tail18，并对28个原始MP4做全片技术解码检查。发布的视频转换不参与指标计算。

Upscaler因果性只相对于固定latent输入；**原H3 decoder仍非因果**，上游编码也不在本轮完整审计范围内，因此不宣称RGB端到端零前视。upscaler-only时延不包含VAE、IO和完整pipeline。需要跨形状的数值一致性时，建议流式推理关闭卷积HF32，并对实际部署形状重新审计；目前的通过范围仅是明确列出的模式、形状与checkpoint，不保证所有NPU模式或任意输入形状都通过。

## 环境与运行

继承[CMN hybrid](../h3_cmn_hybrid_20261105/README.md)的环境、像素helper和既有源码链，不复制或改动之前的实验。配置 `EXP_ROOT`、`H3_BASE_MODEL`、`H3_UPSTREAM_DIR`（包含`open_h3`）、`H3_DIFFUSERS_SRC`，并按设备需要配置`H3_CANN_ENV`、`H3_PYTHON_BIN`及`H3_FFMPEG_BIN`。运行时需兼容H3的patched diffusers，以及torch、torch_npu、torchvision、lpips、NumPy、Pillow、Matplotlib、FFmpeg；LPIPS AlexNet与线性权重应预先缓存。

```bash
source ./env.sh
# 5仅是示例；先检查锁和npu-smi，使用明确空闲的物理卡。
bash "$EXP_CODE_DIR/run_h3_conv3d_hybrid_preflight.sh" 5
# 只有对应审计与预检真正通过，才能启动正式pipeline：
bash "$EXP_CODE_DIR/run_h3_conv3d_hybrid_pipeline.sh" 5
```

当前入口按顺序保留默认精度长序列失败及其非流式因果检查，再运行HF32off严格诊断和像素预检；只有所需JSON、checkpoint与源码合同均一致且通过，pipeline才启动正式训练。原先截至42步的进度属于已中断的attempt1，不能当作attempt2进度；入口本身不作为训练完成证据。新运行拒绝覆盖既有目录；不要用pipeline重复启动已有训练。训练恢复需要原始运行目录、相同源码/配置/数据/精度合同与optimizer/RNG，不声称bitwise resume已验证。旧checkpoint、模型、缓存latent、完整历史评估和teacher cache均在外部；本仓库精简JSON/视频不能代替外部tensor。

`H3_CONV3D_HYBRID_HOME`默认为`$EXP_ROOT/upscaler_conv3d_hybrid`；`H3_CONV3D_INITIAL_CHECKPOINT`为旧causal eWarp的best.pt；原版upscaler权重由`H3_UPSCALER_WEIGHTS`指定，pairs由`H3_ULTRAFLASH_PAIRS`指定，teacher缓存继承`H3_HYBRID_TEACHER_CACHE`。所有计算入口继续通过advisory lock和npu-smi检查卡占用。

## 源码合同与发布边界

[source_fingerprints.json](source_fingerprints.json)记录实际运行目录中源码的SHA、公开SHA及每项修改。公开trainer使用`module_source(name)`对实际导入文件定位，在建立合同和训练结束时都校验继承helper；原始运行版本用同目录文件，因此二者的trainer SHA会不同。公开模型路径读取环境变量；shell改为公开helper链与显式路径变量。继承helper的位置与SHA也逐项列出。

不得把公开SHA回填进原训练合同，也不得修改历史checkpoint绕过校验。原运行复核须使用原执行源码和原数据合同；公开版本的新运行应建立自身合同，历史评估及teacher缓存须与该运行的源码链一致。阶段性失败与后续修正分别保存，未通过的诊断不会标成通过。不发布模型权重、训练缓存、原始源视频、私有绝对路径或凭据。
