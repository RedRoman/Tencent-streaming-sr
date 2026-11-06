# 原始teacher、旧causal与v5：同样本局部视觉核对

结论：原始teacher本身已经包含部分明显的面部/眼部结构错误和同期姿态偏差；旧causal与v5还保留这些问题，部分重复线条更突出。因此，不能将所有伪影归咎于本轮motion训练或因果化，也不能因teacher同样有问题就认定因果改造没有额外损失。本轮motion是否解决主要问题的结论仍为负面；以下是对更早的误差来源范围进行核对，不是为本轮质量作辩护。

这里的teacher是旧六列对照中标为`pretrained`的原始latent upscaler输出，经既有H3 decoder解码；不是新运行完整regenerator/refiner所得的视频。

## 实际查看范围

- 四个clip：0864318人像、0128747大象、0086441卡通、0380588拆巧克力包装。
- 先查看每例已有comparison.png中的f0、60、119；再从同目录原始comparison.mp4按f/24请求提取f16、20、22、60、62，使用已有AVAssetImageGenerator工具、容差0。
- 六列原图顺序：pretrained、old_causal_ewarp、cmn、cmn_hybrid、conv3d_hybrid(v5)、decoded HR reference。局部拼图仅取第0、1、4、5列，未缩放、锐化或增强，只裁切与添加列/帧标签。查看工具对拆包装拼图做了显示缩小，其余局部图未提示缩小。
- 这是有限帧/局部的实际图像观察，不是完整视频实时播放，更不是全35clip主观评测；不能据此评价全片闪烁或稳定排名。视频展示有压缩，不代替未压缩域指标。

## 逐例观察

|clip|teacher实际可见问题|旧causal/v5相对关系|可支持的判断|
|---|---|---|---|
|0864318，f16/20/22/60/62|口鼻与嘴唇处仍有重复/扭曲线条，眼周偏糊；f20/22同期reference明显张嘴，teacher嘴形不同。|old causal和v5的面部重复轮廓、嘴边皱折状纹路更突出；v5与teacher都有同期表情差异。|teacher已经有面部结构/表情误差；因果化相关改动可能额外放大，但这不是单因素实验，不能量化归因。|
|0086441，f16/20/22/60/62|帽子红色符号已变形，猴子眼部有多余细线；f22 reference护士猴闭眼，teacher仍睁眼/出现杂线。|old causal与v5在同位置出现更重的眼周重复线条与符号错形，手/蛋/篮子边缘偏软。|原teacher也没保住同期眼睛开闭；这类错误不始于本轮motion，也不能全部解释成因果化新增。|
|0128747，f16/20/22/60/62|耳部褶皱、鼻尖和前脚的形态/位置与reference仍不同，主体可辨；周围叶片细节粘连。|old causal与v5继承这些偏差，部分耳缘/鼻部细节较糊；所看帧没有证据支持把所有形变算作新causal独有。|teacher存在结构和纹理误差，但不足以单独定位到输入latent、上采样器或decoder。|
|0380588，f16/20/22/60/62|手指/指甲和锡纸高频细节偏软、部分折角与reference不同，整体手形完整。|old causal/v5也保留相似偏差，三个输出之间差异较小。|这些被查看帧里不能称teacher出现严重手部结构崩坏；不将所有软化都夸大成严重形变。|

## 现有本地图像/视频

以下目录均已有comparison.png、comparison.mp4、comparison.h264.mp4；无需下载新缓存或运行NPU。

- 0864318：`${EXPERIMENT_ROOT}/upscaler_conv3d_hybrid/evaluation_test16_extension/decoded/VIDGEN5S_0864318/`
- 0128747：`${EXPERIMENT_ROOT}/upscaler_conv3d_hybrid/evaluation/decoded/VIDGEN5S_0128747/`
- 0086441：`${EXPERIMENT_ROOT}/upscaler_conv3d_hybrid/evaluation/decoded/VIDGEN5S_0086441/`
- 0380588：`${EXPERIMENT_ROOT}/upscaler_conv3d_hybrid/evaluation/decoded/VIDGEN5S_0380588/`

本次只读观察产生的临时裁切证据：`${SCRATCH}/h3_teacher_v5_visual_check_20261106/{0864318,0128747,0086441,0380588}/teacher_oldcausal_v5_reference_contact.png`。
裁切坐标按单列672×412、含顶部标签计算：0864318=(235,35,430,300)，0128747=(210,80,465,395)，0086441=(290,90,620,355)，0380588=(105,90,585,385)。

## 归因限制

已有teacher与causal都经过相同既有decoder，但这不是完整消融。共同错误可能涉及LR输入latent信息损失、latent时间对应、teacher上采样自身误差或解码；不能仅凭这几张对照图选定原因。不同输出的“表情与同期reference不同”是观察事实，尚不能直接下结论为时间错位。下一步诊断应单独对齐输入、teacher latent、causal latent与decoder输出的同一时刻，而不是继续仅用全局平均指标宣称伪影改善。
