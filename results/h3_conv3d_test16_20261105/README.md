# 固定step1800：H3 Conv3d test16扩展结果

本轮没有训练或重新选模：13条新增像素评估与旧3条组成全部16条test，1条旧legacy单独报告；重跑anchor通过历史一致性校验且不重复计入均值。测试集已复用，不是盲测。

以下为16条等权平均。主参考为decoded HR，次参考为source RGB；每clip全120帧，PSNR取每clip PSNR的均值。

## vs_decoded_reference

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 26.4083 | 0.131143 | 0.0016295 |
| old_causal_ewarp | 24.3314 | 0.200394 | 0.0023937 |
| cmn | 20.7444 | 0.373120 | 0.0034168 |
| cmn_hybrid | 21.2410 | 0.307227 | 0.0030357 |
| conv3d_hybrid | 25.5670 | 0.131721 | 0.0019088 |

相对旧causal eWarp提升 / 持平 / 退步：psnr_db: 16 / 0 / 0；lpips_alex_v0_1: 16 / 0 / 0；temporal_difference_mse: 16 / 0 / 0

## vs_source_rgb_secondary

| 模型 | PSNR ↑ | LPIPS ↓ | 时间差分MSE ↓ |
| --- | ---: | ---: | ---: |
| pretrained | 25.9717 | 0.140243 | 0.0018464 |
| old_causal_ewarp | 24.0302 | 0.207775 | 0.0025935 |
| cmn | 20.5871 | 0.378067 | 0.0036020 |
| cmn_hybrid | 21.1062 | 0.316272 | 0.0032193 |
| conv3d_hybrid | 25.2658 | 0.143361 | 0.0021142 |

相对旧causal eWarp提升 / 持平 / 退步：psnr_db: 16 / 0 / 0；lpips_alex_v0_1: 16 / 0 / 0；temporal_difference_mse: 16 / 0 / 0

完整16条test中，teacher平均PSNR为26.4083dB、LPIPS为0.131143；新模型为25.5670dB、0.131721。新模型有16/16条PSNR、16/16条像素时间差分MSE不及teacher；LPIPS仅4/16条更好，12/16条更差。完整test结果没有超过teacher。

此前16条test的latent eWarp L1仍为旧causal 0.225486 → 新模型 0.245695（+8.96%，退步）。这是SHA绑定的上一轮指标沿用，本轮没有重训或重选；像素时间差分MSE与latent eWarp是不同指标，像素改善不能覆盖该退步。

全16、新增13、旧3、每个类别及legacy单列的双参考、prefix102/tail18与逐clip差值见[完整指标表](metrics_detail.md)及[独立重算JSON](aggregate_recomputed.json)。所有新增片都公开展示，没有按效果筛选。

新Conv3d/CMN/CMN hybrid复用历史SHA绑定prediction；14条teacher重新推理，旧causal eWarp14条独立checkpoint输出与缓存比较，不代表14次checkpoint重载。上一轮19条样本的独立checkpoint预测校验与因果审计属于历史证据。decoder仍非因果，不能把新增像素结果写成端到端流式因果证明。

14重算例的98个原始MP4完成全片技术审计；13新增例的26段发布H264均120帧/24fps/5秒。六列比较展示由4032×412缩放到3840×392，单列672×384不变；该缩放与压缩不参与指标，原始RGB指标保持原样。原始/发布SHA及辅助工具关系分别保留。

## 视频

| clip | 六列比较 | 新模型 |
| --- | --- | --- |
| VIDGEN5S_0014509 | [comparison](VIDGEN5S_0014509/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0014509/conv3d_hybrid.h264.mp4) |
| VIDGEN5S_0059416 | [comparison](VIDGEN5S_0059416/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0059416/conv3d_hybrid.h264.mp4) |
| VIDGEN5S_0100402 | [comparison](VIDGEN5S_0100402/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0100402/conv3d_hybrid.h264.mp4) |
| VIDGEN5S_0184359 | [comparison](VIDGEN5S_0184359/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0184359/conv3d_hybrid.h264.mp4) |
| VIDGEN5S_0288595 | [comparison](VIDGEN5S_0288595/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0288595/conv3d_hybrid.h264.mp4) |
| VIDGEN5S_0361101 | [comparison](VIDGEN5S_0361101/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0361101/conv3d_hybrid.h264.mp4) |
| VIDGEN5S_0362565 | [comparison](VIDGEN5S_0362565/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0362565/conv3d_hybrid.h264.mp4) |
| VIDGEN5S_0441001 | [comparison](VIDGEN5S_0441001/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0441001/conv3d_hybrid.h264.mp4) |
| VIDGEN5S_0496359 | [comparison](VIDGEN5S_0496359/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0496359/conv3d_hybrid.h264.mp4) |
| VIDGEN5S_0602499 | [comparison](VIDGEN5S_0602499/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0602499/conv3d_hybrid.h264.mp4) |
| VIDGEN5S_0632078 | [comparison](VIDGEN5S_0632078/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0632078/conv3d_hybrid.h264.mp4) |
| VIDGEN5S_0820879 | [comparison](VIDGEN5S_0820879/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0820879/conv3d_hybrid.h264.mp4) |
| VIDGEN5S_0864318 | [comparison](VIDGEN5S_0864318/comparison.h264.mp4) | [Conv3d](VIDGEN5S_0864318/conv3d_hybrid.h264.mp4) |

人工抽样检查合计覆盖全部13条新增clip，每条检查frame0、60、119；部分样例另看原生分辨率裁切。未逐帧人工观看完整动态，以下不构成正式主观评分。

人工检查范围：查看新增13例中前7例的原始comparison.png，每例包含frame0、60、119；另对0184359、0288595、0361101的frame60做原生分辨率四列裁切查看。列顺序为teacher、old causal、CMN、CMN hybrid、新Conv3d、decoded HR reference。以下是抽样帧观察，不是逐帧人工观看或正式主观评分；视频完整解码属于另一项技术检查。

- VIDGEN5S_0014509，田野：新Conv3d保留田垄走向和远处树的轮廓，与teacher更接近；两版CMN的树和田垄有明显叠影。新模型草丛和土壤细节仍比reference平滑。
- VIDGEN5S_0059416，暗处楼梯人物：新Conv3d的人物剪影、衣服条纹及墙上涂画较连贯，两版CMN有人物轮廓和墙面纹理的叠影。暗部细节仍损失，不能由这三帧断言整个视频没有闪烁。
- VIDGEN5S_0100402，石拱门下两名动画人物：新Conv3d保留眼睛、手和石头的主要形状，接近teacher；旧causal较软，两版CMN的人脸及手出现明显重影。头发与背景叶片细节仍不及reference。
- VIDGEN5S_0184359，叶片上蜘蛛：新Conv3d较旧causal改善腿部及胸部轮廓，但细毛、眼缘和腿部高光仍有软化和边缘光晕；reference更清楚，不能称恢复原始细节。该例新模型相对teacher的LPIPS仍较差。
- VIDGEN5S_0288595，厨房冰箱前人物：新Conv3d较旧causal减少手臂与瓶罐边缘杂乱，人物和冰箱整体结构保持；毛发、手指和冰箱小物仍不如reference清楚。它是新增集中相对旧causal PSNR改善最大的样例，同时也是相对teacher LPIPS落后最多的新增样例，二者不矛盾。
- VIDGEN5S_0361101，手上的蜥蜴：新Conv3d的眼部、头部及手的轮廓较两版CMN清楚，与teacher接近；眼睛、细鳞片和皮肤纹理仍比reference模糊，旧causal和新模型间的视觉差异比CMN差异小。
- VIDGEN5S_0362565，早餐动画：新Conv3d能保留眼睛、手、壶和饼的边界，两版CMN明显叠影；绒毛等高频纹理仍较平滑。观察到的是空间画面差异，未据此声称运动完全无伪影。

这批抽样帧支持新Conv3d明显优于两版CMN、较旧causal更接近teacher的方向，但不支持超过teacher或达到reference的结论。完整16例数值比较、相对teacher的退步和原latent eWarp的退步应与这些观察一同阅读。

# 新增测试片段的抽帧视觉检查

已实际查看以下六例的原始 `comparison.png`。每例只检查第 0、60、119 帧；六列依次为 teacher、旧 causal eWarp、CMN、CMN hybrid、新 Conv3d hybrid、decoded HR reference。此检查不代表观看完整视频，不能据此判断闪烁、瞬时伪影或运动连续性。

- **VIDGEN5S_0441001（蓝色毛发怪物与铁链）**：新 Conv3d 的身体轮廓、手爪、红眼与铁链明显比两个 CMN 清楚，后两者有较重的涂抹与重边。相对旧 causal，改善较细微；毛发仍偏平，部分链环黏连，HR 的毛束和链环分离更清晰。
- **VIDGEN5S_0496359（遥控直升机）**：新模型的机身、起落架、尾杆较完整，CMN 两列有明显红黑重边。相对旧 causal，机头和窗框略干净，但尾旋翼的小几何形状、旋翼轴和座舱细节仍失真或发软。抽帧中的旋翼模糊也可能来自原本运动，不能直接归因于模型时序问题。
- **VIDGEN5S_0602499（隧道、电缆与轨道）**：新模型保留较清楚的透视与线条，较 CMN 两列的重复电缆、波纹状边缘好很多。旧 causal 与新模型差距较小；远处轨道、电缆仍糊成一团，墙面接缝和灯具边缘不如 HR 清楚。
- **VIDGEN5S_0632078（戴墨镜、黑色 KNIGHT 上衣的人物）**：新模型的人脸轮廓、墨镜和胸前文字比 CMN 清晰，文字基本可辨。相对旧 causal 主要是边缘稍整洁，脸颊、嘴部和墨镜附近仍有刻线般的纹理与细重边；字形和背景树叶仍不如 HR 自然。
- **VIDGEN5S_0820879（床边穿白色印字上衣的人物）**：新模型的头肩、手、电脑和床沿更稳定成形，较 CMN 明显改善；但眼睛、嘴和脸部纹理仍不自然，尤其第 60 帧。上衣字句仍破碎、黏连，HR 中的字句清楚；床单和皮肤细节也偏平。不能把总体指标改善理解为已恢复准确的人脸和文字。
- **VIDGEN5S_0864318（草地上人物与牛群）**：新模型的人物与手臂轮廓完整，较 CMN 的重边和水彩状背景好很多；相对旧 causal 的变化较温和。眼口仍有线状伪影，小牛的腿、花纹与草地细节仍模糊黏连；这些抽帧中 teacher 的脸部通常更干净。

总体看，新 Conv3d hybrid 对比两个小 CMN 的结构和重边改善很明显；对比旧 causal eWarp 更像局部清理，并非每处都有显著变化。人脸、字形、细小机械结构、链环和密集线条仍是主要残留问题，HR reference 依然明显更清楚。上述判断来自抽帧目测，未用汇总指标替代视觉检查。

[源码/环境](../../experiments/h3_conv3d_test16_20261105/README.md)、[原始公开副本](summary.json)、[发布SHA映射](source_fingerprints.json)、[转换记录](h264_conversion_manifest.json)。旧训练、源码、结果和manifest均保持原字节；不发布权重或缓存。
