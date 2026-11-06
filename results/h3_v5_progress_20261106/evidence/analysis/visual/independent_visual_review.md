# 独立局部连续帧观察

结论：这4个针对性样本仍保留明显的运动重影或结构/纹理误差。v5、control、motion三个输出在所选局部很相近；我没有看到足够稳定、足够显著的目视证据，能给motion相对control作出明确的整体优劣排序。完整测试的数值改善不能直接表述为“高速运动伪影已明显消除”。这里也不能把三个模型共有的问题归因于本轮新增损失。

本记录由独立审核代理实际调用view_image查看图像后写下；没有修改metrics、模型或运行源码，没有启动NPU、重新生成视频或下载数据。

## 查看范围与方法

- 样本：dev奔马0421364、dev拨弦0665341、test大象0128747、test卡通0086441。选择针对明显运动及已有LPIPS退步例，非随机或盲评。
- 每例查看零基帧14–23（0.5833–0.9583秒）及58–62（2.4167–2.5833秒），共60个四列画面。17–21为blend帧；16→17与21→22分别覆盖进入/退出边界；58–62均为nonblend。
- 从已有comparison_h264.mp4以AVAssetImageGenerator精确时间容差0提取，工具返回时间均为frame/24。四列固定为v5、control、motion、decoded HR reference。先查看每例完整f16，再逐行查看12张局部连续帧拼图。
- 提取帧为2688×412，每列672×412含顶部文字。局部裁切和拼图未主动缩放、无锐化或增强；阅读工具对部分整张拼图略缩放。因此这里不作像素级微差排名。
- 这是对选定连续帧的逐帧观察，不是35条视频完整实时播放的主观评测；也不把技术解码通过视为主观通过。

## 具体观察

| 样本与帧 | 实际看见的现象 | 对motion的判断 |
|---|---|---|
| dev 0421364，f14–15、f18–20 | 三个输出的马腿附近都有多条细的半透明轮廓；f18–19腿部部分边缘与地面融在一起。参考列的主要腿形、伸展姿态清楚得多。 | motion没有消除这些重影；三输出间差异远小于它们与参考列的差异。 |
| dev 0421364，f58–62，尤其f60–61 | 在nonblend区间仍可见前后腿及马嘴附近叠加的条纹/轮廓。参考中的单个腿姿势在三个输出中变为多重边缘。 | 可见伪影并不限于blend进入/退出边界；不能仅凭本观察认定修blend即能解决。未观察到motion有明确、稳定的局部优势。 |
| dev 0665341，f14–23 | 实际是人物拨弦。三个输出眼睛/鼻口附近有重叠边缘，局部像多画了眼线；f19–21拨弦手、琴弦与指板边缘出现重复细线。参考面部轮廓和琴弦分离更自然。 | motion仍有相同种类的问题；不能称“面部与手部重影已修复”。 |
| dev 0665341，f58–62 | 嘴角/鼻下的重复轮廓和拨弦手的模糊叠影继续存在，说明这不是只出现在第一段blend。参考眼睛朝向与面部轮廓也不同。 | 三模型之间没有足够清楚的目视差异支持稳定排序；不把参考列自身的自然运动模糊全部称为错误。 |
| test 0128747，f14–23、f58–62 | 大象主体完整可辨，但三个输出的象鼻横纹、耳缘褶皱与腿/鼻尖位置和参考列仍有差别，周围叶片细节较粘连。f58–62的鼻尖与前脚局部并未恢复成参考形态。 | motion没有明显修复这类结构与纹理差距。与control之间的微小细节差异不足以作可靠视觉优劣判断；不把该例已知LPIPS退步直接等同于每一帧都更差。 |
| test 0086441，f14–23，尤其f22–23 | 三个输出中护士帽红色符号和猴子眼部存在重复/错乱细线，女孩伸手轮廓也较模糊。参考f22–23护士猴眼睛闭合，而输出仍有错乱的眼部边缘。 | motion保留了这些错形；原模型共有误差没有被本轮明显解决。 |
| test 0086441，f58–62 | 面部/帽子符号和女孩手臂的细节偏差继续存在，整体构图稳定。 | 未见独立证据证明motion在此局部稳定胜过control；同样不据这几帧断言它全片更差。 |

## 连续性与局限

所看f16→17和f21→22没有出现全画面突变或新场景切换，但局部伪影持续跨越边界；这只排除了这几个被看位置上的明显整帧跳变。非blend样本中的残留说明误差不能全部解释成拼接边界。当前观察没有新增低分辨率输入、原双向teacher或decoder中间结果对照，不能据此确定误差来自upscaler、输入latent还是decoder，也不能断言因果化是唯一原因。

参考列是decoded HR reference，并非直接查看源RGB。观察图来自H264展示转码，不能用于替代未压缩域指标。单个已知失败例和短片段不支持全数据集主观优劣结论；本记录有意保留“未能可靠判别”的结果，不为平均指标的提升寻找未经观察的视觉证据。

## 本地观察证据

裁切坐标均相对于单个672×412比较列，含顶部文字坐标：奔马(150,130,530,380)，拨弦(190,35,530,395)，大象(170,70,480,395)，卡通(160,90,620,350)。每例的boundary_a、boundary_b、nonblend拼图分别覆盖14–18、19–23、58–62。

- dev/VIDGEN5S_0421364：展示视频SHA256 `1074e846e17b960d236c5ffe7325caf59dd6123244d727fc46b122d84d8c9805`。观察拼图：[f14–18](independent_frames/VIDGEN5S_0421364/boundary_a_crop_contact.png)、[f19–23](independent_frames/VIDGEN5S_0421364/boundary_b_crop_contact.png)、[f58–62](independent_frames/VIDGEN5S_0421364/nonblend_crop_contact.png)。
- dev/VIDGEN5S_0665341：展示视频SHA256 `b39aa23190c9e79bb650f0d21ce161a8be02f6f1146539d28f213ecda9ab013c`。观察拼图：[f14–18](independent_frames/VIDGEN5S_0665341/boundary_a_crop_contact.png)、[f19–23](independent_frames/VIDGEN5S_0665341/boundary_b_crop_contact.png)、[f58–62](independent_frames/VIDGEN5S_0665341/nonblend_crop_contact.png)。
- test/VIDGEN5S_0128747：展示视频SHA256 `1cf9ddcb35fff5a9ebbe5ac9fd6f5aa619a2d8c814ab4802d007c1c7701442b4`。观察拼图：[f14–18](independent_frames/VIDGEN5S_0128747/boundary_a_crop_contact.png)、[f19–23](independent_frames/VIDGEN5S_0128747/boundary_b_crop_contact.png)、[f58–62](independent_frames/VIDGEN5S_0128747/nonblend_crop_contact.png)。
- test/VIDGEN5S_0086441：展示视频SHA256 `041529cf30bfd4c17336932606b6a82fefdc08cf6219214c5fa6e83095153996`。观察拼图：[f14–18](independent_frames/VIDGEN5S_0086441/boundary_a_crop_contact.png)、[f19–23](independent_frames/VIDGEN5S_0086441/boundary_b_crop_contact.png)、[f58–62](independent_frames/VIDGEN5S_0086441/nonblend_crop_contact.png)。

提取工具SHA256：`76f6a12b7fcad4956b540aae871853dccbadb8bdcfef85b93e67779957f7bff2`。记录时间：2026-11-06T02:20:19.152386+00:00。
