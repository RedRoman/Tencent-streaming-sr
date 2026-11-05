# H3 Conv3d：固定step1800的test16像素扩展

本目录是独立扩展评估源码快照。固定原step1800权重，不训练、不重新选模；新增13条既有test，另重跑1条旧test作为一致性anchor。all16使用13新增+3个旧原记录，anchor不双计；旧1条legacy单独报告。测试集已复用，不是盲测。

原Conv3d/CMN/CMN hybrid复用历史SHA绑定prediction。Teacher重推14条；旧causal eWarp使用独立加载的checkpoint推理14条并比对缓存，**14次输出比对不代表14次模型重载**。上一轮19条样本的独立checkpoint预测校验及因果审计属于历史证据，不能写成本轮新增检查。decoder、LPIPS、数学模式与指标定义保持原合同；98个原始MP4审计覆盖14例×7视频。

## 环境与源代码合同

本目录env继承既有h3_conv3d_hybrid目录，继续要求EXP_ROOT、H3_BASE_MODEL、H3_UPSTREAM_DIR、H3_DIFFUSERS_SRC等外部模型/运行环境；模型、训练缓存和源视频不发布。新Python使用module_source定位实际继承模块，wrapper复用既有锁助手。先设置环境变量，再source本目录env.sh；参数可用python evaluate_h3_conv3d_test16.py --help查看。

原执行源码和公开路径版的SHA不同，source_fingerprints分别记录，**原历史合同内部SHA保持原意，不替换或绕过核验**。原inputs/verify_history会严格比对所有历史源文件。要精确重放本次已有checkpoint和原始运行合同，必须使用保留的原执行源码链与环境；公开路径版独立运行需要其自身匹配的历史合同。直接把公开路径版接到原始合同会因SHA不同而拒绝，这是预期行为，不能宣称此目录是可直接重放原实验的无依赖包。

## 独立展示转换

convert_display_videos.py只创建H264展示副本，不参与指标计算，不覆盖已有文件。可用macOS Swift编译audit_h264.swift，再通过--auditor传入编译文件；该审计逐帧解码检查120帧/24fps/5秒。比较图视频允许并记录4032×412→3840×392展示缩放，单列672×384不变。每次转换保存独立receipt；源与输出SHA被绑定。转换工具和审计工具是独立发布辅助代码，不属于NPU模型运行源码合同。

最终质量结论只见完成后的[结果目录](../../results/h3_conv3d_test16_20261105/README.md)。源码公开本身不代表评估已完成或效果提升。
